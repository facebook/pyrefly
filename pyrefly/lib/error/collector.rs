/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::fmt::Debug;
use std::mem;

use dupe::Dupe;
use pyrefly_util::lock::Mutex;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use vec1::Vec1;

use crate::config::error::ErrorConfig;
use crate::config::error_kind::Severity;
use crate::error::context::ErrorInfo;
use crate::error::error::Error;
use crate::error::style::ErrorStyle;
use crate::module::module_info::ModuleInfo;

#[derive(Debug, Default, Clone)]
struct ModuleErrors {
    /// Set to `true` when we have no duplicates and are sorted.
    clean: bool,
    items: Vec<Error>,
}

impl ModuleErrors {
    fn push(&mut self, err: Error) {
        self.clean = false;
        self.items.push(err);
    }

    fn extend(&mut self, errs: ModuleErrors) {
        self.clean = false;
        self.items.extend(errs.items);
    }

    fn cleanup(&mut self) {
        if self.clean {
            return;
        }
        self.clean = true;
        // We want to sort only by source-range, not by message.
        // When we get an overload error, we want that overload to remain before whatever the precise overload failure is.
        self.items
            .sort_by_key(|x| (x.range().start(), x.range().end()));

        // Within a single source range we want to dedupe, even if the error messages aren't adjacent
        let mut res = Vec::with_capacity(self.items.len());
        mem::swap(&mut res, &mut self.items);

        // The range and where that range started in self.items
        let mut previous_range = TextRange::default();
        let mut previous_start = 0;
        for x in res {
            if x.range() != previous_range {
                previous_range = x.range();
                previous_start = self.items.len();
                self.items.push(x);
            } else if !self.items[previous_start..].contains(&x) {
                self.items.push(x);
            }
        }
    }

    fn is_empty(&self) -> bool {
        // No need to do cleanup if it's empty.
        self.items.is_empty()
    }

    fn len(&mut self) -> usize {
        self.cleanup();
        self.items.len()
    }

    /// Iterates over all errors, including ignored ones.
    fn iter(&mut self) -> impl ExactSizeIterator<Item = &Error> {
        self.cleanup();
        self.items.iter()
    }
}

#[derive(Debug, Default)]
pub struct CollectedErrors {
    /// Errors that will be reported to the user.
    pub shown: Vec<Error>,
    /// Errors that are suppressed with inline ignore comments.
    pub suppressed: Vec<Error>,
    /// Errors that are disabled with configuration options.
    pub disabled: Vec<Error>,
    /// Errors that are suppressed by baseline file.
    pub baseline: Vec<Error>,
}

/// Collects the user errors (e.g. type errors) associated with a module.
// Deliberately don't implement Clone,
#[derive(Debug)]
pub struct ErrorCollector {
    module_info: ModuleInfo,
    style: ErrorStyle,
    errors: Mutex<ModuleErrors>,
}

impl ErrorCollector {
    pub fn new(module_info: ModuleInfo, style: ErrorStyle) -> Self {
        Self {
            module_info,
            style,
            errors: Mutex::new(Default::default()),
        }
    }

    pub fn extend(&self, other: ErrorCollector) {
        if self.style != ErrorStyle::Never {
            self.errors.lock().extend(other.errors.into_inner());
        }
    }

    pub fn add(&self, range: TextRange, info: ErrorInfo, mut msg: Vec1<String>) {
        if self.style == ErrorStyle::Never {
            return;
        }
        let (kind, ctx) = match info {
            ErrorInfo::Context(ctx) => {
                let ctx = ctx();
                (ctx.as_error_kind(), Some(ctx))
            }
            ErrorInfo::Kind(kind) => (kind, None),
        };
        if let Some(ctx) = ctx {
            msg.insert(0, ctx.format());
        }
        let err = Error::new(self.module_info.dupe(), range, msg, kind);
        self.errors.lock().push(err);
    }

    pub fn module(&self) -> &ModuleInfo {
        &self.module_info
    }

    pub fn style(&self) -> ErrorStyle {
        self.style
    }

    pub fn is_empty(&self) -> bool {
        self.errors.lock().is_empty()
    }

    pub fn len(&self) -> usize {
        self.errors.lock().len()
    }

    pub fn collect_into(&self, error_config: &ErrorConfig, result: &mut CollectedErrors) {
        let mut errors = self.errors.lock();
        if !(self.module_info.is_generated() && error_config.ignore_errors_in_generated_code) {
            for err in errors.iter() {
                if err.is_ignored(&error_config.enabled_ignores) {
                    result.suppressed.push(err.clone());
                } else {
                    match error_config.display_config.severity(err.error_kind()) {
                        Severity::Error => result.shown.push(err.with_severity(Severity::Error)),
                        Severity::Warn => result.shown.push(err.with_severity(Severity::Warn)),
                        Severity::Info => result.shown.push(err.with_severity(Severity::Info)),
                        Severity::Ignore => result.disabled.push(err.clone()),
                    }
                }
            }
        }
    }

    pub fn collect(&self, error_config: &ErrorConfig) -> CollectedErrors {
        let mut result = CollectedErrors::default();
        self.collect_into(error_config, &mut result);
        result
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::path::Path;
    use std::path::PathBuf;
    use std::sync::Arc;

    use pyrefly_python::ignore::Tool;
    use pyrefly_python::module_name::ModuleName;
    use pyrefly_python::module_path::ModulePath;
    use pyrefly_util::prelude::SliceExt;
    use ruff_python_ast::name::Name;
    use ruff_text_size::TextSize;
    use vec1::vec1;

    use super::*;
    use crate::config::error::ErrorDisplayConfig;
    use crate::config::error_kind::ErrorKind;
    use crate::config::error_kind::Severity;

    fn add(errors: &ErrorCollector, range: TextRange, kind: ErrorKind, msg: String) {
        errors.add(range, ErrorInfo::Kind(kind), vec1![msg]);
    }

    #[test]
    fn test_error_collector() {
        let mi = ModuleInfo::new(
            ModuleName::from_name(&Name::new_static("main")),
            ModulePath::filesystem(Path::new("main.py").to_owned()),
            Arc::new("contents".to_owned()),
        );
        let errors = ErrorCollector::new(mi.dupe(), ErrorStyle::Delayed);
        add(
            &errors,
            TextRange::new(TextSize::new(1), TextSize::new(3)),
            ErrorKind::InternalError,
            "b".to_owned(),
        );
        add(
            &errors,
            TextRange::new(TextSize::new(1), TextSize::new(3)),
            ErrorKind::InternalError,
            "a".to_owned(),
        );
        add(
            &errors,
            TextRange::new(TextSize::new(1), TextSize::new(3)),
            ErrorKind::InternalError,
            "a".to_owned(),
        );
        add(
            &errors,
            TextRange::new(TextSize::new(2), TextSize::new(3)),
            ErrorKind::InternalError,
            "a".to_owned(),
        );
        add(
            &errors,
            TextRange::new(TextSize::new(1), TextSize::new(3)),
            ErrorKind::InternalError,
            "b".to_owned(),
        );
        assert_eq!(
            errors
                .collect(&ErrorConfig::new(
                    &ErrorDisplayConfig::default(),
                    false,
                    Tool::default_enabled(),
                ))
                .shown
                .map(|x| x.msg()),
            vec!["b", "a", "a"]
        );
    }

    #[test]
    fn test_error_collector_with_disabled_errors() {
        let mi = ModuleInfo::new(
            ModuleName::from_name(&Name::new_static("main")),
            ModulePath::filesystem(Path::new("main.py").to_owned()),
            Arc::new("contents".to_owned()),
        );
        let errors = ErrorCollector::new(mi.dupe(), ErrorStyle::Delayed);
        add(
            &errors,
            TextRange::new(TextSize::new(1), TextSize::new(3)),
            ErrorKind::InternalError,
            "a".to_owned(),
        );
        add(
            &errors,
            TextRange::new(TextSize::new(1), TextSize::new(3)),
            ErrorKind::NotAsync,
            "b".to_owned(),
        );
        add(
            &errors,
            TextRange::new(TextSize::new(1), TextSize::new(3)),
            ErrorKind::BadAssignment,
            "c".to_owned(),
        );
        add(
            &errors,
            TextRange::new(TextSize::new(2), TextSize::new(3)),
            ErrorKind::BadMatch,
            "d".to_owned(),
        );
        add(
            &errors,
            TextRange::new(TextSize::new(1), TextSize::new(3)),
            ErrorKind::NotIterable,
            "e".to_owned(),
        );

        let display_config = ErrorDisplayConfig::new(HashMap::from([
            (ErrorKind::NotAsync, Severity::Error),
            (ErrorKind::BadAssignment, Severity::Ignore),
            (ErrorKind::NotIterable, Severity::Ignore),
        ]));
        let config = ErrorConfig::new(&display_config, false, Tool::default_enabled());

        assert_eq!(
            errors.collect(&config).shown.map(|x| x.msg()),
            vec!["a", "b", "d"]
        );
    }

    #[test]
    fn test_error_collector_generated_code() {
        let mi = ModuleInfo::new(
            ModuleName::from_name(&Name::new_static("main")),
            ModulePath::filesystem(Path::new("main.py").to_owned()),
            Arc::new(format!("# {}{}\ncontents", "@", "generated")),
        );
        let errors = ErrorCollector::new(mi.dupe(), ErrorStyle::Delayed);
        add(
            &errors,
            TextRange::new(TextSize::new(1), TextSize::new(3)),
            ErrorKind::InternalError,
            "a".to_owned(),
        );

        let display_config = ErrorDisplayConfig::default();
        let config0 = ErrorConfig::new(&display_config, false, Tool::default_enabled());
        assert_eq!(errors.collect(&config0).shown.map(|x| x.msg()), vec!["a"]);

        let config1 = ErrorConfig::new(&display_config, true, Tool::default_enabled());
        assert!(errors.collect(&config1).shown.map(|x| x.msg()).is_empty());
    }

    #[test]
    fn test_errors_not_sorted() {
        let mi = ModuleInfo::new(
            ModuleName::from_name(&Name::new_static("main")),
            ModulePath::filesystem(PathBuf::from("main.py")),
            Arc::new("test".to_owned()),
        );
        let errors = ErrorCollector::new(mi.dupe(), ErrorStyle::Delayed);
        add(
            &errors,
            TextRange::new(TextSize::new(1), TextSize::new(1)),
            ErrorKind::InternalError,
            "Overload".to_owned(),
        );
        add(
            &errors,
            TextRange::new(TextSize::new(1), TextSize::new(1)),
            ErrorKind::InternalError,
            "A specific error".to_owned(),
        );
        assert_eq!(
            errors
                .collect(&ErrorConfig::new(
                    &ErrorDisplayConfig::default(),
                    false,
                    Tool::default_enabled(),
                ))
                .shown
                .map(|x| x.msg()),
            vec!["Overload", "A specific error"]
        );
    }
}
