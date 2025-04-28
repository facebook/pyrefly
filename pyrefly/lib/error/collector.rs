/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::fmt;
use std::fmt::Debug;
use std::fmt::Display;

use dupe::Dupe;
use ruff_text_size::TextRange;
use vec1::vec1;

use crate::config::error::ErrorConfig;
use crate::error::context::ErrorContext;
use crate::error::error::Error;
use crate::error::kind::ErrorKind;
use crate::error::style::ErrorStyle;
use crate::module::module_info::ModuleInfo;
use crate::util::lock::Mutex;

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
        self.items.sort();
        self.items.dedup();
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
}

/// Collects the user errors (e.g. type errors) associated with a module.
// Deliberately don't implement Clone,
#[derive(Debug)]
pub struct ErrorCollector {
    module_info: ModuleInfo,
    style: ErrorStyle,
    errors: Mutex<ModuleErrors>,
}

impl Display for ErrorCollector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for err in self.errors.lock().iter() {
            writeln!(f, "ERROR: {err}")?;
        }
        Ok(())
    }
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

    pub fn add(
        &self,
        range: TextRange,
        msg: String,
        kind: ErrorKind,
        context: Option<&dyn Fn() -> ErrorContext>,
    ) {
        let source_range = self.module_info.source_range(range);
        let is_ignored = self.module_info.is_ignored(&source_range, &msg);
        let full_msg = match context {
            Some(ctx) => vec1![ctx().format(), msg],
            None => vec1![msg],
        };
        if self.style != ErrorStyle::Never {
            let err = Error::new(
                self.module_info.path().dupe(),
                source_range,
                full_msg,
                is_ignored,
                kind,
            );
            self.errors.lock().push(err);
        }
    }

    pub fn module_info(&self) -> &ModuleInfo {
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
                if err.is_ignored() {
                    result.suppressed.push(err.clone());
                } else if !error_config.display_config.is_enabled(err.error_kind()) {
                    result.disabled.push(err.clone());
                } else {
                    result.shown.push(err.clone());
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
    use std::sync::Arc;

    use ruff_python_ast::name::Name;
    use ruff_text_size::TextSize;

    use super::*;
    use crate::config::error::ErrorDisplayConfig;
    use crate::module::module_name::ModuleName;
    use crate::module::module_path::ModulePath;
    use crate::util::prelude::SliceExt;

    #[test]
    fn test_error_collector() {
        let mi = ModuleInfo::new(
            ModuleName::from_name(&Name::new_static("main")),
            ModulePath::filesystem(Path::new("main.py").to_owned()),
            Arc::new("contents".to_owned()),
        );
        let errors = ErrorCollector::new(mi.dupe(), ErrorStyle::Delayed);
        errors.add(
            TextRange::new(TextSize::new(1), TextSize::new(3)),
            "b".to_owned(),
            ErrorKind::InternalError,
            None,
        );
        errors.add(
            TextRange::new(TextSize::new(1), TextSize::new(3)),
            "a".to_owned(),
            ErrorKind::InternalError,
            None,
        );
        errors.add(
            TextRange::new(TextSize::new(1), TextSize::new(3)),
            "a".to_owned(),
            ErrorKind::InternalError,
            None,
        );
        errors.add(
            TextRange::new(TextSize::new(2), TextSize::new(3)),
            "a".to_owned(),
            ErrorKind::InternalError,
            None,
        );
        errors.add(
            TextRange::new(TextSize::new(1), TextSize::new(3)),
            "b".to_owned(),
            ErrorKind::InternalError,
            None,
        );
        assert_eq!(
            errors
                .collect(&ErrorConfig::new(&ErrorDisplayConfig::default(), false))
                .shown
                .map(|x| x.msg()),
            vec!["a", "b", "a"]
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
        errors.add(
            TextRange::new(TextSize::new(1), TextSize::new(3)),
            "a".to_owned(),
            ErrorKind::InternalError,
            None,
        );
        errors.add(
            TextRange::new(TextSize::new(1), TextSize::new(3)),
            "b".to_owned(),
            ErrorKind::AsyncError,
            None,
        );
        errors.add(
            TextRange::new(TextSize::new(1), TextSize::new(3)),
            "c".to_owned(),
            ErrorKind::BadAssignment,
            None,
        );
        errors.add(
            TextRange::new(TextSize::new(2), TextSize::new(3)),
            "d".to_owned(),
            ErrorKind::MatchError,
            None,
        );
        errors.add(
            TextRange::new(TextSize::new(1), TextSize::new(3)),
            "e".to_owned(),
            ErrorKind::NotIterable,
            None,
        );

        let display_config = ErrorDisplayConfig::new(HashMap::from([
            (ErrorKind::AsyncError, true),
            (ErrorKind::BadAssignment, false),
            (ErrorKind::NotIterable, false),
        ]));
        let config = ErrorConfig::new(&display_config, false);

        assert_eq!(
            errors.collect(&config).shown.map(|x| x.msg()),
            vec!["b", "a", "d"]
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
        errors.add(
            TextRange::new(TextSize::new(1), TextSize::new(3)),
            "a".to_owned(),
            ErrorKind::InternalError,
            None,
        );

        let display_config = ErrorDisplayConfig::default();
        let config0 = ErrorConfig::new(&display_config, false);
        assert_eq!(errors.collect(&config0).shown.map(|x| x.msg()), vec!["a"]);

        let config1 = ErrorConfig::new(&display_config, true);
        assert!(errors.collect(&config1).shown.map(|x| x.msg()).is_empty());
    }
}
