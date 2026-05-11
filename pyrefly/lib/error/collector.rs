/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::fmt::Debug;
use std::mem;

use dupe::Dupe;
use pyrefly_config::error_kind::ErrorKind;
use pyrefly_python::ignore::Tool;
use pyrefly_util::lined_buffer::LineNumber;
use pyrefly_util::lock::Mutex;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use starlark_map::small_map::SmallMap;
use vec1::Vec1;
use vec1::vec1;

use crate::config::error::ErrorConfig;
use crate::config::error_kind::Severity;
use crate::error::context::ErrorContext;
use crate::error::context::ErrorInfo;
use crate::error::error::Error;
use crate::error::error::ErrorQuickFix;
use crate::error::style::ErrorStyle;
use crate::module::module_info::ModuleInfo;
use crate::state::errors::find_containing_range;

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
    /// Ordinary diagnostics (errors, warnings, info) that passed severity and
    /// suppression filters. These participate in baseline exclusion,
    /// suppression, and min-severity filtering.
    pub ordinary: Vec<Error>,
    /// Directive diagnostics (e.g. `reveal_type`) that are always displayed to
    /// the user. Directives are never subject to baseline exclusion,
    /// suppression, or min-severity filtering.
    pub directives: Vec<Error>,
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

    /// Start building an error. Returns a no-op builder if style is Never.
    pub fn error_builder(
        &self,
        range: TextRange,
        kind: ErrorKind,
        header: String,
    ) -> ErrorBuilder<'_> {
        ErrorBuilder {
            collector: self,
            active: self.style != ErrorStyle::Never,
            range,
            kind,
            header,
            details: Vec::new(),
            context: None,
            annotations: Vec::new(),
            quick_fixes: Vec::new(),
        }
    }

    pub fn add(&self, range: TextRange, info: ErrorInfo, msg: Vec1<String>) {
        self.add_with_annotations_and_quick_fixes(range, info, msg, Vec::new(), Vec::new());
    }

    pub fn add_with_annotations_and_quick_fixes(
        &self,
        range: TextRange,
        info: ErrorInfo,
        mut msg: Vec1<String>,
        annotations: Vec<(TextRange, String)>,
        quick_fixes: Vec<ErrorQuickFix>,
    ) {
        if self.style == ErrorStyle::Never {
            return;
        }
        let (kind, ctx_annotations) = match info {
            ErrorInfo::Context(ctx) => {
                let ctx = ctx();
                let kind = ctx.as_error_kind();
                let ctx_annotations = ctx.annotations();
                msg.insert(0, ctx.format());
                (kind, ctx_annotations)
            }
            ErrorInfo::Kind(kind) => (kind, Vec::new()),
        };
        let mut err = Error::new(self.module_info.dupe(), range, msg, kind);
        for (range, label) in ctx_annotations.into_iter().chain(annotations) {
            err = err.with_annotation(range, label);
        }
        for quick_fix in quick_fixes {
            err = err.with_quick_fix(quick_fix);
        }
        self.errors.lock().push(err);
    }

    pub fn internal_error(&self, range: TextRange, header: String) {
        self.error_builder(range, ErrorKind::InternalError, header)
            .with_detail(
                "Sorry, Pyrefly encountered an internal error, \
                 this is always a bug in Pyrefly itself"
                    .to_owned(),
            )
            .with_detail(
                if cfg!(fbcode_build) {
                    "Please report the bug at https://fb.workplace.com/groups/pyreqa"
                } else {
                    "Please report the bug at https://github.com/facebook/pyrefly/issues/new"
                }
                .to_owned(),
            )
            .emit();
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

    /// Checks whether an error is suppressed, considering ignore-all directives,
    /// per-line suppressions, and (for errors inside multi-line f/t-strings)
    /// suppressions on the f-string's start or end lines.
    fn is_error_suppressed(
        err: &Error,
        fstring_ranges: &[(LineNumber, LineNumber)],
        ignore_all: &SmallMap<Tool, LineNumber>,
        error_config: &ErrorConfig,
    ) -> bool {
        // Check whole-file ignore-all directives first.
        // UnusedIgnore errors cannot be suppressed to prevent infinite loops.
        if err.error_kind() != ErrorKind::UnusedIgnore
            && error_config
                .enabled_ignores
                .iter()
                .any(|tool| ignore_all.contains_key(tool))
        {
            return true;
        }
        if err.is_ignored(&error_config.enabled_ignores) {
            return true;
        }
        // Check if the error is inside a multi-line f/t-string. If so, a
        // suppression that covers the f-string's start or end line should also apply.
        let line = err.display_range().start.line_within_file();
        if let Some((fs_start, fs_end)) = find_containing_range(fstring_ranges, line) {
            let ignore = err.module().ignore();
            let enabled = &error_config.enabled_ignores;
            // Check both this kind's name and any parent kind's name.
            for kind in err.error_kind().suppression_names() {
                if fs_start != line && ignore.is_ignored(fs_start, kind, enabled) {
                    return true;
                }
                if fs_end != line && ignore.is_ignored(fs_end, kind, enabled) {
                    return true;
                }
            }
        }
        false
    }

    pub fn collect_into(
        &self,
        error_config: &ErrorConfig,
        fstring_ranges: &[(LineNumber, LineNumber)],
        ignore_all: &SmallMap<Tool, LineNumber>,
        result: &mut CollectedErrors,
    ) {
        let mut errors = self.errors.lock();
        if !(self.module_info.is_generated() && error_config.ignore_errors_in_generated_code) {
            for err in errors.iter() {
                if err.error_kind().is_directive() {
                    // Directives bypass suppression, baseline, and
                    // min-severity, but still respect explicit severity
                    // overrides (e.g. --ignore reveal-type).
                    let severity = error_config.display_config.severity(err.error_kind());
                    if severity == Severity::Ignore {
                        result.disabled.push(err.clone());
                    } else {
                        result.directives.push(err.with_severity(severity));
                    }
                } else if Self::is_error_suppressed(err, fstring_ranges, ignore_all, error_config) {
                    result.suppressed.push(err.clone());
                } else {
                    match error_config.display_config.severity(err.error_kind()) {
                        Severity::Error => result.ordinary.push(err.with_severity(Severity::Error)),
                        Severity::Warn => result.ordinary.push(err.with_severity(Severity::Warn)),
                        Severity::Info => result.ordinary.push(err.with_severity(Severity::Info)),
                        Severity::Ignore => result.disabled.push(err.clone()),
                    }
                }
            }
        }
    }

    pub fn collect(&self, error_config: &ErrorConfig) -> CollectedErrors {
        let mut result = CollectedErrors::default();
        self.collect_into(error_config, &[], &SmallMap::new(), &mut result);
        result
    }
}

/// A builder for constructing and emitting errors incrementally.
/// Chain decoration methods and call `.emit()` to push the error into the collector.
#[must_use = "errors are not emitted until .emit() is called"]
pub struct ErrorBuilder<'a> {
    collector: &'a ErrorCollector,
    active: bool,
    range: TextRange,
    kind: ErrorKind,
    header: String,
    details: Vec<String>,
    context: Option<ErrorContext>,
    annotations: Vec<(TextRange, String)>,
    quick_fixes: Vec<ErrorQuickFix>,
}

impl ErrorBuilder<'_> {
    /// Append a detail line (shown indented below the header).
    pub fn with_detail(mut self, msg: String) -> Self {
        if self.active {
            self.details.push(msg);
        }
        self
    }

    /// Add a secondary labeled span.
    pub fn with_annotation(mut self, range: TextRange, label: String) -> Self {
        if self.active {
            self.annotations.push((range, label));
        }
        self
    }

    /// Add a structured quick fix.
    pub fn with_quick_fix(mut self, fix: ErrorQuickFix) -> Self {
        if self.active {
            self.quick_fixes.push(fix);
        }
        self
    }

    /// Set the ErrorContext. At emit time, the context's message becomes the header
    /// (demoting the original header to first detail), its annotations are prepended,
    /// and the ErrorKind is overridden. If called more than once, the last context wins.
    /// `with_context(None)` clears the context.
    pub fn with_context(mut self, ctx: Option<impl FnOnce() -> ErrorContext>) -> Self {
        if self.active {
            self.context = ctx.map(|ctx| ctx());
        }
        self
    }

    /// Emit the error into the collector.
    pub fn emit(self) {
        if !self.active {
            return;
        }
        let (mut kind, mut header, mut details, mut annotations) =
            (self.kind, self.header, self.details, self.annotations);
        if let Some(ctx) = self.context {
            kind = ctx.as_error_kind();
            details.insert(0, header);
            header = ctx.format();
            let mut ctx_annotations = ctx.annotations();
            ctx_annotations.extend(annotations);
            annotations = ctx_annotations;
        }
        let msg = if details.is_empty() {
            vec1![header]
        } else {
            let mut v = vec1![header];
            v.extend(details);
            v
        };
        let mut err = Error::new(self.collector.module_info.dupe(), self.range, msg, kind);
        for (range, label) in annotations {
            err = err.with_annotation(range, label);
        }
        for fix in self.quick_fixes {
            err = err.with_quick_fix(fix);
        }
        self.collector.errors.lock().push(err);
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
                .ordinary
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
            errors.collect(&config).ordinary.map(|x| x.msg()),
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
        assert_eq!(
            errors.collect(&config0).ordinary.map(|x| x.msg()),
            vec!["a"]
        );

        let config1 = ErrorConfig::new(&display_config, true, Tool::default_enabled());
        assert!(
            errors
                .collect(&config1)
                .ordinary
                .map(|x| x.msg())
                .is_empty()
        );
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
                .ordinary
                .map(|x| x.msg()),
            vec!["Overload", "A specific error"]
        );
    }
}
