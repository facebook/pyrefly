/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::path::Path;
use std::sync::Arc;

use dupe::Dupe;
use pyrefly_config::error_kind::ErrorKind;
use pyrefly_config::error_kind::Severity;
use pyrefly_python::ignore::Ignore;
use pyrefly_python::ignore::MalformedReason;
use pyrefly_python::ignore::Tool;
use pyrefly_python::ignore::find_comment_start_in_line;
use pyrefly_python::ignore::parse_ignore_all;
use pyrefly_python::module::Module;
use pyrefly_python::module_path::ModulePath;
use pyrefly_util::arc_id::ArcId;
use pyrefly_util::lined_buffer::LineNumber;
use pyrefly_util::visit::Visit;
use ruff_python_ast::Expr;
use ruff_python_ast::ModModule;
use ruff_python_ast::visitor::Visitor;
use ruff_python_ast::visitor::walk_expr;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;
use starlark_map::small_map::SmallMap;
use starlark_map::small_set::SmallSet;

use crate::config::config::ConfigFile;
use crate::error::baseline::BaselineProcessor;
use crate::error::collector::CollectedErrors;
use crate::error::error::Error;
use crate::error::expectation::Expectation;
use crate::error::style::ErrorStyle;
use crate::state::load::Load;

/// Extracts `(start_line, end_line)` ranges for all multi-line strings from
/// the AST, including plain strings, byte strings, f-strings, and t-strings.
/// Single-line strings (where start == end) are excluded. The returned list
/// is sorted by start_line.
pub fn sorted_multi_line_string_ranges(
    ast: &ModModule,
    module: &Module,
) -> Vec<(LineNumber, LineNumber)> {
    let mut ranges = Vec::new();
    ast.visit(&mut |expr: &Expr| {
        collect_string_ranges(expr, module, &mut ranges);
    });
    ranges.sort();
    ranges
}

/// Recursively collects multi-line string ranges from an expression and all
/// of its sub-expressions. The `Visit<Expr> for Stmt` implementation only
/// visits top-level expressions in statements, so nested strings (e.g., an
/// f-string inside a function call's arguments) would be missed without
/// explicit recursion here.
fn collect_string_ranges(expr: &Expr, module: &Module, ranges: &mut Vec<(LineNumber, LineNumber)>) {
    let text_range = match expr {
        Expr::FString(x) => Some(x.range),
        Expr::TString(x) => Some(x.range),
        Expr::StringLiteral(x) => Some(x.range),
        Expr::BytesLiteral(x) => Some(x.range),
        _ => None,
    };
    if let Some(range) = text_range {
        let display = module.display_range(range);
        let start = display.start.line_within_file();
        let end = display.end.line_within_file();
        if start != end {
            // Multi-line string found. Record its range but skip recursing
            // into its children — for nested f-strings we want errors to
            // remap to the outermost string's start line, and the outer
            // range already covers the inner one.
            ranges.push((start, end));
            return;
        }
    }
    expr.recurse(&mut |child: &Expr| {
        collect_string_ranges(child, module, ranges);
    });
}

/// Finds contiguous backslash continuation blocks in the source lines.
/// A block starts at the first line ending with `\` and ends at the first
/// subsequent line that does NOT end with `\` (inclusive — that line is the
/// last line of the continued expression). Returns sorted, non-overlapping
/// `(start, end)` ranges using 1-indexed `LineNumber`.
///
/// Comments are stripped before checking for trailing backslashes, so
/// `x = 1  # comment \` is not treated as a continuation. Lines inside
/// multiline strings are also excluded, since `\` at end of line inside a
/// triple-quoted string is string content, not a line continuation.
pub fn sorted_backslash_continuation_ranges(
    lines: &[&str],
    multiline_string_ranges: &[(LineNumber, LineNumber)],
) -> Vec<(LineNumber, LineNumber)> {
    /// Returns true if the code portion of `line` (ignoring comments) ends
    /// with a backslash continuation character.
    fn is_continuation(line: &str) -> bool {
        let code = match find_comment_start_in_line(line) {
            Some(pos) => &line[..pos],
            None => line,
        };
        code.trim_end().ends_with('\\')
    }

    let mut ranges = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        let line_num = LineNumber::from_zero_indexed(i as u32);
        if find_containing_range(multiline_string_ranges, line_num).is_some() {
            i += 1;
        } else if is_continuation(lines[i]) {
            let start = i;
            while i < lines.len()
                && is_continuation(lines[i])
                && find_containing_range(
                    multiline_string_ranges,
                    LineNumber::from_zero_indexed(i as u32),
                )
                .is_none()
            {
                i += 1;
            }
            // Include the first line that doesn't end with \ (the tail of
            // the continued expression), if it exists.
            let end = if i < lines.len() { i } else { i - 1 };
            ranges.push((
                LineNumber::from_zero_indexed(start as u32),
                LineNumber::from_zero_indexed(end as u32),
            ));
            i += 1;
        } else {
            i += 1;
        }
    }
    ranges
}

/// Binary search over sorted f-string ranges to find the range containing `line`.
pub fn find_containing_range(
    ranges: &[(LineNumber, LineNumber)],
    line: LineNumber,
) -> Option<(LineNumber, LineNumber)> {
    let idx = ranges.partition_point(|(start, _)| *start <= line);
    if idx == 0 {
        return None;
    }
    let (start, end) = ranges[idx - 1];
    if line >= start && line <= end {
        Some((start, end))
    } else {
        None
    }
}

/// Extracts `(start_line, end_line)` line ranges for multi-line expressions
/// where a suppression comment placed on the line *above* an interior line
/// would be relocated by a formatter (e.g. Black) out of the bracketed group,
/// so the checker would no longer associate it with the error. These are the
/// conditional and operator expressions — ternaries (`a if c else b`), boolean
/// operators (`a and b`), binary operators (`a + b`), and comparisons
/// (`a < b`) — split across physical lines via bracketed continuation. Errors
/// on a non-first line of such a range must be suppressed inline instead.
///
/// Collection literals (dicts, lists, sets, tuples, comprehensions) are
/// intentionally *not* recorded: each element sits on its own line, so a
/// comment on the line above an element is stable under formatting and should
/// be kept. We still recurse into them, so an operator expression used *as* an
/// element (e.g. a multi-line ternary dict value) is still captured.
///
/// The returned list is sorted by start line and non-overlapping: nested
/// expressions (e.g. a binary operator inside a ternary) are merged into a
/// single enclosing range, preserving the binary-search invariant of
/// `find_containing_range`.
pub fn sorted_bracketed_continuation_ranges(
    ast: &ModModule,
    module: &Module,
) -> Vec<(LineNumber, LineNumber)> {
    struct Collector<'a> {
        module: &'a Module,
        ranges: Vec<(LineNumber, LineNumber)>,
    }
    impl<'a> Visitor<'a> for Collector<'a> {
        fn visit_expr(&mut self, expr: &'a Expr) {
            // Conditional/operator expressions float an interior own-line
            // comment when split across lines; collection literals do not.
            if matches!(
                expr,
                Expr::If(_) | Expr::BoolOp(_) | Expr::BinOp(_) | Expr::Compare(_)
            ) {
                let display = self.module.display_range(expr.range());
                let start = display.start.line_within_file();
                let end = display.end.line_within_file();
                if start != end {
                    self.ranges.push((start, end));
                }
            }
            walk_expr(self, expr);
        }
    }
    let mut collector = Collector {
        module,
        ranges: Vec::new(),
    };
    collector.visit_body(&ast.body);
    collector.ranges.sort();

    // Merge overlapping/nested ranges so the result stays non-overlapping and
    // sorted, as `find_containing_range` requires.
    let mut merged: Vec<(LineNumber, LineNumber)> = Vec::new();
    for (start, end) in collector.ranges {
        match merged.last_mut() {
            Some(last) if start <= last.1 => {
                if end > last.1 {
                    last.1 = end;
                }
            }
            _ => merged.push((start, end)),
        }
    }
    merged
}

/// Per-module multi-line ranges and ignore-all directives, computed after parsing.
#[derive(Debug, Clone)]
pub struct ModuleRanges {
    /// Multi-line string and backslash-continuation ranges.
    pub multi_line: Vec<(LineNumber, LineNumber)>,
    /// Top-level ignore-all directives (e.g. `# pyrefly: ignore-errors`).
    pub ignore_all: SmallMap<Tool, LineNumber>,
}

impl ModuleRanges {
    /// Compute multi-line ranges and ignore-all directives from the AST and module source.
    pub fn compute(ast: &ModModule, module_info: &Module) -> Self {
        let mut multi_line = sorted_multi_line_string_ranges(ast, module_info);
        let lines: Vec<&str> = module_info.contents().lines().collect();
        multi_line.extend(sorted_backslash_continuation_ranges(&lines, &multi_line));
        multi_line.sort();
        let ignore_all = parse_ignore_all(module_info.contents(), &multi_line);
        Self {
            multi_line,
            ignore_all,
        }
    }
}

/// The errors from a collection of modules.
#[derive(Debug)]
pub struct Errors {
    // Sorted by module name and path (so deterministic display order)
    loads: Vec<(Arc<Load>, Option<Arc<ModuleRanges>>, ArcId<ConfigFile>)>,
}

impl Errors {
    pub fn new(mut loads: Vec<(Arc<Load>, Option<Arc<ModuleRanges>>, ArcId<ConfigFile>)>) -> Self {
        loads.sort_by_key(|x| (x.0.module_info.name(), x.0.module_info.path().dupe()));
        Self { loads }
    }

    fn merge_display_errors(mut ordinary: Vec<Error>, directives: Vec<Error>) -> Vec<Error> {
        ordinary.extend(directives);
        ordinary.sort_by_cached_key(|e| {
            (
                e.module().name(),
                e.path().dupe(),
                e.range().start(),
                e.range().end(),
            )
        });
        ordinary
    }

    pub fn collect_errors(&self) -> CollectedErrors {
        let mut errors = CollectedErrors::default();
        for (load, module_ranges, config) in &self.loads {
            if load.errors.style() == ErrorStyle::Never {
                continue;
            }
            let error_config = config.get_error_config(load.module_info.path().as_path());
            let ranges = module_ranges
                .as_ref()
                .expect("module_ranges must be present when error style is not Never");
            load.errors.collect_into(
                &error_config,
                &ranges.multi_line,
                &ranges.ignore_all,
                &mut errors,
            );
        }
        errors
    }

    /// Apply baseline filtering to already-collected errors.
    /// `relative_to` is the resolved `--relative-to` directory so that
    /// relative paths stored in the baseline file are resolved correctly.
    pub fn apply_baseline(
        &self,
        mut errors: CollectedErrors,
        baseline_path: Option<&Path>,
        relative_to: &Path,
    ) -> CollectedErrors {
        if let Some(baseline_path) = baseline_path
            && let Ok(processor) = BaselineProcessor::from_file(baseline_path, relative_to)
        {
            processor.process_errors(&mut errors.ordinary, &mut errors.baseline);
        }
        errors
    }

    pub fn collect_display_errors(&self) -> Vec<Error> {
        let errors = self.collect_errors();
        Self::merge_display_errors(errors.ordinary, errors.directives)
    }

    pub fn collect_display_errors_with_ignore_diagnostics(&self) -> Vec<Error> {
        let collected = self.collect_errors();
        let ignore_diagnostics = self.collect_ignore_diagnostics_for_display(&collected);
        let mut ordinary = collected.ordinary;
        ordinary.extend(ignore_diagnostics.ordinary);
        Self::merge_display_errors(ordinary, collected.directives)
    }

    pub fn collect_ignores(&self) -> SmallMap<&ModulePath, &Ignore> {
        let mut ignore_collection: SmallMap<&ModulePath, &Ignore> = SmallMap::new();
        for (load, _, _) in &self.loads {
            let module_path = load.module_info.path();
            let ignores = load.module_info.ignore();
            ignore_collection.insert(module_path, ignores);
        }
        ignore_collection
    }

    /// Collects errors for unused ignore comments.
    /// Returns a vector of errors with ErrorKind::UnusedIgnore for each
    /// suppression comment that doesn't suppress any actual error.
    /// Accepts pre-collected errors to avoid redundant error collection.
    pub fn collect_unused_ignore_errors(&self, collected: &CollectedErrors) -> Vec<Error> {
        let mut unused_errors = Vec::new();

        // Build a map of which error codes were suppressed on each line, keyed by module path.
        // Key: module_path, Value: map from line number to set of suppressed error codes
        let mut suppressed_codes_by_module: SmallMap<
            &ModulePath,
            SmallMap<LineNumber, SmallSet<String>>,
        > = SmallMap::new();

        // Build per-module lookup maps for f-string ranges and enabled ignores.
        // Skip modules that never computed errors — they have no suppressed
        // errors and no module_ranges.
        let fstring_ranges_by_module: SmallMap<&ModulePath, &[(LineNumber, LineNumber)]> = self
            .loads
            .iter()
            .filter(|(load, _, _)| load.errors.style() != ErrorStyle::Never)
            .map(|(load, module_ranges, _)| {
                (
                    load.module_info.path(),
                    module_ranges
                        .as_ref()
                        .expect("module_ranges must be present when error style is not Never")
                        .multi_line
                        .as_slice(),
                )
            })
            .collect();

        let enabled_ignores_by_module: SmallMap<&ModulePath, SmallSet<Tool>> = self
            .loads
            .iter()
            .map(|(load, _, config)| {
                let path = load.module_info.path();
                (path, config.enabled_ignores(path.as_path()).clone())
            })
            .collect();

        for error in &collected.suppressed {
            let module_path = error.path();
            let enabled_ignores = enabled_ignores_by_module
                .get(&module_path)
                .cloned()
                .unwrap_or_else(Tool::default_enabled);
            if error.is_ignored(&enabled_ignores) {
                let module_path = error.path();
                let start_line = error.display_range().start.line_within_file();
                let end_line = error.display_range().end.line_within_file();

                let module_codes = suppressed_codes_by_module.entry(module_path).or_default();

                // Track both this kind's name and any parent kind's name, so that
                // e.g. `# pyrefly: ignore[bad-override]` is not reported as unused
                // when it suppresses a `bad-override-mutable-attribute` error.
                let error_codes: Vec<String> = error
                    .error_kind()
                    .suppression_names()
                    .map(|s| s.to_owned())
                    .collect();

                // Track the error codes for all lines the error spans.
                for line_idx in start_line.to_zero_indexed()..=end_line.to_zero_indexed() {
                    let line_codes = module_codes
                        .entry(LineNumber::from_zero_indexed(line_idx))
                        .or_default();
                    for code in &error_codes {
                        line_codes.insert(code.clone());
                    }
                }

                // If the error is inside a multi-line f/t-string, also track
                // the code at the f-string's start and end lines so that a
                // suppression comment placed there is recognized as "used".
                if let Some(ranges) = fstring_ranges_by_module.get(&module_path)
                    && let Some((fs_start, fs_end)) = find_containing_range(ranges, start_line)
                {
                    for code in &error_codes {
                        module_codes
                            .entry(fs_start)
                            .or_default()
                            .insert(code.clone());
                        module_codes.entry(fs_end).or_default().insert(code.clone());
                    }
                }
            }
        }

        // Iterate over each module and check for unused ignores
        for (load, _, config) in &self.loads {
            let module = &load.module_info;
            let module_path = module.path();
            let ignore = module.ignore();
            let enabled_ignores = config.enabled_ignores(module_path.as_path());

            // Get the suppressed codes for this module (if any)
            let module_suppressed_codes = suppressed_codes_by_module.get(&module_path);

            for (applies_to_line, suppressions) in ignore.iter() {
                for supp in suppressions {
                    let tool = supp.tool();
                    // Only check tools that are enabled and that we support
                    // reporting unused ignores for (Pyrefly and Pyre).
                    if !enabled_ignores.contains(&tool) {
                        continue;
                    }
                    match tool {
                        Tool::Pyrefly | Tool::Pyre => {}
                        _ => continue,
                    }

                    // Get the error codes actually suppressed on this line
                    let used_codes: SmallSet<String> = module_suppressed_codes
                        .and_then(|m| m.get(applies_to_line))
                        .cloned()
                        .unwrap_or_default();

                    // For Tool::Pyre, error code filtering is not enforced
                    // (any Pyre suppression suppresses all errors on the line),
                    // so we only report it as unused when no errors at all were
                    // suppressed on its line.
                    if tool == Tool::Pyre {
                        if !used_codes.is_empty() {
                            continue; // Pyre suppression is used
                        }
                        let comment_line = supp.comment_line();
                        let line_start = module.lined_buffer().line_start(comment_line);
                        let range = TextRange::new(line_start, line_start + TextSize::new(1));
                        unused_errors.push(Error::new(
                            module.dupe(),
                            range,
                            "Unused pyre-fixme comment".to_owned(),
                            Vec::new(),
                            ErrorKind::UnusedIgnore,
                        ));
                        continue;
                    }

                    // Tool::Pyrefly: check individual error codes
                    let declared_codes: SmallSet<String> =
                        supp.error_codes().iter().cloned().collect();

                    // Determine if the suppression is unused
                    let unused_codes: SmallSet<String> = if declared_codes.is_empty() {
                        // Blanket ignore - unused if no errors were suppressed
                        if used_codes.is_empty() {
                            SmallSet::new() // Mark as unused (empty set signals blanket unused)
                        } else {
                            continue; // Used, skip
                        }
                    } else {
                        // Specific codes - find which are unused
                        let unused: SmallSet<String> = declared_codes
                            .iter()
                            .filter(|code| !used_codes.contains(*code))
                            .cloned()
                            .collect();
                        if unused.is_empty() {
                            continue; // All codes used, skip
                        }
                        unused
                    };

                    // Create an error for the unused suppression
                    let comment_line = supp.comment_line();
                    let line_start = module.lined_buffer().line_start(comment_line);
                    let range = TextRange::new(line_start, line_start + TextSize::new(1));

                    let msg = if declared_codes.is_empty() {
                        "Unused `# pyrefly: ignore` comment".to_owned()
                    } else if unused_codes.len() == declared_codes.len() {
                        format!(
                            "Unused `# pyrefly: ignore` comment for code(s): {}",
                            unused_codes.iter().cloned().collect::<Vec<_>>().join(", ")
                        )
                    } else {
                        format!(
                            "Unused error code(s) in `# pyrefly: ignore`: {}",
                            unused_codes.iter().cloned().collect::<Vec<_>>().join(", ")
                        )
                    };

                    unused_errors.push(Error::new(
                        module.dupe(),
                        range,
                        msg,
                        Vec::new(),
                        ErrorKind::UnusedIgnore,
                    ));
                }
            }
        }

        unused_errors
    }

    /// Collects the ignore-comment lint diagnostics that are independent of the
    /// type checker's findings: `invalid-ignore-comment` (#3752) for comments
    /// that look like an ignore directive but are malformed, and
    /// `ignore-without-code` (#3450) for a bare `# pyrefly: ignore` that names no
    /// error code. Both are gated on the relevant tool being enabled, and emitted
    /// at their default severity; `collect_ignore_diagnostics_for_display` applies
    /// the configured per-kind severity.
    pub fn collect_ignore_comment_lint_errors(&self) -> Vec<Error> {
        let mut errors = Vec::new();
        for (load, _, config) in &self.loads {
            // Don't emit lints for files we are told never to report errors on.
            if load.errors.style() == ErrorStyle::Never {
                continue;
            }
            let module = &load.module_info;
            let enabled_ignores = config.enabled_ignores(module.path().as_path());
            let ignore = module.ignore();

            // #3752: comments that look like an ignore directive but are malformed.
            for malformed in ignore.malformed() {
                if !enabled_ignores.contains(&malformed.tool()) {
                    continue;
                }
                let tool = malformed.tool().as_str();
                let msg = match malformed.reason() {
                    MalformedReason::UnknownDirective => format!(
                        "Invalid ignore comment; did you mean `# {tool}: ignore` or `# {tool}: ignore[code]`?"
                    ),
                    MalformedReason::UnterminatedBracket => {
                        format!("Unterminated `[` in `# {tool}: ignore[...]` error-code list")
                    }
                    MalformedReason::EmptyBracket => {
                        format!("Empty error-code list in `# {tool}: ignore[]`")
                    }
                };
                let line_start = module.lined_buffer().line_start(malformed.comment_line());
                let range = TextRange::new(line_start, line_start + TextSize::new(1));
                errors.push(Error::new(
                    module.dupe(),
                    range,
                    msg,
                    Vec::new(),
                    ErrorKind::InvalidIgnoreComment,
                ));
            }

            // #3450: a bare `# pyrefly: ignore` that names no error code. Scoped to
            // Pyrefly, the only tool whose per-line codes Pyrefly honors.
            if enabled_ignores.contains(&Tool::Pyrefly) {
                for (_, suppressions) in ignore.iter() {
                    for supp in suppressions {
                        if supp.tool() == Tool::Pyrefly && supp.error_codes().is_empty() {
                            let line_start = module.lined_buffer().line_start(supp.comment_line());
                            let range = TextRange::new(line_start, line_start + TextSize::new(1));
                            errors.push(Error::new(
                                module.dupe(),
                                range,
                                "`# pyrefly: ignore` has no error code; specify the code(s) being suppressed, e.g. `# pyrefly: ignore[bad-return]`".to_owned(),
                                Vec::new(),
                                ErrorKind::IgnoreWithoutCode,
                            ));
                        }
                    }
                }
            }
        }
        errors
    }

    /// Collects all ignore-comment diagnostics for display, respecting severity
    /// configuration: `unused-ignore`, plus the `invalid-ignore-comment`
    /// and `ignore-without-code` lints. Each error is filtered by its own
    /// kind's configured severity, so kinds set to `Severity::Ignore` land in
    /// `disabled` rather than `ordinary`. Accepts pre-collected errors to avoid
    /// redundant error collection.
    pub fn collect_ignore_diagnostics_for_display(
        &self,
        collected: &CollectedErrors,
    ) -> CollectedErrors {
        let mut ignore_errors = self.collect_unused_ignore_errors(collected);
        ignore_errors.extend(self.collect_ignore_comment_lint_errors());
        let mut result = CollectedErrors::default();

        // Build a path-to-config map for O(1) lookup instead of O(loads) per error.
        let config_by_path: SmallMap<&ModulePath, &ArcId<ConfigFile>> = self
            .loads
            .iter()
            .map(|(load, _, config)| (load.module_info.path(), config))
            .collect();

        for error in ignore_errors {
            // These diagnostics are generated from `self.loads`, so their path is
            // always present in the map built from the same loads.
            let config = config_by_path
                .get(&error.path())
                .expect("ignore diagnostics are generated from loaded modules");
            let error_config = config.get_error_config(error.path().as_path());
            let severity = error_config.display_config.severity(error.error_kind());
            match severity {
                Severity::Error => result.ordinary.push(error.with_severity(Severity::Error)),
                Severity::Warn => result.ordinary.push(error.with_severity(Severity::Warn)),
                Severity::Info => result.ordinary.push(error.with_severity(Severity::Info)),
                Severity::Ignore => result.disabled.push(error),
            }
        }

        result
    }

    pub fn check_against_expectations(&self) -> anyhow::Result<()> {
        for (load, module_ranges, config) in &self.loads {
            if load.errors.style() == ErrorStyle::Never {
                continue;
            }
            let error_config = config.get_error_config(load.module_info.path().as_path());
            let ranges = module_ranges
                .as_ref()
                .expect("module_ranges must be present when error style is not Never");
            let mut result = CollectedErrors::default();
            load.errors.collect_into(
                &error_config,
                &ranges.multi_line,
                &ranges.ignore_all,
                &mut result,
            );
            let output_errors = Self::merge_display_errors(result.ordinary, result.directives);
            Expectation::parse(load.module_info.dupe(), load.module_info.contents())
                .check(&output_errors)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::Arc;

    use dupe::Dupe;
    use pyrefly_build::handle::Handle;
    use pyrefly_config::error_kind::ErrorKind;
    use pyrefly_python::module_name::ModuleName;
    use pyrefly_python::module_path::ModulePath;
    use pyrefly_python::sys_info::SysInfo;
    use pyrefly_util::arc_id::ArcId;
    use pyrefly_util::fs_anyhow;
    use pyrefly_util::thread_pool::TEST_THREAD_COUNT;
    use regex::Regex;
    use tempfile::TempDir;

    use crate::config::config::ConfigFile;
    use crate::config::finder::ConfigFinder;
    use crate::state::errors::Errors;
    use crate::state::load::FileContents;
    use crate::state::require::Require;
    use crate::state::state::State;

    impl Errors {
        pub fn check_var_leak(&self) -> anyhow::Result<()> {
            let regex = Regex::new(r"@\d+").unwrap();
            for (load, _, config) in &self.loads {
                let error_config = config.get_error_config(load.module_info.path().as_path());
                let errors = load.errors.collect(&error_config).ordinary;
                for error in errors {
                    let msg = error.msg();
                    if regex.is_match(&msg) {
                        return Err(anyhow::anyhow!(
                            "{}:{}: variable ids leaked into error message: {}",
                            error.path(),
                            error.display_range(),
                            msg,
                        ));
                    }
                }
            }
            Ok(())
        }
    }

    fn get_path(tdir: &TempDir) -> PathBuf {
        tdir.path().join("test.py")
    }

    fn get_errors(contents: &str) -> (Errors, TempDir) {
        let tdir = tempfile::tempdir().unwrap();

        let mut config = ConfigFile::default();
        config.python_environment.set_empty_to_default();
        let name = "test";
        fs_anyhow::write(&get_path(&tdir), contents).unwrap();
        config.configure();

        let config = ArcId::new(config);
        let sys_info = SysInfo::default();
        let state = State::new(ConfigFinder::new_constant(config), TEST_THREAD_COUNT);
        let handle = Handle::new(
            ModuleName::from_str(name),
            ModulePath::filesystem(get_path(&tdir)),
            sys_info.dupe(),
        );
        let mut transaction = state.new_transaction(Require::Exports, None);
        transaction.set_memory(vec![(
            get_path(&tdir),
            Some(Arc::new(FileContents::from_source(contents.to_owned()))),
        )]);
        transaction.run(&[handle.dupe()], Require::Everything, None);
        (transaction.get_errors([handle.clone()].iter()), tdir)
    }

    #[test]
    fn test_unused_blanket_ignore() {
        // A blanket ignore comment with no errors to suppress
        let contents = r#"
def f() -> int:
    # pyrefly: ignore
    return 1
"#;
        let (errors, _tdir) = get_errors(contents);
        let collected = errors.collect_errors();
        let unused = errors.collect_unused_ignore_errors(&collected);
        assert_eq!(unused.len(), 1);
        assert!(unused[0].msg().contains("Unused"));
    }

    #[test]
    fn test_unused_specific_code_ignore() {
        // An ignore comment with a specific code that doesn't match any error
        let contents = r#"
def f() -> int:
    # pyrefly: ignore [bad-override]
    return 1
"#;
        let (errors, _tdir) = get_errors(contents);
        let collected = errors.collect_errors();
        let unused = errors.collect_unused_ignore_errors(&collected);
        assert_eq!(unused.len(), 1);
        assert!(unused[0].msg().contains("bad-override"));
    }

    #[test]
    fn test_used_ignore_no_errors() {
        // An ignore comment that is actually used should not be reported
        let contents = r#"
def f() -> int:
    # pyrefly: ignore [bad-return]
    return "hello"
"#;
        let (errors, _tdir) = get_errors(contents);
        let collected = errors.collect_errors();
        let unused = errors.collect_unused_ignore_errors(&collected);
        assert!(unused.is_empty());
    }

    #[test]
    fn test_partially_used_ignore() {
        // An ignore with multiple codes where only some are used
        let contents = r#"
def f() -> int:
    # pyrefly: ignore [bad-return, bad-override]
    return "hello"
"#;
        let (errors, _tdir) = get_errors(contents);
        let collected = errors.collect_errors();
        let unused = errors.collect_unused_ignore_errors(&collected);
        assert_eq!(unused.len(), 1);
        assert!(unused[0].msg().contains("bad-override"));
        assert!(!unused[0].msg().contains("bad-return"));
    }

    #[test]
    fn test_no_ignores_no_errors() {
        // Code with no ignores should produce no unused ignore errors
        let contents = r#"
def f() -> int:
    return 1
"#;
        let (errors, _tdir) = get_errors(contents);
        let collected = errors.collect_errors();
        let unused = errors.collect_unused_ignore_errors(&collected);
        assert!(unused.is_empty());
    }

    #[test]
    fn test_multiple_unused_ignores() {
        // Multiple unused ignores in the same file
        let contents = r#"
def f() -> int:
    # pyrefly: ignore [bad-override]
    return 1

def g() -> str:
    # pyrefly: ignore
    return "hello"
"#;
        let (errors, _tdir) = get_errors(contents);
        let collected = errors.collect_errors();
        let unused = errors.collect_unused_ignore_errors(&collected);
        assert_eq!(unused.len(), 2);
    }

    #[test]
    fn test_invalid_ignore_comment_emitted() {
        // A typo'd directive that suppresses nothing is flagged, while the real
        // error underneath still fires (the lint is purely additive).
        let contents = r#"
x: int = "oops"  # pyrefly: ignoree
"#;
        let (errors, _tdir) = get_errors(contents);
        let lints = errors.collect_ignore_comment_lint_errors();
        assert_eq!(lints.len(), 1);
        assert_eq!(lints[0].error_kind(), ErrorKind::InvalidIgnoreComment);

        // The bad-assignment error is not suppressed by the malformed comment.
        let ordinary = errors.collect_errors().ordinary;
        assert!(
            ordinary
                .iter()
                .any(|e| e.error_kind() == ErrorKind::BadAssignment)
        );
    }

    #[test]
    fn test_invalid_ignore_comment_variants() {
        // Unterminated and empty brackets are flagged; valid forms are not.
        let contents = r#"
a: int = "x"  # pyrefly: ignore[bad-assignment
b: int = "x"  # pyrefly: ignore[]
c: int = "x"  # pyrefly: ignore[bad-assignment]
d = list[int]  # type: list[int]
"#;
        let (errors, _tdir) = get_errors(contents);
        let lints = errors.collect_ignore_comment_lint_errors();
        assert_eq!(lints.len(), 2);
        assert!(
            lints
                .iter()
                .all(|e| e.error_kind() == ErrorKind::InvalidIgnoreComment)
        );
    }

    #[test]
    fn test_ignore_without_code_emitted() {
        // A bare `# pyrefly: ignore` is flagged; a coded one is not.
        let contents = r#"
x: int = "oops"  # pyrefly: ignore
y: int = "oops"  # pyrefly: ignore[bad-assignment]
"#;
        let (errors, _tdir) = get_errors(contents);
        let lints = errors.collect_ignore_comment_lint_errors();
        assert_eq!(lints.len(), 1);
        assert_eq!(lints[0].error_kind(), ErrorKind::IgnoreWithoutCode);
    }

    #[test]
    fn test_ignore_without_code_scoped_to_pyrefly() {
        // `# type: ignore` is bare but belongs to another tool, so it is not flagged.
        let contents = r#"
x: int = "oops"  # type: ignore
"#;
        let (errors, _tdir) = get_errors(contents);
        let lints = errors.collect_ignore_comment_lint_errors();
        assert!(lints.is_empty());
    }

    #[test]
    fn test_empty_bracket_ignore_is_additive() {
        // `# type: ignore[]` suppressed all errors before (per-line codes are not
        // checked for `type`), and must continue to — the new lint only adds a
        // diagnostic on top.
        let contents = r#"
x: int = "oops"  # type: ignore[]
"#;
        let (errors, _tdir) = get_errors(contents);
        let ordinary = errors.collect_errors().ordinary;
        assert!(
            !ordinary
                .iter()
                .any(|e| e.error_kind() == ErrorKind::BadAssignment),
            "empty-bracket `# type: ignore[]` must still suppress the error"
        );
        let lints = errors.collect_ignore_comment_lint_errors();
        assert_eq!(lints.len(), 1);
        assert_eq!(lints[0].error_kind(), ErrorKind::InvalidIgnoreComment);
    }

    #[test]
    fn test_invalid_ignore_comment_is_unsuppressable() {
        // A `# pyrefly: ignore[invalid-ignore-comment]` cannot silence the lint
        // about a malformed comment on the same line: ignore-comment diagnostics
        // are unsuppressable, so the lint still fires.
        let contents = r#"
x = 1  # pyrefly: ignoree  # pyrefly: ignore[invalid-ignore-comment]
"#;
        let (errors, _tdir) = get_errors(contents);
        let lints = errors.collect_ignore_comment_lint_errors();
        assert_eq!(lints.len(), 1);
        assert_eq!(lints[0].error_kind(), ErrorKind::InvalidIgnoreComment);
    }

    #[test]
    fn test_ignore_comment_lints_survive_file_level_ignore_errors() {
        // A whole-file `# pyrefly: ignore-errors` suppresses ordinary errors but
        // not the ignore-comment diagnostics, which are unsuppressable.
        let contents = r#"# pyrefly: ignore-errors
x: int = "oops"  # pyrefly: ignoree
"#;
        let (errors, _tdir) = get_errors(contents);
        assert!(
            !errors
                .collect_errors()
                .ordinary
                .iter()
                .any(|e| e.error_kind() == ErrorKind::BadAssignment),
            "file-level `# pyrefly: ignore-errors` must still suppress the error"
        );
        let lints = errors.collect_ignore_comment_lint_errors();
        assert_eq!(lints.len(), 1);
        assert_eq!(lints[0].error_kind(), ErrorKind::InvalidIgnoreComment);
    }

    #[test]
    fn test_backslash_continuation_ranges_ignores_comment_backslash() {
        use pyrefly_util::lined_buffer::LineNumber;

        use super::sorted_backslash_continuation_ranges;

        let no_strings = vec![];

        // A trailing backslash inside a comment should NOT trigger continuation.
        let lines = vec!["x = 1  # comment \\", "y = 2"];
        let ranges = sorted_backslash_continuation_ranges(&lines, &no_strings);
        assert!(
            ranges.is_empty(),
            "comment backslash should not be a continuation"
        );

        // A real continuation should still be detected.
        let lines = vec!["x = 1 + \\", "    2"];
        let ranges = sorted_backslash_continuation_ranges(&lines, &no_strings);
        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0].0, LineNumber::from_zero_indexed(0));
        assert_eq!(ranges[0].1, LineNumber::from_zero_indexed(1));
    }

    #[test]
    fn test_backslash_continuation_ranges_ignores_multiline_strings() {
        use pyrefly_util::lined_buffer::LineNumber;

        use super::sorted_backslash_continuation_ranges;

        // A backslash at end of line inside a triple-quoted string should
        // NOT be detected as a continuation.
        let lines = vec![
            "x = \"\"\"\\", // line 0: start of triple-quoted string with \
            "hello\\",      // line 1: inside string with \
            "\"\"\"",       // line 2: end of string
        ];
        let string_ranges = vec![(
            LineNumber::from_zero_indexed(0),
            LineNumber::from_zero_indexed(2),
        )];
        let ranges = sorted_backslash_continuation_ranges(&lines, &string_ranges);
        assert!(
            ranges.is_empty(),
            "backslash inside multiline string should not be a continuation"
        );
    }
}
