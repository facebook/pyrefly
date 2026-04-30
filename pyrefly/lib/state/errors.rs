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
use pyrefly_python::ignore::Tool;
use pyrefly_python::ignore::find_comment_start_in_line;
use pyrefly_python::module::Module;
use pyrefly_python::module_path::ModulePath;
use pyrefly_util::arc_id::ArcId;
use pyrefly_util::lined_buffer::LineNumber;
use pyrefly_util::visit::Visit;
use ruff_python_ast::Expr;
use ruff_python_ast::ModModule;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;
use starlark_map::small_map::SmallMap;
use starlark_map::small_set::SmallSet;
use vec1::vec1;

use crate::config::config::ConfigFile;
use crate::error::baseline::BaselineProcessor;
use crate::error::collector::CollectedErrors;
use crate::error::error::Error;
use crate::error::expectation::Expectation;
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
                ranges.push((start, end));
            }
        }
    });
    ranges.sort();
    ranges
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

/// Per-module multi-line ranges and ignore-all directives, computed after parsing.
#[derive(Debug)]
pub struct ModuleRanges {
    /// Multi-line string and backslash-continuation ranges.
    pub multi_line: Vec<(LineNumber, LineNumber)>,
    /// Top-level ignore-all directives (e.g. `# pyrefly: ignore-errors`).
    pub ignore_all: SmallMap<Tool, LineNumber>,
}

/// The errors from a collection of modules.
#[derive(Debug)]
pub struct Errors {
    // Sorted by module name and path (so deterministic display order)
    loads: Vec<(Arc<Load>, ArcId<ConfigFile>, ModuleRanges)>,
}

impl Errors {
    pub fn new(mut loads: Vec<(Arc<Load>, ArcId<ConfigFile>, ModuleRanges)>) -> Self {
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
        for (load, config, ranges) in &self.loads {
            let error_config = config.get_error_config(load.module_info.path().as_path());
            load.errors.collect_into(
                &error_config,
                &ranges.multi_line,
                &ranges.ignore_all,
                &mut errors,
            );
        }
        errors
    }

    pub fn collect_errors_with_baseline(
        &self,
        baseline_path: Option<&Path>,
        relative_to: &Path,
    ) -> CollectedErrors {
        let errors = self.collect_errors();
        self.apply_baseline(errors, baseline_path, relative_to)
    }

    /// Apply baseline filtering to already-collected errors.
    pub fn apply_baseline(
        &self,
        mut errors: CollectedErrors,
        baseline_path: Option<&Path>,
        relative_to: &Path,
    ) -> CollectedErrors {
        if let Some(baseline_path) = baseline_path
            && let Ok(processor) = BaselineProcessor::from_file(baseline_path)
        {
            processor.process_errors(&mut errors.ordinary, &mut errors.baseline, relative_to);
        }
        errors
    }

    pub fn collect_display_errors(&self) -> Vec<Error> {
        let errors = self.collect_errors();
        Self::merge_display_errors(errors.ordinary, errors.directives)
    }

    pub fn collect_display_errors_with_unused_ignores(&self) -> Vec<Error> {
        let collected = self.collect_errors();
        let unused = self.collect_unused_ignore_errors_for_display(&collected);
        let mut ordinary = collected.ordinary;
        ordinary.extend(unused.ordinary);
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
        let fstring_ranges_by_module: SmallMap<&ModulePath, &[(LineNumber, LineNumber)]> = self
            .loads
            .iter()
            .map(|(load, _, ranges)| (load.module_info.path(), ranges.multi_line.as_slice()))
            .collect();

        let enabled_ignores_by_module: SmallMap<&ModulePath, SmallSet<Tool>> = self
            .loads
            .iter()
            .map(|(load, config, _)| {
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
        for (load, config, _) in &self.loads {
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
                            vec1!["Unused pyre-fixme comment".to_owned()],
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
                        vec1![msg],
                        ErrorKind::UnusedIgnore,
                    ));
                }
            }
        }

        unused_errors
    }

    /// Collects unused ignore errors for display, respecting severity configuration.
    /// Unlike `collect_unused_ignore_errors()`, this applies severity filtering so
    /// errors with `Severity::Ignore` are not included in the ordinary results.
    /// Accepts pre-collected errors to avoid redundant error collection.
    pub fn collect_unused_ignore_errors_for_display(
        &self,
        collected: &CollectedErrors,
    ) -> CollectedErrors {
        let unused_errors = self.collect_unused_ignore_errors(collected);
        let mut result = CollectedErrors::default();

        // Build a path-to-config map for O(1) lookup instead of O(loads) per error.
        let config_by_path: SmallMap<&ModulePath, &ArcId<ConfigFile>> = self
            .loads
            .iter()
            .map(|(load, config, _)| (load.module_info.path(), config))
            .collect();

        for error in unused_errors {
            if let Some(config) = config_by_path.get(&error.path()) {
                let error_config = config.get_error_config(error.path().as_path());
                let severity = error_config
                    .display_config
                    .severity(ErrorKind::UnusedIgnore);
                match severity {
                    Severity::Error => result.ordinary.push(error.with_severity(Severity::Error)),
                    Severity::Warn => result.ordinary.push(error.with_severity(Severity::Warn)),
                    Severity::Info => result.ordinary.push(error.with_severity(Severity::Info)),
                    Severity::Ignore => result.disabled.push(error),
                }
            }
        }

        result
    }

    pub fn check_against_expectations(&self) -> anyhow::Result<()> {
        for (load, config, ranges) in &self.loads {
            let error_config = config.get_error_config(load.module_info.path().as_path());
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
    use pyrefly_python::module_name::ModuleName;
    use pyrefly_python::module_path::ModulePath;
    use pyrefly_python::sys_info::SysInfo;
    use pyrefly_util::arc_id::ArcId;
    use pyrefly_util::fs_anyhow;
    use regex::Regex;
    use tempfile::TempDir;

    use crate::config::config::ConfigFile;
    use crate::config::finder::ConfigFinder;
    use crate::state::errors::Errors;
    use crate::state::load::FileContents;
    use crate::state::require::Require;
    use crate::state::state::State;
    use crate::test::util::TEST_THREAD_COUNT;

    impl Errors {
        pub fn check_var_leak(&self) -> anyhow::Result<()> {
            let regex = Regex::new(r"@\d+").unwrap();
            for (load, config, _) in &self.loads {
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
