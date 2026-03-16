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
use pyrefly_python::module::Module;
use pyrefly_python::module_path::ModulePath;
use pyrefly_util::arc_id::ArcId;
use pyrefly_util::lined_buffer::LineNumber;
use pyrefly_util::visit::Visit;
use ruff_python_ast::Expr;
use ruff_python_ast::ModModule;
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

/// Extracts `(start_line, end_line)` ranges for all multi-line f-strings and
/// t-strings from the AST. Single-line f/t-strings (where start == end) are
/// excluded. The returned list is sorted by start_line.
pub fn sorted_multi_line_fstring_ranges(
    ast: &ModModule,
    module: &Module,
) -> Vec<(LineNumber, LineNumber)> {
    let mut ranges = Vec::new();
    ast.visit(&mut |expr: &Expr| {
        let text_range = match expr {
            Expr::FString(x) => Some(x.range),
            Expr::TString(x) => Some(x.range),
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

/// The errors from a collection of modules.
#[derive(Debug)]
pub struct Errors {
    // Sorted by module name and path (so deterministic display order)
    loads: Vec<(Arc<Load>, ArcId<ConfigFile>, Vec<(LineNumber, LineNumber)>)>,
}

impl Errors {
    pub fn new(
        mut loads: Vec<(Arc<Load>, ArcId<ConfigFile>, Vec<(LineNumber, LineNumber)>)>,
    ) -> Self {
        loads.sort_by_key(|x| (x.0.module_info.name(), x.0.module_info.path().dupe()));
        Self { loads }
    }

    pub fn collect_errors(&self) -> CollectedErrors {
        let mut errors = CollectedErrors::default();
        for (load, config, fstring_ranges) in &self.loads {
            let error_config = config.get_error_config(load.module_info.path().as_path());
            load.errors
                .collect_into(&error_config, fstring_ranges, &mut errors);
        }
        errors
    }

    pub fn collect_errors_with_baseline(
        &self,
        baseline_path: Option<&Path>,
        relative_to: &Path,
    ) -> CollectedErrors {
        let mut errors = self.collect_errors();
        if let Some(baseline_path) = baseline_path
            && let Ok(processor) = BaselineProcessor::from_file(baseline_path)
        {
            processor.process_errors(&mut errors.shown, &mut errors.baseline, relative_to);
        }
        errors
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
    pub fn collect_unused_ignore_errors(&self) -> Vec<Error> {
        let collected = self.collect_errors();
        let mut unused_errors = Vec::new();

        // Build a map of which error codes were suppressed on each line, keyed by module path.
        // Key: module_path, Value: map from line number to set of suppressed error codes
        let mut suppressed_codes_by_module: SmallMap<
            &ModulePath,
            SmallMap<LineNumber, SmallSet<String>>,
        > = SmallMap::new();

        // Build a map from module path to f-string ranges for lookup.
        let fstring_ranges_by_module: SmallMap<&ModulePath, &[(LineNumber, LineNumber)]> = self
            .loads
            .iter()
            .map(|(load, _, ranges)| (load.module_info.path(), ranges.as_slice()))
            .collect();

        for error in &collected.suppressed {
            if error.is_ignored(&Tool::default_enabled()) {
                let module_path = error.path();
                let start_line = error.display_range().start.line_within_file();
                let end_line = error.display_range().end.line_within_file();
                let error_code = error.error_kind().to_name().to_owned();

                let module_codes = suppressed_codes_by_module.entry(module_path).or_default();

                // Track the error code for all lines the error spans.
                for line_idx in start_line.to_zero_indexed()..=end_line.to_zero_indexed() {
                    module_codes
                        .entry(LineNumber::from_zero_indexed(line_idx))
                        .or_default()
                        .insert(error_code.clone());
                }

                // If the error is inside a multi-line f/t-string, also track
                // the code at the f-string's start and end lines so that a
                // suppression comment placed there is recognized as "used".
                if let Some(ranges) = fstring_ranges_by_module.get(&module_path)
                    && let Some((fs_start, fs_end)) = find_containing_range(ranges, start_line)
                {
                    module_codes
                        .entry(fs_start)
                        .or_default()
                        .insert(error_code.clone());
                    module_codes
                        .entry(fs_end)
                        .or_default()
                        .insert(error_code.clone());
                }
            }
        }

        // Iterate over each module and check for unused ignores
        for (load, _, _) in &self.loads {
            let module = &load.module_info;
            let module_path = module.path();
            let ignore = module.ignore();

            // Get the suppressed codes for this module (if any)
            let module_suppressed_codes = suppressed_codes_by_module.get(&module_path);

            for (applies_to_line, suppressions) in ignore.iter() {
                for supp in suppressions {
                    // Only check pyrefly suppressions
                    if supp.tool() != Tool::Pyrefly {
                        continue;
                    }

                    let declared_codes: SmallSet<String> =
                        supp.error_codes().iter().cloned().collect();

                    // Get the error codes actually suppressed on this line
                    let used_codes: SmallSet<String> = module_suppressed_codes
                        .and_then(|m| m.get(applies_to_line))
                        .cloned()
                        .unwrap_or_default();

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
    /// errors with `Severity::Ignore` are not included in the shown results.
    pub fn collect_unused_ignore_errors_for_display(&self) -> CollectedErrors {
        let unused_errors = self.collect_unused_ignore_errors();
        let mut result = CollectedErrors::default();

        for error in unused_errors {
            // Find the config for this error's module
            for (load, config, _) in &self.loads {
                if load.module_info.path() == error.path() {
                    let error_config = config.get_error_config(error.path().as_path());
                    let severity = error_config
                        .display_config
                        .severity(ErrorKind::UnusedIgnore);
                    match severity {
                        Severity::Error => result.shown.push(error.with_severity(Severity::Error)),
                        Severity::Warn => result.shown.push(error.with_severity(Severity::Warn)),
                        Severity::Info => result.shown.push(error.with_severity(Severity::Info)),
                        Severity::Ignore => result.disabled.push(error),
                    }
                    break;
                }
            }
        }

        result
    }

    pub fn check_against_expectations(&self) -> anyhow::Result<()> {
        for (load, config, fstring_ranges) in &self.loads {
            let error_config = config.get_error_config(load.module_info.path().as_path());
            let mut result = CollectedErrors::default();
            load.errors
                .collect_into(&error_config, fstring_ranges, &mut result);
            Expectation::parse(load.module_info.dupe(), load.module_info.contents())
                .check(&result.shown)?;
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

    impl Errors {
        pub fn check_var_leak(&self) -> anyhow::Result<()> {
            let regex = Regex::new(r"@\d+").unwrap();
            for (load, config, _) in &self.loads {
                let error_config = config.get_error_config(load.module_info.path().as_path());
                let errors = load.errors.collect(&error_config).shown;
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
        let state = State::new(ConfigFinder::new_constant(config));
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
        let unused = errors.collect_unused_ignore_errors();
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
        let unused = errors.collect_unused_ignore_errors();
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
        let unused = errors.collect_unused_ignore_errors();
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
        let unused = errors.collect_unused_ignore_errors();
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
        let unused = errors.collect_unused_ignore_errors();
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
        let unused = errors.collect_unused_ignore_errors();
        assert_eq!(unused.len(), 2);
    }
}
