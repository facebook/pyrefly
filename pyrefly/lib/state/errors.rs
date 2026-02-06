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
use pyrefly_python::module_path::ModulePath;
use pyrefly_util::arc_id::ArcId;
use pyrefly_util::lined_buffer::LineNumber;
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

/// The errors from a collection of modules.
#[derive(Debug)]
pub struct Errors {
    // Sorted by module name and path (so deterministic display order)
    loads: Vec<(Arc<Load>, ArcId<ConfigFile>)>,
}

impl Errors {
    pub fn new(mut loads: Vec<(Arc<Load>, ArcId<ConfigFile>)>) -> Self {
        loads.sort_by_key(|x| (x.0.module_info.name(), x.0.module_info.path().dupe()));
        Self { loads }
    }

    pub fn collect_errors(&self) -> CollectedErrors {
        let mut errors = CollectedErrors::default();
        for (load, config) in &self.loads {
            let error_config = config.get_error_config(load.module_info.path().as_path());
            load.errors.collect_into(&error_config, &mut errors);
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
        for (load, _) in &self.loads {
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

        for error in &collected.suppressed {
            if error.is_ignored(&Tool::default_enabled()) {
                let module_path = error.path();
                let start_line = error.display_range().start.line_within_file();
                let end_line = error.display_range().end.line_within_file();
                let error_code = error.error_kind().to_name().to_owned();

                // Track the error code for all lines the error spans
                for line_idx in start_line.to_zero_indexed()..=end_line.to_zero_indexed() {
                    suppressed_codes_by_module
                        .entry(module_path)
                        .or_default()
                        .entry(LineNumber::from_zero_indexed(line_idx))
                        .or_default()
                        .insert(error_code.clone());
                }
            }
        }

        // Iterate over each module and check for unused ignores
        for (load, _) in &self.loads {
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
            for (load, config) in &self.loads {
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
        for (load, config) in &self.loads {
            let error_config = config.get_error_config(load.module_info.path().as_path());
            Expectation::parse(load.module_info.dupe(), load.module_info.contents())
                .check(&load.errors.collect(&error_config).shown)?;
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
            for (load, config) in &self.loads {
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
        transaction.run(&[handle.dupe()], Require::Everything);
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
