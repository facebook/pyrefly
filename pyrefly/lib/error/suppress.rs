/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::path::Path;
use std::path::PathBuf;

use anyhow::anyhow;
use pyrefly_config::error_kind::Severity;
use pyrefly_python::ast::Ast;
use pyrefly_python::module::GENERATED_TOKEN;
use pyrefly_python::module_path::ModulePathDetails;
use pyrefly_util::fs_anyhow;
use pyrefly_util::lined_buffer::LineNumber;
use regex::Regex;
use starlark_map::small_map::SmallMap;
use starlark_map::small_set::SmallSet;
use tracing::error;
use tracing::info;

use crate::error::error::Error;
use crate::state::errors::Errors;

/// Combines all errors that affect one line into a single entry.
// The current format is: `# pyrefly: ignore  # error1, error2, ...`
fn dedup_errors(errors: &[Error]) -> SmallMap<usize, String> {
    let mut deduped_errors = SmallMap::new();
    for error in errors {
        let e: &mut String = deduped_errors
            .entry(error.display_range().start.line.to_zero_indexed() as usize)
            .or_default();
        let contains_error = e.contains(error.error_kind().to_name());
        if e.is_empty() {
            e.push_str("# pyrefly: ignore  # ");
        } else if !contains_error {
            e.push_str(", ");
        }

        if !contains_error {
            e.push_str(error.error_kind().to_name());
        }
    }
    deduped_errors
}

// TODO: In future have this return an ast as well as the string for comparison
fn read_and_validate_file(path: &Path) -> anyhow::Result<String> {
    let file = fs_anyhow::read_to_string(path);
    match file {
        Ok(file) => {
            // Check for generated + parsable files
            let (_ast, parse_errors, _unsupported_syntax_errors) = Ast::parse(&file);
            if !parse_errors.is_empty() {
                return Err(anyhow!("File is not parsable"));
            }
            if file.contains(GENERATED_TOKEN) {
                return Err(anyhow!("Generated file"));
            }
            Ok(file)
        }
        Err(e) => Err(e),
    }
}

/// Adds error suppressions for the given errors in the given files.
/// Returns a list of files that failed to be patched, and a list of files that were patched.
/// The list of failures includes the error that occurred, which may be a read or write error.
fn add_suppressions(
    path_errors: &SmallMap<PathBuf, Vec<Error>>,
    same_line: bool,
) -> (Vec<(&PathBuf, anyhow::Error)>, Vec<&PathBuf>) {
    let mut failures = vec![];
    let mut successes = vec![];
    for (path, errors) in path_errors {
        let file = match read_and_validate_file(path) {
            Ok(f) => f,
            Err(e) => {
                failures.push((path, e));
                continue;
            }
        };
        let deduped_errors = dedup_errors(errors);
        let mut buf = String::new();
        for (idx, line) in file.lines().enumerate() {
            if same_line && let Some(error_comment) = deduped_errors.get(&idx) {
                let new_line = format!("{} {}", line, error_comment);
                buf.push_str(&new_line);
                buf.push('\n');
            } else {
                if let Some(error_comment) = deduped_errors.get(&idx) {
                    // As a simple formatting step, indent the error comment to match the line below it.
                    if let Some(first_char) = line.find(|c: char| !c.is_whitespace()) {
                        buf.push_str(&line[..first_char]);
                    }
                    buf.push_str(error_comment);
                    buf.push('\n');
                }
                buf.push_str(line);
                buf.push('\n');
            }
        }
        if let Err(e) = fs_anyhow::write(path, buf) {
            failures.push((path, e));
        } else {
            successes.push(path);
        }
    }
    (failures, successes)
}

pub fn suppress_errors(errors: Vec<Error>, same_line: bool) {
    let mut path_errors: SmallMap<PathBuf, Vec<Error>> = SmallMap::new();
    for e in errors {
        if e.severity() >= Severity::Warn
            && let ModulePathDetails::FileSystem(path) = e.path().details()
        {
            path_errors.entry(path.clone()).or_default().push(e);
        }
    }
    info!("Inserting error suppressions...");
    if path_errors.is_empty() {
        info!("No errors to suppress!");
        return;
    }
    let (failures, successes) = add_suppressions(&path_errors, same_line);
    info!(
        "Finished suppressing errors in {}/{} files",
        successes.len(),
        path_errors.len()
    );
    if !failures.is_empty() {
        info!("Failed to suppress errors in {} files:", failures.len());
        for (path, e) in failures {
            info!("  {path:#?}: {e}");
        }
    }
}

pub fn find_unused_ignores<'a>(
    all_ignores: SmallMap<&'a PathBuf, SmallSet<LineNumber>>,
    suppressed_errors: SmallMap<&PathBuf, SmallSet<LineNumber>>,
) -> SmallMap<&'a PathBuf, SmallSet<LineNumber>> {
    let mut all_unused_ignores: SmallMap<&PathBuf, SmallSet<LineNumber>> = SmallMap::new();
    let default_set: SmallSet<LineNumber> = SmallSet::new();
    // Loop over each path only save the ignores that are not in use
    for (path, ignores) in all_ignores {
        let errors = suppressed_errors.get(path).unwrap_or(&default_set);
        let mut unused_ignores = SmallSet::new();
        for ignore in ignores {
            let location = ignore.increment();
            if !errors.contains(&location) && !errors.contains(&ignore) {
                unused_ignores.insert(ignore);
            }
        }

        all_unused_ignores.insert(path, unused_ignores);
    }
    all_unused_ignores
}

pub fn remove_unused_ignores(loads: &Errors, all: bool) {
    let errors = loads.collect_errors();
    let mut all_ignores: SmallMap<&PathBuf, SmallSet<LineNumber>> = SmallMap::new();
    for (module_path, ignore) in loads.collect_ignores() {
        if let ModulePathDetails::FileSystem(path) = module_path.details() {
            all_ignores.insert(path, ignore.get_pyrefly_ignores(all));
        }
    }

    let mut suppressed_errors: SmallMap<&PathBuf, SmallSet<LineNumber>> = SmallMap::new();
    for e in &errors.suppressed {
        if e.is_ignored(false)
            && let ModulePathDetails::FileSystem(path) = e.path().details()
        {
            suppressed_errors
                .entry(path)
                .or_default()
                .insert(e.display_range().start.line);
        }
    }

    let path_ignores = find_unused_ignores(all_ignores, suppressed_errors);

    // TODO: right now we only remove pyrefly ignores, but we should have options to clean up
    // other comment based ignores as well
    let regex = Regex::new(r"(#\s*pyrefly:\s*ignore.*$|#\s*type:\s*ignore.*$)").unwrap();
    let mut removed_ignores: SmallMap<&PathBuf, usize> = SmallMap::new();
    for (path, ignores) in path_ignores {
        let mut unused_ignore_count = 0;
        let mut ignore_locations: SmallSet<usize> = SmallSet::new();
        for ignore in ignores {
            let above_line = ignore
                .decrement()
                .expect("Invalid error suppression location")
                .to_zero_indexed() as usize;
            let same_line = ignore.to_zero_indexed() as usize;
            ignore_locations.insert(above_line);
            ignore_locations.insert(same_line);
        }
        if let Ok(file) = read_and_validate_file(path) {
            let mut buf = String::with_capacity(file.len());
            let lines = file.lines();
            for (idx, line) in lines.enumerate() {
                if ignore_locations.contains(&idx) {
                    unused_ignore_count += 1;
                    let new_string = regex.replace_all(line, "");
                    if !new_string.trim().is_empty() {
                        buf.push_str(new_string.trim_end());
                    }
                    buf.push('\n');
                    continue;
                }
                buf.push_str(line);
                buf.push('\n');
            }
            if let Err(e) = fs_anyhow::write(path, buf) {
                error!("Failed to remove unused error suppressions in {} files:", e);
            } else if unused_ignore_count > 0 {
                removed_ignores.insert(path, unused_ignore_count);
            }
        }
    }
    info!(
        "Removed {} unused error suppression(s) in {} file(s)",
        removed_ignores.values().sum::<usize>(),
        removed_ignores.len(),
    );
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
    use tempfile;
    use tempfile::TempDir;

    use super::*;
    use crate::config::config::ConfigFile;
    use crate::config::finder::ConfigFinder;
    use crate::error::suppress;
    use crate::state::require::Require;
    use crate::state::state::State;

    #[derive(PartialEq)]
    enum SuppressFlag {
        Remove,
        Add,
    }

    fn get_path(tdir: &TempDir) -> PathBuf {
        tdir.path().join("test.py")
    }

    fn assert_suppress_errors(before: &str, after: &str) {
        assert_suppressions(before, after, SuppressFlag::Add, false, false)
    }

    fn assert_suppress_same_line(before: &str, after: &str) {
        assert_suppressions(before, after, SuppressFlag::Add, false, true)
    }

    fn assert_remove_ignores(before: &str, after: &str, all: bool) {
        assert_suppressions(before, after, SuppressFlag::Remove, all, false)
    }

    fn assert_suppressions(
        before: &str,
        after: &str,
        kind: SuppressFlag,
        all: bool,
        same_line: bool,
    ) {
        let tdir = tempfile::tempdir().unwrap();

        let mut config = ConfigFile::default();
        config.python_environment.set_empty_to_default();
        let name = "test";
        let contents = before;
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
            Some(Arc::new((*contents).to_owned())),
        )]);
        transaction.run(&[handle.dupe()], Require::Everything);
        let loads = transaction.get_errors([handle.clone()].iter());
        if kind == SuppressFlag::Add {
            suppress::suppress_errors(loads.collect_errors().shown, same_line);
        } else {
            suppress::remove_unused_ignores(&loads, all);
        }

        let got_file = fs_anyhow::read_to_string(&get_path(&tdir)).unwrap();
        assert_eq!(after, got_file);
    }

    #[test]
    fn test_add_suppressions() {
        assert_suppress_errors(
            r#"
x: str = 1


def f(y: int) -> None:
    """Doc comment"""
    x = "one" + y
    return x


f(x)

"#,
            r#"
# pyrefly: ignore  # bad-assignment
x: str = 1


def f(y: int) -> None:
    """Doc comment"""
    # pyrefly: ignore  # unsupported-operation
    x = "one" + y
    return x


# pyrefly: ignore  # bad-argument-type
f(x)

"#,
        );
    }

    #[test]
    fn test_add_suppressions_existing_comment() {
        assert_suppress_errors(
            r#"
def foo() -> int:
    # comment
    return ""
"#,
            r#"
def foo() -> int:
    # comment
    # pyrefly: ignore  # bad-return
    return ""
"#,
        );
    }

    #[test]
    fn test_add_suppressions_duplicate_errors() {
        assert_suppress_errors(
            r#"
# comment
def foo() -> int: pass
"#,
            r#"
# comment
# pyrefly: ignore  # bad-return
def foo() -> int: pass
"#,
        );
    }

    #[test]
    fn test_add_suppressions_multiple_errors_one_line() {
        assert_suppress_errors(
            r#"
# comment
def foo(x: int) -> str:
    return ""
x: int = foo("Hello")
"#,
            r#"
# comment
def foo(x: int) -> str:
    return ""
# pyrefly: ignore  # bad-assignment, bad-argument-type
x: int = foo("Hello")
"#,
        );
    }

    #[test]
    fn test_add_suppressions_unparsable_line_break() {
        assert_suppress_errors(
            r#"
def foo() -> None:
    line_break = \\
        [
            param
        ]
    unrelated_line = 0
        "#,
            r#"
def foo() -> None:
    line_break = \\
        [
            param
        ]
    unrelated_line = 0
        "#,
        );
    }

    #[test]
    fn test_no_suppress_generated_files() {
        let file_contents = format!(
            r#"
{GENERATED_TOKEN}

def bar() -> None: 
pass 
    "#,
        );
        assert_suppress_errors(&file_contents, &file_contents);
    }

    #[test]
    fn test_remove_suppression_above() {
        let input = r#"
def f() -> int:
    # pyrefly: ignore # bad-return
    return 1
"#;
        let want = r#"
def f() -> int:

    return 1
"#;
        assert_remove_ignores(input, want, false);
    }

    #[test]
    fn test_remove_suppression_above_two() {
        let input = r#"
def g() -> str:
    # pyrefly: ignore # bad-return
    return "hello"
"#;
        let want = r#"
def g() -> str:

    return "hello"
"#;
        assert_remove_ignores(input, want, false);
    }

    #[test]
    fn test_remove_suppression_inline() {
        let input = r#"
def g() -> str:
    return "hello" # pyrefly: ignore # bad-return
"#;
        let want = r#"
def g() -> str:
    return "hello"
"#;
        assert_remove_ignores(input, want, false);
    }

    #[test]
    fn test_remove_suppression_multiple() {
        let input = r#"
def g() -> str:
    return "hello" # pyrefly: ignore # bad-return
def f() -> int:
    # pyrefly: ignore
    return 1
"#;
        let output = r##"
def g() -> str:
    return "hello"
def f() -> int:

    return 1
"##;
        assert_remove_ignores(input, output, false);
    }

    #[test]
    fn test_no_remove_suppression_generated() {
        let input = format!(
            r#"
{GENERATED_TOKEN}
def g() -> str:
    return "hello" # pyrefly: ignore # bad-return
def f() -> int:
    # pyrefly: ignore
    return 1
"#,
        );
        assert_remove_ignores(&input, &input, false);
    }

    #[test]
    fn test_no_remove_suppression() {
        let input = r#"
def g() -> int:
    return "hello" # pyrefly: ignore # bad-return
"#;
        assert_remove_ignores(input, input, false);
    }
    #[test]
    fn test_remove_generic_suppression() {
        let before = r#"
def g() -> str:
    return "hello" # type: ignore # bad-return
"#;
        let after = r#"
def g() -> str:
    return "hello"
"#;
        assert_remove_ignores(before, after, true);
    }
    #[test]
    fn test_remove_generic_suppression_error_type() {
        let before = r#"
def g() -> str:
    return "hello" # type: ignore[bad-test]
"#;
        let after = r#"
def g() -> str:
    return "hello"
"#;
        assert_remove_ignores(before, after, true);
    }
    #[test]
    fn test_add_suppressions_same_line() {
        assert_suppress_same_line(
            r#"
x: str = 1

"#,
            r#"
x: str = 1 # pyrefly: ignore  # bad-assignment

"#,
        );
    }
}
