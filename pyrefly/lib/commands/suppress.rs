/*
* Copyright (c) Meta Platforms, Inc. and affiliates.
*
* This source code is licensed under the MIT license found in the
* LICENSE file in the root directory of this source tree.
*/

use std::path::Path;
use std::path::PathBuf;

use anyhow::anyhow;
use pyrefly_python::ast::Ast;
use pyrefly_util::fs_anyhow;
use pyrefly_util::lined_buffer::LineNumber;
use regex::Regex;
use starlark_map::small_map::SmallMap;
use starlark_map::small_set::SmallSet;
use tracing::error;

use crate::error::error::Error;
use crate::module::module_info::GENERATED_TOKEN;

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
            let (_ast, parse_errors) = Ast::parse(&file);
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
/// Returns a list of files that failed to be be patched, and a list of files that were patched.
/// The list of failures includes the error that occurred, which may be a read or write error.
fn add_suppressions(
    path_errors: &SmallMap<PathBuf, Vec<Error>>,
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
        if let Err(e) = fs_anyhow::write(path, buf.as_bytes()) {
            failures.push((path, e));
        } else {
            successes.push(path);
        }
    }
    (failures, successes)
}

pub fn suppress_errors(path_errors: &SmallMap<PathBuf, Vec<Error>>) {
    eprintln!("Inserting error suppressions...");
    if path_errors.is_empty() {
        eprintln!("No errors to suppress!");
        return;
    }
    let (failures, successes) = add_suppressions(path_errors);
    eprintln!(
        "Finished suppressing errors in {}/{} files",
        successes.len(),
        path_errors.len()
    );
    if !failures.is_empty() {
        eprintln!("Failed to suppress errors in {} files:", failures.len());
        for (path, e) in failures {
            eprintln!("  {path:#?}: {e}");
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

pub fn remove_unused_ignores(path_ignores: SmallMap<&PathBuf, SmallSet<LineNumber>>) {
    let regex = Regex::new(r"# pyrefly: ignore.*$").unwrap();
    let mut removed_ignores: SmallMap<&PathBuf, usize> = SmallMap::new();
    for (path, ignores) in path_ignores {
        let mut unused_ignore_count = 0;
        let zero_index_ignores: SmallSet<usize> = ignores
            .iter()
            .map(|i| i.to_zero_indexed() as usize)
            .collect();
        if let Ok(file) = read_and_validate_file(path) {
            let mut buf = String::with_capacity(file.len());
            let lines = file.lines();
            for (idx, line) in lines.enumerate() {
                if zero_index_ignores.contains(&idx) {
                    unused_ignore_count += 1;
                    // TODO: Expand support of what we remove and thoroughly test
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
            if let Err(e) = fs_anyhow::write(path, buf.as_bytes()) {
                error!("Failed to remove unused error suppressions in {} files:", e);
            } else if unused_ignore_count > 0 {
                removed_ignores.insert(path, unused_ignore_count);
            }
        }
    }
    eprintln!(
        "Removed {} unused error suppression(s) in {} file(s)",
        removed_ignores.values().sum::<usize>(),
        removed_ignores.len(),
    );
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;
    use std::sync::Arc;

    use pretty_assertions::assert_str_eq;
    use pyrefly_python::module_name::ModuleName;
    use pyrefly_python::module_path::ModulePath;
    use pyrefly_util::lined_buffer::DisplayPos;
    use ruff_text_size::TextRange;
    use ruff_text_size::TextSize;
    use tempfile;
    use vec1::Vec1;

    use super::*;
    use crate::error::kind::ErrorKind;
    use crate::module::module_info::ModuleInfo;

    fn error(path: PathBuf, row: usize, column: usize, error_kind: ErrorKind) -> Error {
        let contents = format!("{}{}", "\n".repeat(row - 1), " ".repeat(column - 1));
        let pos = TextSize::new(contents.len() as u32);
        let e = Error::new(
            ModuleInfo::new(
                ModuleName::unknown(),
                ModulePath::filesystem(path),
                Arc::new(contents),
            ),
            TextRange::new(pos, pos),
            Vec1::new("test message".to_owned()),
            error_kind,
        );
        assert_eq!(
            e.display_range().start,
            DisplayPos {
                line: LineNumber::new(row as u32).unwrap(),
                column: NonZeroU32::new(column as u32).unwrap()
            }
        );
        e
    }

    fn test_remove_suppressions(lines: SmallSet<LineNumber>, input: &str, want: &str) {
        let tdir = tempfile::tempdir().unwrap();
        let path = tdir.path().join("test.py");
        fs_anyhow::write(&path, input.as_bytes()).unwrap();
        let map = SmallMap::from_iter([(&path, lines)]);
        remove_unused_ignores(map);
        let got_file = fs_anyhow::read_to_string(&path).unwrap();
        assert_str_eq!(want, got_file);
    }

    fn assert_suppress_errors(
        path_errors: Vec<(usize, usize, ErrorKind)>,
        input: &str,
        output: &str,
    ) {
        let tdir = tempfile::tempdir().unwrap();
        let path = tdir.path().join("test.py");
        fs_anyhow::write(&path, input.as_bytes()).unwrap();
        let errors = {
            let mut e = SmallMap::new();
            e.insert(
                path.clone(),
                path_errors
                    .into_iter()
                    .map(|x| error(path.clone(), x.0, x.1, x.2))
                    .collect(),
            );
            e
        };
        let (failures, successes) = add_suppressions(&errors);
        assert!(failures.is_empty());
        assert_eq!(vec![&path], successes);
        let got_file = fs_anyhow::read_to_string(&path).unwrap();
        assert_str_eq!(output, got_file);
    }

    fn assert_no_error_suppression(
        path_errors: Vec<(usize, usize, ErrorKind)>,
        input: &str,
        output: &str,
    ) {
        let tdir = tempfile::tempdir().unwrap();
        let path = tdir.path().join("test.py");
        fs_anyhow::write(&path, input.as_bytes()).unwrap();
        let errors = {
            let mut e = SmallMap::new();
            e.insert(
                path.clone(),
                path_errors
                    .into_iter()
                    .map(|x| error(path.clone(), x.0, x.1, x.2))
                    .collect(),
            );
            e
        };
        let (failures, _successes) = add_suppressions(&errors);
        assert!(!failures.is_empty());
        let got_file = fs_anyhow::read_to_string(&path).unwrap();
        assert_str_eq!(output, got_file);
    }

    #[test]
    fn test_add_suppressions() {
        assert_suppress_errors(
            vec![
                (1, 10, ErrorKind::BadAssignment),
                (6, 7, ErrorKind::BadArgumentType),
                (7, 10, ErrorKind::BadReturn),
                (10, 3, ErrorKind::BadArgumentType),
            ],
            r#"x: str = 1


def f(y: int) -> None:
    """Doc comment"""
    x = "one" + y
    return x


f(x)

"#,
            r#"# pyrefly: ignore  # bad-assignment
x: str = 1


def f(y: int) -> None:
    """Doc comment"""
    # pyrefly: ignore  # bad-argument-type
    x = "one" + y
    # pyrefly: ignore  # bad-return
    return x


# pyrefly: ignore  # bad-argument-type
f(x)

"#,
        );
    }
    #[test]
    fn test_add_suppressions_existing_comment() {
        assert_suppress_errors(
            vec![(3, 10, ErrorKind::BadAssignment)],
            r#"
# comment
def foo() -> None: pass
"#,
            r#"
# comment
# pyrefly: ignore  # bad-assignment
def foo() -> None: pass
"#,
        );
    }

    #[test]
    fn test_add_suppressions_duplicate_errors() {
        assert_suppress_errors(
            vec![
                (3, 10, ErrorKind::BadAssignment),
                (3, 10, ErrorKind::BadAssignment),
            ],
            r#"
# comment
def foo() -> None: pass
"#,
            r#"
# comment
# pyrefly: ignore  # bad-assignment
def foo() -> None: pass
"#,
        );
    }

    #[test]
    fn test_add_suppressions_multiple_errors_one_line() {
        assert_suppress_errors(
            vec![
                (3, 10, ErrorKind::BadAssignment),
                (3, 10, ErrorKind::TypeAliasError),
            ],
            r#"
# comment
def foo() -> None: pass
"#,
            r#"
# comment
# pyrefly: ignore  # bad-assignment, type-alias-error
def foo() -> None: pass
"#,
        );
    }

    #[test]
    fn test_add_suppressions_unparseable_line_break() {
        assert_no_error_suppression(
            vec![(3, 10, ErrorKind::BadAssignment)],
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
{}

def bar() -> None: 
pass 
    "#,
            GENERATED_TOKEN
        );
        assert_no_error_suppression(
            vec![(1, 10, ErrorKind::BadAssignment)],
            &file_contents,
            &file_contents,
        );
    }

    #[test]
    fn test_remove_suppression_above() {
        let lines = SmallSet::from_iter([LineNumber::new(3).unwrap()]);
        let input = r#"
def f() -> int:
    # pyrefly: ignore # bad-return
    return 1
"#;
        let want = r#"
def f() -> int:

    return 1
"#;
        test_remove_suppressions(lines, input, want);
    }

    #[test]
    fn test_remove_suppression_above_two() {
        let lines = SmallSet::from_iter([LineNumber::new(3).unwrap()]);
        let input = r#"
def g() -> str:
    # pyrefly: ignore # bad-return
    return "hello"
"#;
        let want = r#"
def g() -> str:

    return "hello"
"#;
        test_remove_suppressions(lines, input, want);
    }

    #[test]
    fn test_remove_suppression_inline() {
        let lines = SmallSet::from_iter([LineNumber::new(3).unwrap()]);
        let input = r#"
def g() -> str:
    return "hello" # pyrefly: ignore # bad-return
"#;
        let want = r#"
def g() -> str:
    return "hello"
"#;
        test_remove_suppressions(lines, input, want);
    }

    #[test]
    fn test_remove_suppression_multiple() {
        let lines = SmallSet::from_iter([LineNumber::new(3).unwrap(), LineNumber::new(5).unwrap()]);
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
        test_remove_suppressions(lines, input, output);
    }

    #[test]
    fn test_no_remove_suppression_generated() {
        let lines = SmallSet::from_iter([LineNumber::new(3).unwrap(), LineNumber::new(5).unwrap()]);
        let input = format!(
            r#"
{}
def g() -> str:
    return "hello" # pyrefly: ignore # bad-return
def f() -> int:
    # pyrefly: ignore
    return 1
"#,
            GENERATED_TOKEN
        );
        test_remove_suppressions(lines, &input, &input);
    }
}
