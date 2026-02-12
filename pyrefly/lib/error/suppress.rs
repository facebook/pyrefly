/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashSet;
use std::path::Path;
use std::path::PathBuf;
use std::sync::LazyLock;

use anyhow::anyhow;
use pyrefly_config::error_kind::ErrorKind;
use pyrefly_python::ast::Ast;
use pyrefly_python::ignore::find_comment_start_in_line;
use pyrefly_python::module::GENERATED_TOKEN;
use pyrefly_python::module_path::ModulePathDetails;
use pyrefly_util::fs_anyhow;
use regex::Regex;
use ruff_python_ast::PySourceType;
use serde::Deserialize;
use serde::Serialize;
use starlark_map::small_map::SmallMap;
use starlark_map::small_set::SmallSet;
use tracing::info;

use crate::error::error::Error;

/// Regex to match pyrefly/type ignore comments with optional error codes and trailing semicolon.
/// Preserves any following comments (e.g., "# pyrefly: ignore [x]; # other" -> "# other").
static IGNORE_COMMENT_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"#\s*pyrefly:\s*ignore\s*(\[[^\]]*\])?\s*;?\s*|#\s*type:\s*ignore\s*(\[[^\]]*\])?\s*;?\s*",
    )
    .unwrap()
});

/// A serializable representation of an error for JSON input/output.
/// This struct holds the fields needed to add or remove a suppression comment.
#[derive(Deserialize, Serialize)]
pub struct SerializedError {
    /// The file path where the error occurs.
    pub path: PathBuf,
    /// The 0-indexed line number where the error occurs.
    pub line: usize,
    /// The kebab-case name of the error kind (e.g., "bad-assignment").
    pub name: String,
    /// The error message. Used for UnusedIgnore errors to determine what to remove.
    pub message: String,
}

impl SerializedError {
    /// Creates a SerializedError from an internal Error.
    /// Returns None if the error is not from a filesystem path.
    pub fn from_error(error: &Error) -> Option<Self> {
        if let ModulePathDetails::FileSystem(path) = error.path().details() {
            Some(Self {
                path: (**path).clone(),
                line: error
                    .display_range()
                    .start
                    .line_within_file()
                    .to_zero_indexed() as usize,
                name: error.error_kind().to_name().to_owned(),
                message: error.msg().to_owned(),
            })
        } else {
            None
        }
    }

    /// Returns true if this error is an UnusedIgnore error.
    pub fn is_unused_ignore(&self) -> bool {
        self.name == ErrorKind::UnusedIgnore.to_name()
    }
}

/// Detects the line ending style used in a string.
/// Returns "\r\n" if CRLF is detected, otherwise returns "\n".
fn detect_line_ending(content: &str) -> &'static str {
    if content.contains("\r\n") {
        "\r\n"
    } else {
        "\n"
    }
}

/// Combines all errors that affect one line into a single entry.
/// The current format is: `# pyrefly: ignore [error1, error2, ...]`
fn dedup_errors(errors: &[SerializedError]) -> SmallMap<usize, String> {
    let mut deduped_errors: SmallMap<usize, HashSet<String>> = SmallMap::new();
    for error in errors {
        deduped_errors
            .entry(error.line)
            .or_default()
            .insert(error.name.clone());
    }
    let mut formatted_errors = SmallMap::new();
    for (line, error_set) in deduped_errors {
        let mut error_codes: Vec<_> = error_set.into_iter().collect();
        error_codes.sort();
        let error_codes_str = error_codes.join(", ");
        let comment = format!("# pyrefly: ignore [{}]", error_codes_str);
        formatted_errors.insert(line, comment);
    }
    formatted_errors
}

// TODO: In future have this return an ast as well as the string for comparison
fn read_and_validate_file(path: &Path) -> anyhow::Result<String> {
    let source_type = if path.extension().and_then(|e| e.to_str()) == Some("ipynb") {
        return Err(anyhow!("Cannot suppress errors in notebook file"));
    } else {
        PySourceType::Python
    };
    let file = fs_anyhow::read_to_string(path);
    match file {
        Ok(file) => {
            // Check for generated + parsable files
            let (_ast, parse_errors, _unsupported_syntax_errors) = Ast::parse(&file, source_type);
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

/// Extracts error codes from an existing pyrefly ignore comment.
/// Returns Some(Vec<String>) if the line contains a valid ignore comment, None otherwise.
/// Uses string-aware parsing to avoid matching inside string literals.
fn parse_ignore_comment(line: &str) -> Option<Vec<String>> {
    let comment_start = find_comment_start_in_line(line)?;
    let comment_part = &line[comment_start..];
    let regex = Regex::new(r"#\s*pyrefly:\s*ignore\s*\[([^\]]*)\]").unwrap();
    regex.captures(comment_part).map(|caps| {
        caps.get(1)
            .map(|m| {
                m.as_str()
                    .split(',')
                    .map(|s| s.trim().to_owned())
                    .filter(|s| !s.is_empty())
                    .collect()
            })
            .unwrap_or_default()
    })
}

/// Location where a suppression comment exists relative to an error line.
enum SuppressionLocation {
    Inline,
    Above,
}

/// Finds an existing suppression comment near the error line.
/// Checks inline first, then above.
fn find_existing_suppression(
    error_line: usize,
    lines: &[&str],
    existing_suppressions: &SmallMap<usize, Vec<String>>,
) -> Option<(SuppressionLocation, Vec<String>)> {
    // Check inline
    if let Some(codes) = existing_suppressions.get(&error_line) {
        return Some((SuppressionLocation::Inline, codes.clone()));
    }

    // Check above
    if error_line > 0
        && let Some(codes) = existing_suppressions.get(&(error_line - 1))
    {
        let above_line = lines[error_line - 1];
        if above_line.trim_start().starts_with("#") {
            return Some((SuppressionLocation::Above, codes.clone()));
        }
    }

    None
}

/// Extracts the leading whitespace from a line for indentation matching.
fn get_indentation(line: &str) -> &str {
    if let Some(first_char) = line.find(|c: char| !c.is_whitespace()) {
        &line[..first_char]
    } else {
        ""
    }
}

/// Merges new error codes with existing ones in a suppression comment.
/// Returns the updated comment string with merged and sorted error codes.
fn merge_error_codes(existing_codes: Vec<String>, new_codes: &[String]) -> String {
    let mut all_codes: SmallSet<String> = SmallSet::new();
    for code in existing_codes {
        all_codes.insert(code);
    }
    for code in new_codes {
        all_codes.insert(code.clone());
    }
    let mut sorted_codes: Vec<_> = all_codes.into_iter().collect();
    sorted_codes.sort();
    format!("# pyrefly: ignore [{}]", sorted_codes.join(", "))
}

/// Replaces the ignore comment in a line with the merged version.
/// Preserves the rest of the line content.
/// Uses string-aware parsing to only replace in the comment portion.
fn replace_ignore_comment(line: &str, merged_comment: &str) -> String {
    if let Some(comment_start) = find_comment_start_in_line(line) {
        let code_part = &line[..comment_start];
        let comment_part = &line[comment_start..];
        let regex = Regex::new(r"#\s*pyrefly:\s*ignore\s*\[[^\]]*\]").unwrap();
        format!(
            "{}{}",
            code_part,
            regex.replace(comment_part, merged_comment)
        )
    } else {
        line.to_owned()
    }
}

/// Adds error suppressions for the given errors in the given files.
/// Returns a list of files that failed to be patched, and a list of files that were patched.
/// The list of failures includes the error that occurred, which may be a read or write error.
fn add_suppressions(
    path_errors: &SmallMap<PathBuf, Vec<SerializedError>>,
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
        let mut deduped_errors = dedup_errors(errors);

        // Pre-scan to find existing suppressions and merge with new error codes
        let lines: Vec<&str> = file.lines().collect();

        // Build a map of lines that have existing suppressions
        let mut existing_suppressions: SmallMap<usize, Vec<String>> = SmallMap::new();
        for (idx, line) in lines.iter().enumerate() {
            if let Some(codes) = parse_ignore_comment(line) {
                existing_suppressions.insert(idx, codes);
            }
        }

        // Track which suppression lines should be skipped because they're being merged
        let mut lines_to_skip: SmallSet<usize> = SmallSet::new();
        // Track which error lines have inline suppressions that were merged (so we replace inline)
        let mut has_inline_suppression: SmallSet<usize> = SmallSet::new();

        // Merge existing suppressions with new ones
        for (&error_line, new_comment) in deduped_errors.iter_mut() {
            let new_codes = extract_error_codes(new_comment);

            if let Some((location, existing_codes)) =
                find_existing_suppression(error_line, &lines, &existing_suppressions)
            {
                *new_comment = merge_error_codes(existing_codes, &new_codes);

                match location {
                    SuppressionLocation::Above => {
                        lines_to_skip.insert(error_line - 1);
                    }
                    SuppressionLocation::Inline => {
                        has_inline_suppression.insert(error_line);
                    }
                }
            }
        }

        let line_ending = detect_line_ending(&file);
        let mut buf = String::new();
        for (idx, line) in lines.iter().enumerate() {
            // Skip old standalone suppression lines that are being replaced
            if lines_to_skip.contains(&idx) {
                continue;
            }

            // Separate line mode
            if let Some(error_comment) = deduped_errors.get(&idx) {
                // Check if this line had an inline suppression that was merged
                if has_inline_suppression.contains(&idx) {
                    // Replace the inline suppression with the merged version
                    let updated_line = replace_ignore_comment(line, error_comment);
                    buf.push_str(&updated_line);
                    buf.push_str(line_ending);
                } else {
                    // Calculate once whether suppression goes below this line
                    let suppression_below =
                        idx + 1 < lines.len() && lines_to_skip.contains(&(idx + 1));

                    if !suppression_below {
                        // Add suppression line above (normal case)
                        buf.push_str(get_indentation(line));
                        buf.push_str(error_comment);
                        buf.push_str(line_ending);
                    }

                    // Write the current line as-is
                    buf.push_str(line);
                    buf.push_str(line_ending);

                    if suppression_below {
                        // Add suppression line below
                        buf.push_str(get_indentation(lines[idx + 1]));
                        buf.push_str(error_comment);
                        buf.push_str(line_ending);
                    }
                }
            } else {
                // No error on this line, write as-is
                buf.push_str(line);
                buf.push_str(line_ending);
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

/// Extracts error codes from a comment string like "# pyrefly: ignore [code1, code2]".
fn extract_error_codes(comment: &str) -> Vec<String> {
    parse_ignore_comment(comment).unwrap_or_default()
}

/// Suppresses errors by adding ignore comments to source files.
/// Takes a list of SerializedErrors
pub fn suppress_errors(errors: Vec<SerializedError>) {
    let mut path_errors: SmallMap<PathBuf, Vec<SerializedError>> = SmallMap::new();
    for e in errors {
        path_errors.entry(e.path.clone()).or_default().push(e);
    }
    if path_errors.is_empty() {
        info!("No errors to suppress!");
        return;
    }
    info!("Inserting error suppressions...");
    let (failures, successes) = add_suppressions(&path_errors);
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

/// Given a line with a pyrefly ignore comment and sets of used/unused error codes,
/// returns the updated line. If all codes are unused, removes the entire comment.
/// If some codes are used, keeps only the used codes in the comment.
/// Uses string-aware parsing to only modify the comment portion of the line.
fn update_ignore_comment_with_used_codes(
    line: &str,
    used_codes: &SmallSet<String>,
    unused_codes: &SmallSet<String>,
) -> Option<String> {
    // If there are no unused codes, keep the line as-is
    if unused_codes.is_empty() {
        return None;
    }

    let comment_start = find_comment_start_in_line(line)?;
    let code_part = &line[..comment_start];
    let comment_part = &line[comment_start..];

    // If there are no used codes, remove the entire comment
    if used_codes.is_empty() {
        if IGNORE_COMMENT_REGEX.is_match(comment_part) {
            let new_comment = IGNORE_COMMENT_REGEX.replace_all(comment_part, "");
            let result = format!("{}{}", code_part, new_comment);
            return Some(result.trim_end().to_owned());
        }
        return None;
    }

    // Some codes are used, some are unused - rebuild the comment with only used codes
    let regex = Regex::new(r"#\s*pyrefly:\s*ignore\s*\[[^\]]*\]").unwrap();
    if regex.is_match(comment_part) {
        let mut sorted_codes: Vec<_> = used_codes.iter().cloned().collect();
        sorted_codes.sort();
        let new_comment = format!("# pyrefly: ignore [{}]", sorted_codes.join(", "));
        let updated = regex.replace(comment_part, new_comment.as_str());
        return Some(format!("{}{}", code_part, updated));
    }
    None
}

/// Removes unused ignore comments from source files.
/// Takes a list of UnusedIgnore errors (from collect_unused_ignore_errors) and uses
/// the error location and message to determine what to remove:
/// - "Unused `# pyrefly: ignore` comment" -> remove entire comment
/// - "Unused `# pyrefly: ignore` comment for code(s): X, Y" -> remove entire comment
/// - "Unused error code(s) in `# pyrefly: ignore`: X, Y" -> remove only those codes
pub fn remove_unused_ignores(unused_ignore_errors: Vec<Error>) -> usize {
    let serialized: Vec<SerializedError> = unused_ignore_errors
        .iter()
        .filter_map(SerializedError::from_error)
        .collect();
    remove_unused_ignores_from_serialized(serialized)
}

/// Removes unused ignore comments from source files using SerializedError.
/// This is similar to remove_unused_ignores but works with SerializedError instead of Error,
/// allowing it to be used with errors parsed from JSON.
pub fn remove_unused_ignores_from_serialized(unused_ignore_errors: Vec<SerializedError>) -> usize {
    if unused_ignore_errors.is_empty() {
        return 0;
    }

    // Group errors by file path
    let mut errors_by_path: SmallMap<PathBuf, Vec<&SerializedError>> = SmallMap::new();
    for error in &unused_ignore_errors {
        errors_by_path
            .entry(error.path.clone())
            .or_default()
            .push(error);
    }

    let mut removed_ignores: SmallMap<PathBuf, usize> = SmallMap::new();

    for (path, path_errors) in &errors_by_path {
        // Build a map from line number to the error
        let mut line_errors: SmallMap<usize, &SerializedError> = SmallMap::new();
        for error in path_errors {
            line_errors.insert(error.line, *error);
        }

        if let Ok(file) = read_and_validate_file(path) {
            let line_ending = detect_line_ending(&file);
            let mut buf = String::with_capacity(file.len());
            let lines: Vec<&str> = file.lines().collect();
            let mut unused_count = 0;

            for (idx, line) in lines.iter().enumerate() {
                if let Some(error) = line_errors.get(&idx) {
                    // Use string-aware comment detection instead of raw regex
                    if let Some(comment_start) = find_comment_start_in_line(line) {
                        let comment_part = &line[comment_start..];
                        if IGNORE_COMMENT_REGEX.is_match(comment_part) {
                            let msg = &error.message;

                            // Determine action based on error message
                            if msg.starts_with("Unused `# pyrefly: ignore` comment") {
                                // Remove entire comment (blanket unused or all codes unused)
                                let code_part = &line[..comment_start];
                                let new_comment =
                                    IGNORE_COMMENT_REGEX.replace_all(comment_part, "");
                                let new_line = format!("{}{}", code_part, new_comment);
                                let new_line = new_line.trim_end();
                                unused_count += 1;
                                if !new_line.is_empty() {
                                    buf.push_str(new_line);
                                    buf.push_str(line_ending);
                                }
                                continue;
                            } else if msg.starts_with("Unused error code(s)") {
                                // Partially unused - extract codes from message and remove only those
                                // Message format: "Unused error code(s) in `# pyrefly: ignore`: code1, code2"
                                if let Some(codes_part) = msg.split(": ").last() {
                                    let unused_codes: SmallSet<String> = codes_part
                                        .split(", ")
                                        .map(|s| s.trim().to_owned())
                                        .collect();

                                    if let Some(existing_codes) = parse_ignore_comment(line) {
                                        let used_codes: SmallSet<String> = existing_codes
                                            .into_iter()
                                            .filter(|c| !unused_codes.contains(c))
                                            .collect();

                                        if let Some(updated) = update_ignore_comment_with_used_codes(
                                            line,
                                            &used_codes,
                                            &unused_codes,
                                        ) {
                                            unused_count += 1;
                                            if !updated.trim().is_empty() {
                                                buf.push_str(&updated);
                                                buf.push_str(line_ending);
                                            }
                                            continue;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                buf.push_str(line);
                buf.push_str(line_ending);
            }

            // Write the modified content back to the file
            if unused_count > 0 && fs_anyhow::write(path, buf).is_ok() {
                removed_ignores.insert(path.clone(), unused_count);
            }
        }
    }

    let removals = removed_ignores.values().sum::<usize>();
    info!(
        "Removed {} unused error suppression(s) in {} file(s)",
        removals,
        removed_ignores.len(),
    );
    removals
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::Arc;

    use dupe::Dupe;
    use pyrefly_build::handle::Handle;
    use pyrefly_config::error_kind::Severity;
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
    use crate::state::errors::Errors;
    use crate::state::load::FileContents;
    use crate::state::require::Require;
    use crate::state::state::State;

    fn get_path(tdir: &TempDir) -> PathBuf {
        tdir.path().join("test.py")
    }

    fn assert_suppress_errors(before: &str, after: &str) {
        let (errors, tdir) = get_errors(before);
        let suppressable_errors: Vec<SerializedError> = errors
            .collect_errors()
            .shown
            .iter()
            .filter(|e| e.severity() >= Severity::Warn)
            .filter_map(SerializedError::from_error)
            .collect();
        suppress::suppress_errors(suppressable_errors);
        let got_file = fs_anyhow::read_to_string(&get_path(&tdir)).unwrap();
        assert_eq!(after, got_file);
    }

    fn assert_remove_ignores(before: &str, after: &str, expected_removals: usize) {
        let (errors, tdir) = get_errors(before);
        let unused_errors = errors.collect_unused_ignore_errors();
        let removals = suppress::remove_unused_ignores(unused_errors);
        let got_file = fs_anyhow::read_to_string(&get_path(&tdir)).unwrap();
        assert_eq!(after, got_file);
        assert_eq!(removals, expected_removals);
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
# pyrefly: ignore [bad-assignment]
x: str = 1


def f(y: int) -> None:
    """Doc comment"""
    # pyrefly: ignore [unsupported-operation]
    x = "one" + y
    return x


# pyrefly: ignore [bad-argument-type]
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
    # pyrefly: ignore [bad-return]
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
# pyrefly: ignore [bad-return]
def foo() -> int: pass
"#,
        );
    }

    #[test]
    fn test_add_suppressions_multiple_errors_update_ignore() {
        assert_suppress_errors(
            r#"
def foo() -> str:
    # pyrefly: ignore [unsupported-operation]
    return 1 + []
"#,
            r#"
def foo() -> str:
    # pyrefly: ignore [bad-return, unsupported-operation]
    return 1 + []
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
# pyrefly: ignore [bad-argument-type, bad-assignment]
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
    # pyrefly: ignore [bad-return]
    return 1
"#;
        let want = r#"
def f() -> int:
    return 1
"#;
        assert_remove_ignores(input, want, 1);
    }

    #[test]
    fn test_remove_suppression_above_two() {
        let input = r#"
def g() -> str:
    # pyrefly: ignore [bad-return]
    return "hello"
"#;
        let want = r#"
def g() -> str:
    return "hello"
"#;
        assert_remove_ignores(input, want, 1);
    }

    #[test]
    fn test_remove_suppression_inline() {
        let input = r#"
def g() -> str:
    return "hello" # pyrefly: ignore [bad-return]
"#;
        let want = r#"
def g() -> str:
    return "hello"
"#;
        assert_remove_ignores(input, want, 1);
    }

    #[test]
    fn test_remove_suppression_multiple() {
        let input = r#"
def g() -> str:
    return "hello" # pyrefly: ignore [bad-return]
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
        assert_remove_ignores(input, output, 2);
    }

    #[test]
    fn test_errors_deduped() {
        let file_contents = r#"
# pyrefly: ignore [bad-return]
def bar(x: int, y: str) -> int:
    pass

bar("", 1)
"#;

        let after = r#"
# pyrefly: ignore [bad-return]
def bar(x: int, y: str) -> int:
    pass

# pyrefly: ignore [bad-argument-type]
bar("", 1)
"#;
        assert_suppress_errors(file_contents, after);
    }

    #[test]
    fn test_do_not_remove_suppression_needed() {
        // We should not remove this suppression, since it is needed.
        let input = r#"
def foo(s: str) -> int:
    pass

def bar(x: int) -> int:
    pass


foo(
    bar( # pyrefly: ignore [bad-argument-type]
        12323423423
    )
)
foo(
    # pyrefly: ignore [bad-argument-type]
    bar(
        12323423423
    )
)
"#;
        assert_remove_ignores(input, input, 0);
    }

    #[test]
    fn test_remove_suppression_first_line() {
        let input = r#"x = 1 + 1  # pyrefly: ignore
"#;
        let want = r#"x = 1 + 1
"#;
        assert_remove_ignores(input, want, 1);
    }

    #[test]
    fn test_remove_first_unused_ignore_only() {
        // Regression test for https://github.com/facebook/pyrefly/issues/1310
        let input = r#""""Test."""
x = 1 + 1  # pyrefly: ignore
y = 1 + "oops"  # pyrefly: ignore
"#;
        let want = r#""""Test."""
x = 1 + 1
y = 1 + "oops"  # pyrefly: ignore
"#;
        assert_remove_ignores(input, want, 1);
    }

    #[test]
    fn test_remove_unused_suppression_within_multiline_error_range() {
        // Test that an unused suppression within a multi-line error's range is removed.
        // The bad-argument-type error spans the multi-line argument expression, but only
        // a suppression at the error's start line is used. An unrelated suppression on a
        // different line within the error's range should be removed.
        let input = r#"
def foo(s: str) -> int:
    pass

foo(
    # pyrefly: ignore [bad-argument-type]
    1 +
    # pyrefly: ignore [bad-return]
    2
)
"#;
        let want = r#"
def foo(s: str) -> int:
    pass

foo(
    # pyrefly: ignore [bad-argument-type]
    1 +
    2
)
"#;
        assert_remove_ignores(input, want, 1);
    }

    #[test]
    fn test_keep_both_same_line_ignores() {
        let input = r#"
class A:
    x = 1 + "oops"  # pyrefly: ignore[unsupported-operation]
    y: int = ""  # pyrefly: ignore[bad-assignment]
"#;
        let want = r#"
class A:
    x = 1 + "oops"  # pyrefly: ignore[unsupported-operation]
    y: int = ""  # pyrefly: ignore[bad-assignment]
"#;
        assert_remove_ignores(input, want, 0);
    }

    #[test]
    fn test_no_remove_suppression_generated() {
        let input = format!(
            r#"
{GENERATED_TOKEN}
def g() -> str:
    return "hello" # pyrefly: ignore [bad-return]
def f() -> int:
    # pyrefly: ignore
    return 1
"#,
        );
        assert_remove_ignores(&input, &input, 0);
    }

    #[test]
    fn test_no_remove_suppression() {
        let input = r#"
def g() -> int:
    return "hello" # pyrefly: ignore [bad-return]"#;
        // No trailing newline on purpose.
        // Ensures files with only used suppressions are not rewritten (no newline added).
        // https://github.com/facebook/pyrefly/issues/2185
        assert_remove_ignores(input, input, 0);
    }

    #[test]
    fn test_remove_unused_ignores_no_ignores() {
        let input = r#"
def f(x: int) -> int:
    return x + 1"#;
        // No trailing newline on purpose.
        // Ensures files without suppressions are not rewritten (no newline added).
        // https://github.com/facebook/pyrefly/issues/2185
        assert_remove_ignores(input, input, 0);
    }

    #[test]
    fn test_remove_unused_ignores_existing_comment() {
        let input = r#"
def f(x: int) -> int:
    # noqa: E501,RUF100  # pyrefly: ignore[unsupported-operation]  # ty: ignore[not-subscriptable]
    return x + 1"#;
        let after = r#"
def f(x: int) -> int:
    # noqa: E501,RUF100  # ty: ignore[not-subscriptable]
    return x + 1
"#;

        assert_remove_ignores(input, after, 1);
    }

    #[test]
    fn test_strip_unused_error_code_from_multi_code_suppression() {
        // Only bad-assignment is used, bad-override should be stripped
        let before = r#"
# pyrefly: ignore[bad-assignment,bad-override]
a: int = ""
"#;
        let after = r#"
# pyrefly: ignore [bad-assignment]
a: int = ""
"#;
        assert_remove_ignores(before, after, 1);
    }

    #[test]
    fn test_strip_unused_error_codes_keeps_all_used() {
        // Both error codes are used, nothing should be stripped
        let before = r#"
def g() -> str:
    # pyrefly: ignore [bad-return, unsupported-operation]
    return 1 + []
"#;
        assert_remove_ignores(before, before, 0);
    }

    #[test]
    fn test_strip_unused_error_code_inline() {
        // Test inline comment with partial unused codes
        let before = r#"
a: int = "" # pyrefly: ignore[bad-assignment, bad-override]
"#;
        let after = r#"
a: int = "" # pyrefly: ignore [bad-assignment]
"#;
        assert_remove_ignores(before, after, 1);
    }
    #[test]
    fn test_parse_ignore_comment() {
        let line = "    # pyrefly: ignore [unsupported-operation]";
        let codes = parse_ignore_comment(line);
        assert_eq!(codes, Some(vec!["unsupported-operation".to_owned()]));

        let line2 = "    # pyrefly: ignore [bad-return, unsupported-operation]";
        let codes2 = parse_ignore_comment(line2);
        assert_eq!(
            codes2,
            Some(vec![
                "bad-return".to_owned(),
                "unsupported-operation".to_owned()
            ])
        );

        let line3 = "    return 1 + []";
        let codes3 = parse_ignore_comment(line3);
        assert_eq!(codes3, None);
    }

    #[test]
    fn test_merge_error_codes() {
        let existing = vec!["unsupported-operation".to_owned()];
        let new = vec!["bad-return".to_owned()];
        let merged = merge_error_codes(existing, &new);
        assert_eq!(
            merged,
            "# pyrefly: ignore [bad-return, unsupported-operation]"
        );
    }

    #[test]
    fn test_detect_line_ending() {
        assert_eq!(detect_line_ending("line1\nline2\n"), "\n");
        assert_eq!(detect_line_ending("line1\r\nline2\r\n"), "\r\n");
        assert_eq!(detect_line_ending("single line"), "\n");
        assert_eq!(detect_line_ending("mixed\r\nlines\n"), "\r\n");
    }

    #[test]
    fn test_remove_unused_ignores_preserves_crlf_line_endings() {
        let input = "def g() -> str:\r\n    return \"hello\" # pyrefly: ignore [bad-return]\r\n";
        let want = "def g() -> str:\r\n    return \"hello\"\r\n";
        assert_remove_ignores(input, want, 1);
    }

    #[test]
    fn test_add_suppressions_preserves_crlf_line_endings() {
        let before = "\r\nx: str = 1\r\n";
        let after = "\r\n# pyrefly: ignore [bad-assignment]\r\nx: str = 1\r\n";
        assert_suppress_errors(before, after);
    }

    // Helper function to test remove_unused_ignores_from_serialized
    fn assert_remove_ignores_from_serialized(
        file_content: &str,
        mut serialized_errors: Vec<SerializedError>,
        expected_content: &str,
        expected_removals: usize,
    ) {
        let tdir = tempfile::tempdir().unwrap();
        let path = get_path(&tdir);
        fs_anyhow::write(&path, file_content).unwrap();

        // Update error paths to point to the temp file
        for error in &mut serialized_errors {
            error.path = path.clone();
        }

        let removals = suppress::remove_unused_ignores_from_serialized(serialized_errors);

        let got_file = fs_anyhow::read_to_string(&path).unwrap();
        assert_eq!(expected_content, got_file);
        assert_eq!(removals, expected_removals);
    }

    #[test]
    fn test_remove_unused_ignores_from_serialized_blanket() {
        let input = r#"def g() -> str:
    return "hello" # pyrefly: ignore
"#;
        let want = r#"def g() -> str:
    return "hello"
"#;
        let errors = vec![SerializedError {
            path: PathBuf::from("test.py"),
            line: 1,
            name: "unused-ignore".to_owned(),
            message: "Unused `# pyrefly: ignore` comment".to_owned(),
        }];
        assert_remove_ignores_from_serialized(input, errors, want, 1);
    }

    #[test]
    fn test_remove_unused_ignores_from_serialized_partial() {
        let before = r#"
# pyrefly: ignore[bad-assignment,bad-override]
a: int = ""
"#;
        let after = r#"
# pyrefly: ignore [bad-assignment]
a: int = ""
"#;
        let errors = vec![SerializedError {
            path: PathBuf::from("test.py"),
            line: 1,
            name: "unused-ignore".to_owned(),
            message: "Unused error code(s) in `# pyrefly: ignore`: bad-override".to_owned(),
        }];
        assert_remove_ignores_from_serialized(before, errors, after, 1);
    }

    #[test]
    fn test_remove_unused_ignores_from_serialized_multiple_codes() {
        let before = r#"
def foo() -> str:
    # pyrefly: ignore [bad-return, unsupported-operation, bad-assignment]
    return 1 + []
"#;
        let after = r#"
def foo() -> str:
    # pyrefly: ignore [bad-return, unsupported-operation]
    return 1 + []
"#;
        let errors = vec![SerializedError {
            path: PathBuf::from("test.py"),
            line: 2,
            name: "unused-ignore".to_owned(),
            message: "Unused error code(s) in `# pyrefly: ignore`: bad-assignment".to_owned(),
        }];
        assert_remove_ignores_from_serialized(before, errors, after, 1);
    }

    #[test]
    fn test_remove_unused_ignores_from_serialized_inline() {
        let input = r#"
def g() -> str:
    return "hello" # pyrefly: ignore [bad-return]
"#;
        let want = r#"
def g() -> str:
    return "hello"
"#;
        let errors = vec![SerializedError {
            path: PathBuf::from("test.py"),
            line: 2,
            name: "unused-ignore".to_owned(),
            message: "Unused `# pyrefly: ignore` comment for code(s): bad-return".to_owned(),
        }];
        assert_remove_ignores_from_serialized(input, errors, want, 1);
    }

    #[test]
    fn test_remove_unused_ignores_from_serialized_multiple_files() {
        let tdir = tempfile::tempdir().unwrap();
        let path1 = tdir.path().join("file1.py");
        let path2 = tdir.path().join("file2.py");

        let content1 = "x = 1  # pyrefly: ignore\n";
        let content2 = "y = 2  # pyrefly: ignore\n";

        fs_anyhow::write(&path1, content1).unwrap();
        fs_anyhow::write(&path2, content2).unwrap();

        let errors = vec![
            SerializedError {
                path: path1.clone(),
                line: 0,
                name: "unused-ignore".to_owned(),
                message: "Unused `# pyrefly: ignore` comment".to_owned(),
            },
            SerializedError {
                path: path2.clone(),
                line: 0,
                name: "unused-ignore".to_owned(),
                message: "Unused `# pyrefly: ignore` comment".to_owned(),
            },
        ];

        let removals = suppress::remove_unused_ignores_from_serialized(errors);

        assert_eq!(fs_anyhow::read_to_string(&path1).unwrap(), "x = 1\n");
        assert_eq!(fs_anyhow::read_to_string(&path2).unwrap(), "y = 2\n");
        assert_eq!(removals, 2);
    }

    #[test]
    fn test_remove_unused_ignores_from_serialized_empty_list() {
        let errors: Vec<SerializedError> = vec![];
        let removals = suppress::remove_unused_ignores_from_serialized(errors);
        assert_eq!(removals, 0);
    }

    #[test]
    fn test_remove_unused_ignores_from_serialized_preserves_crlf() {
        let input = "def g() -> str:\r\n    return \"hello\" # pyrefly: ignore [bad-return]\r\n";
        let want = "def g() -> str:\r\n    return \"hello\"\r\n";
        let errors = vec![SerializedError {
            path: PathBuf::from("test.py"),
            line: 1,
            name: "unused-ignore".to_owned(),
            message: "Unused `# pyrefly: ignore` comment".to_owned(),
        }];
        assert_remove_ignores_from_serialized(input, errors, want, 1);
    }

    #[test]
    fn test_remove_ignores_preserves_string_literal() {
        // A string literal containing "# pyrefly: ignore" should not be modified.
        // Only the real unused ignore comment on a different line should be removed.
        let input = r##"
x = "# pyrefly: ignore [bad-override]"
y = 1 + 1  # pyrefly: ignore
"##;
        let want = r##"
x = "# pyrefly: ignore [bad-override]"
y = 1 + 1
"##;
        assert_remove_ignores(input, want, 1);
    }

    #[test]
    fn test_remove_ignores_string_literal_same_line() {
        // A line with both a string literal containing "# pyrefly: ignore" and a real
        // inline unused ignore comment. Only the comment should be removed.
        let input = r##"x = "# pyrefly: ignore [bad-override]"  # pyrefly: ignore
"##;
        let want = r##"x = "# pyrefly: ignore [bad-override]"
"##;
        let errors = vec![SerializedError {
            path: PathBuf::from("test.py"),
            line: 0,
            name: "unused-ignore".to_owned(),
            message: "Unused `# pyrefly: ignore` comment".to_owned(),
        }];
        assert_remove_ignores_from_serialized(input, errors, want, 1);
    }

    #[test]
    fn test_parse_ignore_comment_ignores_string_literal() {
        // parse_ignore_comment should not match ignore comments inside string literals
        let line = r##"x = "# pyrefly: ignore [bad-override]""##;
        assert_eq!(parse_ignore_comment(line), None);

        // But it should still match real comments
        let line2 = r##"x = "hello"  # pyrefly: ignore [bad-override]"##;
        assert_eq!(
            parse_ignore_comment(line2),
            Some(vec!["bad-override".to_owned()])
        );
    }

    #[test]
    fn test_add_suppressions_ignores_string_literal() {
        // A string literal containing "# pyrefly: ignore" should not be treated as
        // an existing suppression. The error suppression should be added above.
        assert_suppress_errors(
            r##"
x: str = 1
y = "# pyrefly: ignore [bad-assignment]"
"##,
            r##"
# pyrefly: ignore [bad-assignment]
x: str = 1
y = "# pyrefly: ignore [bad-assignment]"
"##,
        );
    }
}
