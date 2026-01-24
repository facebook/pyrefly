/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashSet;
use std::path::Path;
use std::path::PathBuf;

use anyhow::anyhow;
use pyrefly_python::ast::Ast;
use pyrefly_python::ignore::Ignore;
use pyrefly_python::ignore::Tool;
use pyrefly_python::module::GENERATED_TOKEN;
use pyrefly_python::module_path::ModulePathDetails;
use pyrefly_util::fs_anyhow;
use pyrefly_util::lined_buffer::LineNumber;
use regex::Regex;
use ruff_python_ast::PySourceType;
use starlark_map::small_map::SmallMap;
use starlark_map::small_set::SmallSet;
use starlark_map::smallset;
use tracing::error;
use tracing::info;

use crate::error::error::Error;
use crate::state::errors::Errors;

/// A minimal representation of an error for suppression purposes.
/// This struct holds only the fields needed to add or remove a suppression comment.
pub struct SuppressableError {
    /// The file path where the error occurs.
    pub path: PathBuf,
    /// The 0-indexed line number where the error occurs.
    pub line: usize,
    /// The kebab-case name of the error kind (e.g., "bad-assignment").
    pub name: String,
}

impl SuppressableError {
    /// Creates a SuppressableError from an internal Error.
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
            })
        } else {
            None
        }
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
fn dedup_errors(errors: &[SuppressableError]) -> SmallMap<usize, String> {
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
fn parse_ignore_comment(line: &str) -> Option<Vec<String>> {
    let regex = Regex::new(r"#\s*pyrefly:\s*ignore\s*\[([^\]]*)\]").unwrap();
    regex.captures(line).map(|caps| {
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
fn replace_ignore_comment(line: &str, merged_comment: &str) -> String {
    let regex = Regex::new(r"#\s*pyrefly:\s*ignore\s*\[[^\]]*\]").unwrap();
    regex.replace(line, merged_comment).to_string()
}

/// Adds error suppressions for the given errors in the given files.
/// Returns a list of files that failed to be patched, and a list of files that were patched.
/// The list of failures includes the error that occurred, which may be a read or write error.
fn add_suppressions(
    path_errors: &SmallMap<PathBuf, Vec<SuppressableError>>,
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
/// Takes a list of SuppressableErrors
pub fn suppress_errors(errors: Vec<SuppressableError>) {
    let mut path_errors: SmallMap<PathBuf, Vec<SuppressableError>> = SmallMap::new();
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
fn update_ignore_comment_with_used_codes(
    line: &str,
    used_codes: &SmallSet<String>,
    unused_codes: &SmallSet<String>,
) -> Option<String> {
    // If there are no unused codes, keep the line as-is
    if unused_codes.is_empty() {
        return None;
    }

    // If there are no used codes, remove the entire comment
    if used_codes.is_empty() {
        let regex = Regex::new(r"(#\s*pyrefly:\s*ignore.*$|#\s*type:\s*ignore.*$)").unwrap();
        if regex.is_match(line) {
            let new_string = regex.replace_all(line, "");
            return Some(new_string.trim_end().to_owned());
        }
        return None;
    }

    // Some codes are used, some are unused - rebuild the comment with only used codes
    let regex = Regex::new(r"#\s*pyrefly:\s*ignore\s*\[[^\]]*\]").unwrap();
    if regex.is_match(line) {
        let mut sorted_codes: Vec<_> = used_codes.iter().cloned().collect();
        sorted_codes.sort();
        let new_comment = format!("# pyrefly: ignore [{}]", sorted_codes.join(", "));
        let updated = regex.replace(line, new_comment.as_str()).to_string();
        return Some(updated);
    }
    None
}

pub fn remove_unused_ignores(loads: &Errors, all: bool) -> usize {
    let errors = loads.collect_errors();

    // Collect ignores with full suppression data (including comment_line)
    let mut all_ignores: SmallMap<&PathBuf, &Ignore> = SmallMap::new();
    for (module_path, ignore) in loads.collect_ignores() {
        if let ModulePathDetails::FileSystem(path) = module_path.details() {
            if !ignore.is_empty() {
                all_ignores.insert(path, ignore);
            }
        }
    }

    // Track which specific error codes are used on each line (not just line presence)
    // Key: (path, line_number), Value: set of error code names that were suppressed
    let mut suppressed_error_codes: SmallMap<&PathBuf, SmallMap<LineNumber, SmallSet<String>>> =
        SmallMap::new();
    for e in &errors.suppressed {
        if e.is_ignored(&Tool::default_enabled())
            && let ModulePathDetails::FileSystem(path) = e.path().details()
        {
            // Track only the error's start line, not the entire range. Suppressions
            // are matched by start line only (see Ignore::is_ignored), so an error
            // spanning lines 10-20 only "uses" a suppression on line 10.
            let start = e.display_range().start.line_within_file();
            let end = e.display_range().end.line_within_file();
            let error_code = e.error_kind().to_name().to_owned();
            for line_idx in start.to_zero_indexed()..=end.to_zero_indexed() {
                suppressed_error_codes
                    .entry(path)
                    .or_default()
                    .entry(LineNumber::from_zero_indexed(line_idx))
                    .or_default()
                    .insert(error_code.clone());
            }
        }
    }

    let regex = Regex::new(r"(#\s*pyrefly:\s*ignore.*$|#\s*type:\s*ignore.*$)").unwrap();
    let mut removed_ignores: SmallMap<&PathBuf, usize> = SmallMap::new();

    for (path, ignore) in &all_ignores {
        let mut unused_ignore_count = 0;

        // Build a map from comment line number to the line the suppression applies to
        // and the error codes declared in the suppression
        let mut comment_line_info: SmallMap<usize, (LineNumber, SmallSet<String>)> =
            SmallMap::new();

        for (applies_to_line, suppressions) in ignore.iter() {
            for supp in suppressions {
                // Filter to only pyrefly (and type: ignore if all=true)
                let dominated_tools = if all {
                    supp.tool() == Tool::Pyrefly || supp.tool() == Tool::Type
                } else {
                    supp.tool() == Tool::Pyrefly
                };
                if !dominated_tools {
                    continue;
                }

                let comment_idx = supp.comment_line().to_zero_indexed() as usize;
                let declared_codes: SmallSet<String> = supp.error_codes().iter().cloned().collect();

                comment_line_info.insert(comment_idx, (*applies_to_line, declared_codes));
            }
        }

        if let Ok(file) = read_and_validate_file(path) {
            let line_ending = detect_line_ending(&file);
            let mut buf = String::with_capacity(file.len());
            let lines: Vec<&str> = file.lines().collect();
            for (idx, line) in lines.iter().enumerate() {
                if let Some((applies_to_line, declared_codes)) = comment_line_info.get(&idx)
                    && regex.is_match(line)
                {
                    // Get the error codes actually used on the line this suppression applies to
                    let used_codes: SmallSet<String> = suppressed_error_codes
                        .get(path)
                        .and_then(|m| m.get(applies_to_line))
                        .cloned()
                        .unwrap_or_default();

                    // Calculate which codes are unused
                    // If the ignore has no specific codes (suppresses all), treat it as fully unused if no errors
                    let (final_used_codes, unused_codes): (SmallSet<String>, SmallSet<String>) =
                        if declared_codes.is_empty() {
                            // Blanket ignore - if there are any errors, it's used; otherwise unused
                            if used_codes.is_empty() {
                                (SmallSet::new(), smallset! { String::new() })
                            } else {
                                (used_codes.clone(), SmallSet::new())
                            }
                        } else {
                            // Specific codes - find which are used vs unused
                            let used: SmallSet<String> = declared_codes
                                .iter()
                                .filter(|code| used_codes.contains(*code))
                                .cloned()
                                .collect();
                            let unused: SmallSet<String> = declared_codes
                                .iter()
                                .filter(|code| !used_codes.contains(*code))
                                .cloned()
                                .collect();
                            (used, unused)
                        };

                    if let Some(updated_line) = update_ignore_comment_with_used_codes(
                        line,
                        &final_used_codes,
                        &unused_codes,
                    ) {
                        unused_ignore_count += 1;
                        if !updated_line.trim().is_empty() {
                            buf.push_str(&updated_line);
                            buf.push_str(line_ending);
                        }
                        // Skip writing newline if the line becomes empty after removing the ignore
                        continue;
                    }
                }
                buf.push_str(line);
                buf.push_str(line_ending);
            }
            if unused_ignore_count > 0 {
                if let Err(e) = fs_anyhow::write(path, buf) {
                    error!("Failed to remove unused error suppressions in {} files:", e);
                } else {
                    removed_ignores.insert(path, unused_ignore_count);
                }
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
    use crate::state::load::FileContents;
    use crate::state::require::Require;
    use crate::state::state::State;

    fn get_path(tdir: &TempDir) -> PathBuf {
        tdir.path().join("test.py")
    }

    fn assert_suppress_errors(before: &str, after: &str) {
        let (errors, tdir) = get_errors(before);
        let suppressable_errors: Vec<SuppressableError> = errors
            .collect_errors()
            .shown
            .iter()
            .filter(|e| e.severity() >= Severity::Warn)
            .filter_map(SuppressableError::from_error)
            .collect();
        suppress::suppress_errors(suppressable_errors);
        let got_file = fs_anyhow::read_to_string(&get_path(&tdir)).unwrap();
        assert_eq!(after, got_file);
    }

    fn assert_remove_ignores(before: &str, after: &str, all: bool, expected_removals: usize) {
        let (errors, tdir) = get_errors(before);
        let removals = suppress::remove_unused_ignores(&errors, all);
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
        assert_remove_ignores(input, want, false, 1);
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
        assert_remove_ignores(input, want, false, 1);
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
        assert_remove_ignores(input, want, false, 1);
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
        assert_remove_ignores(input, output, false, 2);
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
        assert_remove_ignores(input, input, false, 0);
    }

    #[test]
    fn test_remove_suppression_first_line() {
        let input = r#"x = 1 + 1  # pyrefly: ignore
"#;
        let want = r#"x = 1 + 1
"#;
        assert_remove_ignores(input, want, false, 1);
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
        assert_remove_ignores(input, want, false, 1);
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
        assert_remove_ignores(input, want, false, 1);
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
        assert_remove_ignores(input, want, false, 0);
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
        assert_remove_ignores(&input, &input, false, 0);
    }

    #[test]
    fn test_no_remove_suppression() {
        let input = r#"
def g() -> int:
    return "hello" # pyrefly: ignore [bad-return]"#;
        // No trailing newline on purpose.
        // Ensures files with only used suppressions are not rewritten (no newline added).
        // https://github.com/facebook/pyrefly/issues/2185
        assert_remove_ignores(input, input, false, 0);
    }

    #[test]
    fn test_remove_unused_ignores_no_ignores() {
        let input = r#"
def f(x: int) -> int:
    return x + 1"#;
        // No trailing newline on purpose.
        // Ensures files without suppressions are not rewritten (no newline added).
        // https://github.com/facebook/pyrefly/issues/2185
        assert_remove_ignores(input, input, false, 0);
    }

    #[test]
    fn test_remove_generic_suppression() {
        let before = r#"
def g() -> str:
    return "hello" # type: ignore [bad-return]
"#;
        let after = r#"
def g() -> str:
    return "hello"
"#;
        assert_remove_ignores(before, after, true, 1);
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
        assert_remove_ignores(before, after, true, 1);
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
        assert_remove_ignores(before, after, false, 1);
    }

    #[test]
    fn test_strip_unused_error_codes_keeps_all_used() {
        // Both error codes are used, nothing should be stripped
        let before = r#"
def g() -> str:
    # pyrefly: ignore [bad-return, unsupported-operation]
    return 1 + []
"#;
        assert_remove_ignores(before, before, false, 0);
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
        assert_remove_ignores(before, after, false, 1);
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
        assert_remove_ignores(input, want, false, 1);
    }

    #[test]
    fn test_add_suppressions_preserves_crlf_line_endings() {
        let before = "\r\nx: str = 1\r\n";
        let after = "\r\n# pyrefly: ignore [bad-assignment]\r\nx: str = 1\r\n";
        assert_suppress_errors(before, after);
    }
}
