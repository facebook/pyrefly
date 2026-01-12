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
use pyrefly_config::error_kind::Severity;
use pyrefly_python::ast::Ast;
use pyrefly_python::ignore::Tool;
use pyrefly_python::module::GENERATED_TOKEN;
use pyrefly_python::module_path::ModulePathDetails;
use pyrefly_util::fs_anyhow;
use pyrefly_util::lined_buffer::LineNumber;
use regex::Regex;
use ruff_python_ast::PySourceType;
use starlark_map::small_map::SmallMap;
use starlark_map::small_set::SmallSet;
use tracing::error;
use tracing::info;

use crate::error::error::Error;
use crate::state::errors::Errors;

/// Combines all errors that affect one line into a single entry.
/// The current format is: `# pyrefly: ignore [error1, error2, ...]`
fn dedup_errors(errors: &[Error]) -> SmallMap<usize, String> {
    let mut deduped_errors: SmallMap<usize, HashSet<String>> = SmallMap::new();
    for error in errors {
        let line = error
            .display_range()
            .start
            .line_within_file()
            .to_zero_indexed() as usize;
        let error_name = error.error_kind().to_name().to_owned();
        deduped_errors.entry(line).or_default().insert(error_name);
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
                    buf.push('\n');
                } else {
                    // Calculate once whether suppression goes below this line
                    let suppression_below =
                        idx + 1 < lines.len() && lines_to_skip.contains(&(idx + 1));

                    if !suppression_below {
                        // Add suppression line above (normal case)
                        buf.push_str(get_indentation(line));
                        buf.push_str(error_comment);
                        buf.push('\n');
                    }

                    // Write the current line as-is
                    buf.push_str(line);
                    buf.push('\n');

                    if suppression_below {
                        // Add suppression line below
                        buf.push_str(get_indentation(lines[idx + 1]));
                        buf.push_str(error_comment);
                        buf.push('\n');
                    }
                }
            } else {
                // No error on this line, write as-is
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

/// Extracts error codes from a comment string like "# pyrefly: ignore [code1, code2]".
fn extract_error_codes(comment: &str) -> Vec<String> {
    parse_ignore_comment(comment).unwrap_or_default()
}

pub fn suppress_errors(errors: Vec<Error>) {
    let mut path_errors: SmallMap<PathBuf, Vec<Error>> = SmallMap::new();
    for e in errors {
        if e.severity() >= Severity::Warn
            && let ModulePathDetails::FileSystem(path) = e.path().details()
        {
            path_errors.entry((**path).clone()).or_default().push(e);
        }
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
            // An ignore is unused if there's no error on that line.
            // This matches the is_ignored() logic which checks if a suppression
            // exists on any line within an error's range.
            if !errors.contains(&ignore) {
                unused_ignores.insert(ignore);
            }
        }

        all_unused_ignores.insert(path, unused_ignores);
    }
    all_unused_ignores
}

pub fn remove_unused_ignores(loads: &Errors, all: bool) -> usize {
    let errors = loads.collect_errors();
    let mut all_ignores: SmallMap<&PathBuf, SmallSet<LineNumber>> = SmallMap::new();
    for (module_path, ignore) in loads.collect_ignores() {
        if let ModulePathDetails::FileSystem(path) = module_path.details() {
            all_ignores.insert(path, ignore.get_pyrefly_ignores(all));
        }
    }

    let mut suppressed_errors: SmallMap<&PathBuf, SmallSet<LineNumber>> = SmallMap::new();
    for e in &errors.suppressed {
        if e.is_ignored(&Tool::default_enabled())
            && let ModulePathDetails::FileSystem(path) = e.path().details()
        {
            // Insert all lines in the error's range, not just the start line.
            // This matches the logic in is_ignored() which checks if a suppression
            // exists on any line within the error's range.
            let start = e.display_range().start.line_within_file();
            let end = e.display_range().end.line_within_file();
            for line_idx in start.to_zero_indexed()..=end.to_zero_indexed() {
                suppressed_errors
                    .entry(path)
                    .or_default()
                    .insert(LineNumber::from_zero_indexed(line_idx));
            }
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
            if let Some(above_line) = ignore.decrement() {
                ignore_locations.insert(above_line.to_zero_indexed() as usize);
            }
            let same_line = ignore.to_zero_indexed() as usize;
            ignore_locations.insert(same_line);
        }
        if let Ok(file) = read_and_validate_file(path) {
            let mut buf = String::with_capacity(file.len());
            let lines = file.lines();
            for (idx, line) in lines.enumerate() {
                if ignore_locations.contains(&idx) {
                    if regex.is_match(line) {
                        unused_ignore_count += 1;
                        let new_string = regex.replace_all(line, "");
                        if !new_string.trim().is_empty() {
                            buf.push_str(new_string.trim_end());
                            buf.push('\n');
                        }
                        // Skip writing newline if the line becomes empty after removing the ignore
                        continue;
                    }
                    buf.push_str(line);
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
        suppress::suppress_errors(errors.collect_errors().shown);
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
    return "hello" # pyrefly: ignore [bad-return]
"#;
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
}
