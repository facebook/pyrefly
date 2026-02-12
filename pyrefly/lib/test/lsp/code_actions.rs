/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use pretty_assertions::assert_eq;
use pyrefly_build::handle::Handle;
use pyrefly_python::module::Module;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;

use crate::module::module_info::ModuleInfo;
use crate::state::lsp::ImportFormat;
use crate::state::lsp::LocalRefactorCodeAction;
use crate::state::require::Require;
use crate::state::state::State;
use crate::test::util::extract_cursors_for_test;
use crate::test::util::get_batched_lsp_operations_report_allow_error;
use crate::test::util::mk_multi_file_state;
use crate::test::util::mk_multi_file_state_assert_no_errors;

fn apply_patch(info: &ModuleInfo, range: TextRange, patch: String) -> (String, String) {
    let before = info.contents().as_str().to_owned();
    let after = [
        &before[0..range.start().to_usize()],
        patch.as_str(),
        &before[range.end().to_usize()..],
    ]
    .join("");
    (before, after)
}

fn get_test_report(state: &State, handle: &Handle, position: TextSize) -> String {
    let mut report = "Code Actions Results:\n".to_owned();
    let transaction = state.transaction();
    for (title, info, range, patch) in transaction
        .local_quickfix_code_actions_sorted(
            handle,
            TextRange::new(position, position),
            ImportFormat::Absolute,
        )
        .unwrap_or_default()
    {
        let (before, after) = apply_patch(&info, range, patch);
        report.push_str("# Title: ");
        report.push_str(&title);
        report.push('\n');
        report.push_str("\n## Before:\n");
        report.push_str(&before);
        report.push_str("\n## After:\n");
        report.push_str(&after);
        report.push('\n');
    }
    report
}

fn apply_refactor_edits_for_module(
    module: &ModuleInfo,
    edits: &[(Module, TextRange, String)],
) -> String {
    let mut relevant_edits: Vec<(TextRange, String)> = edits
        .iter()
        .filter(|(edit_module, _, _)| edit_module.path() == module.path())
        .map(|(_, range, text)| (*range, text.clone()))
        .collect();
    relevant_edits.sort_by_key(|(range, _)| range.start());
    let mut result = module.contents().as_str().to_owned();
    for (range, replacement) in relevant_edits.into_iter().rev() {
        result.replace_range(
            range.start().to_usize()..range.end().to_usize(),
            &replacement,
        );
    }
    result
}

fn find_marked_range(source: &str) -> TextRange {
    find_marked_range_with(source, "# EXTRACT-START", "# EXTRACT-END")
}

fn find_marked_range_with(source: &str, start_marker: &str, end_marker: &str) -> TextRange {
    let start_idx = source
        .find(start_marker)
        .expect("missing start marker for extract refactor test");
    let start_line_end = source[start_idx..]
        .find('\n')
        .map(|offset| start_idx + offset + 1)
        .unwrap_or(source.len());
    let end_idx = source
        .find(end_marker)
        .expect("missing end marker for extract refactor test");
    let end_line_start = source[..end_idx]
        .rfind('\n')
        .map(|idx| idx + 1)
        .unwrap_or(end_idx);
    TextRange::new(
        TextSize::try_from(start_line_end).unwrap(),
        TextSize::try_from(end_line_start).unwrap(),
    )
}

/// Finds the text range for the Nth occurrence of `needle` in `source`.
///
/// This is used by tests that need to select a specific repeated token without
/// adding extra inline markers.
fn find_nth_range(source: &str, needle: &str, occurrence: usize) -> TextRange {
    assert!(occurrence > 0, "occurrence is 1-based");
    let mut start = 0;
    let mut seen = 0;
    while let Some(found) = source[start..].find(needle) {
        let abs = start + found;
        seen += 1;
        if seen == occurrence {
            let end = abs + needle.len();
            return TextRange::new(
                TextSize::try_from(abs).unwrap(),
                TextSize::try_from(end).unwrap(),
            );
        }
        start = abs + needle.len();
    }
    panic!(
        "could not find occurrence {} of '{}' in source",
        occurrence, needle
    );
}

fn compute_extract_actions(
    code: &str,
) -> (
    ModuleInfo,
    Vec<Vec<(Module, TextRange, String)>>,
    Vec<String>,
) {
    let (handles, state) =
        mk_multi_file_state_assert_no_errors(&[("main", code)], Require::Everything);
    let handle = handles.get("main").unwrap();
    let transaction = state.transaction();
    let module_info = transaction.get_module_info(handle).unwrap();
    let selection = find_marked_range(code);
    let actions = transaction
        .extract_function_code_actions(handle, selection)
        .unwrap_or_default();
    let edit_sets: Vec<Vec<(Module, TextRange, String)>> =
        actions.iter().map(|action| action.edits.clone()).collect();
    let titles = actions.iter().map(|action| action.title.clone()).collect();
    (module_info, edit_sets, titles)
}

fn apply_first_extract_action(code: &str) -> Option<String> {
    let (module_info, actions, _) = compute_extract_actions(code);
    let edits = actions.first()?;
    Some(apply_refactor_edits_for_module(&module_info, edits))
}

fn compute_extract_variable_actions(
    code: &str,
) -> (
    ModuleInfo,
    Vec<Vec<(Module, TextRange, String)>>,
    Vec<String>,
) {
    let (handles, state) =
        mk_multi_file_state_assert_no_errors(&[("main", code)], Require::Everything);
    let handle = handles.get("main").unwrap();
    let transaction = state.transaction();
    let module_info = transaction.get_module_info(handle).unwrap();
    let selection = find_marked_range(code);
    let actions = transaction
        .extract_variable_code_actions(handle, selection)
        .unwrap_or_default();
    let edit_sets: Vec<Vec<(Module, TextRange, String)>> =
        actions.iter().map(|action| action.edits.clone()).collect();
    let titles = actions.iter().map(|action| action.title.clone()).collect();
    (module_info, edit_sets, titles)
}

fn apply_first_extract_variable_action(code: &str) -> Option<String> {
    let (module_info, actions, _) = compute_extract_variable_actions(code);
    let edits = actions.first()?;
    Some(apply_refactor_edits_for_module(&module_info, edits))
}

fn compute_invert_boolean_actions(
    code: &str,
    selection: TextRange,
) -> (
    ModuleInfo,
    Vec<Vec<(Module, TextRange, String)>>,
    Vec<String>,
) {
    let (handles, state) =
        mk_multi_file_state_assert_no_errors(&[("main", code)], Require::Everything);
    let handle = handles.get("main").unwrap();
    let transaction = state.transaction();
    let module_info = transaction.get_module_info(handle).unwrap();
    let actions = transaction
        .invert_boolean_code_actions(handle, selection)
        .unwrap_or_default();
    let edit_sets: Vec<Vec<(Module, TextRange, String)>> =
        actions.iter().map(|action| action.edits.clone()).collect();
    let titles = actions.iter().map(|action| action.title.clone()).collect();
    (module_info, edit_sets, titles)
}

fn apply_first_invert_boolean_action(code: &str, selection: TextRange) -> Option<String> {
    let (module_info, actions, _) = compute_invert_boolean_actions(code, selection);
    let edits = actions.first()?;
    Some(apply_refactor_edits_for_module(&module_info, edits))
}

fn assert_no_invert_boolean_action(code: &str, selection: TextRange) {
    let (_, actions, _) = compute_invert_boolean_actions(code, selection);
    assert!(
        actions.is_empty(),
        "expected no invert-boolean actions, found {}",
        actions.len()
    );
}

fn compute_invert_boolean_actions_allow_errors(
    code: &str,
    selection: TextRange,
) -> (
    ModuleInfo,
    Vec<Vec<(Module, TextRange, String)>>,
    Vec<String>,
) {
    let (handles, state) = mk_multi_file_state(&[("main", code)], Require::Everything, false);
    let handle = handles.get("main").unwrap();
    let transaction = state.transaction();
    let module_info = transaction.get_module_info(handle).unwrap();
    let actions = transaction
        .invert_boolean_code_actions(handle, selection)
        .unwrap_or_default();
    let edit_sets: Vec<Vec<(Module, TextRange, String)>> =
        actions.iter().map(|action| action.edits.clone()).collect();
    let titles = actions.iter().map(|action| action.title.clone()).collect();
    (module_info, edit_sets, titles)
}

fn assert_no_invert_boolean_action_allow_errors(code: &str, selection: TextRange) {
    let (_, actions, _) = compute_invert_boolean_actions_allow_errors(code, selection);
    assert!(
        actions.is_empty(),
        "expected no invert-boolean actions, found {}",
        actions.len()
    );
}

fn cursor_selection(code: &str) -> TextRange {
    let position = extract_cursors_for_test(code)
        .first()
        .copied()
        .expect("expected cursor marker");
    TextRange::new(position, position)
}

fn apply_first_inline_variable_action(code: &str) -> Option<String> {
    let (handles, state) =
        mk_multi_file_state_assert_no_errors(&[("main", code)], Require::Everything);
    let handle = handles.get("main").unwrap();
    let transaction = state.transaction();
    let module_info = transaction.get_module_info(handle).unwrap();
    let selection = cursor_selection(code);
    let actions = transaction
        .inline_variable_code_actions(handle, selection)
        .unwrap_or_default();
    let edits = actions.first()?.edits.clone();
    Some(apply_refactor_edits_for_module(&module_info, &edits))
}

fn apply_first_inline_method_action(code: &str) -> Option<String> {
    let (handles, state) =
        mk_multi_file_state_assert_no_errors(&[("main", code)], Require::Everything);
    let handle = handles.get("main").unwrap();
    let transaction = state.transaction();
    let module_info = transaction.get_module_info(handle).unwrap();
    let selection = cursor_selection(code);
    let actions = transaction
        .inline_method_code_actions(handle, selection)
        .unwrap_or_default();
    let edits = actions.first()?.edits.clone();
    Some(apply_refactor_edits_for_module(&module_info, &edits))
}

fn apply_first_inline_method_action_allow_errors(code: &str) -> Option<String> {
    let (handles, state) = mk_multi_file_state(&[("main", code)], Require::Everything, false);
    let handle = handles.get("main").unwrap();
    let transaction = state.transaction();
    let module_info = transaction.get_module_info(handle).unwrap();
    let selection = cursor_selection(code);
    let actions = transaction
        .inline_method_code_actions(handle, selection)
        .unwrap_or_default();
    let edits = actions.first()?.edits.clone();
    Some(apply_refactor_edits_for_module(&module_info, &edits))
}

fn apply_first_inline_parameter_action(code: &str) -> Option<String> {
    let (handles, state) =
        mk_multi_file_state_assert_no_errors(&[("main", code)], Require::Everything);
    let handle = handles.get("main").unwrap();
    let transaction = state.transaction();
    let module_info = transaction.get_module_info(handle).unwrap();
    let selection = cursor_selection(code);
    let actions = transaction
        .inline_parameter_code_actions(handle, selection)
        .unwrap_or_default();
    let edits = actions.first()?.edits.clone();
    Some(apply_refactor_edits_for_module(&module_info, &edits))
}

fn compute_introduce_parameter_actions(
    code: &str,
) -> (
    ModuleInfo,
    Vec<Vec<(Module, TextRange, String)>>,
    Vec<String>,
) {
    let (handles, state) =
        mk_multi_file_state_assert_no_errors(&[("main", code)], Require::Everything);
    let handle = handles.get("main").unwrap();
    let transaction = state.transaction();
    let module_info = transaction.get_module_info(handle).unwrap();
    let selection = find_marked_range(module_info.contents());
    let actions = transaction
        .introduce_parameter_code_actions(handle, selection)
        .unwrap_or_default();
    let edit_sets: Vec<Vec<(Module, TextRange, String)>> =
        actions.iter().map(|action| action.edits.clone()).collect();
    let titles = actions.iter().map(|action| action.title.clone()).collect();
    (module_info, edit_sets, titles)
}

fn apply_introduce_parameter_action(code: &str, index: usize) -> Option<String> {
    let (module_info, actions, _) = compute_introduce_parameter_actions(code);
    let edits = actions.get(index)?;
    Some(apply_refactor_edits_for_module(&module_info, edits))
}

fn assert_no_introduce_parameter_action(code: &str) {
    let (_, actions, _) = compute_introduce_parameter_actions(code);
    assert!(
        actions.is_empty(),
        "expected no introduce-parameter actions, found {}",
        actions.len()
    );
}

fn assert_no_extract_variable_action(code: &str) {
    let (_, actions, _) = compute_extract_variable_actions(code);
    assert!(
        actions.is_empty(),
        "expected no extract-variable actions, found {}",
        actions.len()
    );
}

fn assert_no_extract_action(code: &str) {
    let (_, actions, _) = compute_extract_actions(code);
    assert!(
        actions.is_empty(),
        "expected no extract-function actions, found {}",
        actions.len()
    );
}

fn compute_move_actions(
    code: &str,
    selection: TextRange,
    compute: impl Fn(
        &crate::state::state::Transaction<'_>,
        &Handle,
        TextRange,
    ) -> Option<Vec<LocalRefactorCodeAction>>,
) -> (
    ModuleInfo,
    Vec<Vec<(Module, TextRange, String)>>,
    Vec<String>,
) {
    let (handles, state) =
        mk_multi_file_state_assert_no_errors(&[("main", code)], Require::Everything);
    let handle = handles.get("main").unwrap();
    let transaction = state.transaction();
    let module_info = transaction.get_module_info(handle).unwrap();
    let actions = compute(&transaction, handle, selection).unwrap_or_default();
    let edit_sets: Vec<Vec<(Module, TextRange, String)>> =
        actions.iter().map(|action| action.edits.clone()).collect();
    let titles = actions.iter().map(|action| action.title.clone()).collect();
    (module_info, edit_sets, titles)
}

fn compute_module_member_move_actions(
    code_by_module: &[(&'static str, &str)],
    module_name: &'static str,
    selection: TextRange,
) -> (
    ModuleInfo,
    Vec<Vec<(Module, TextRange, String)>>,
    Vec<String>,
    std::collections::HashMap<String, ModuleInfo>,
) {
    let (handles, state) =
        mk_multi_file_state_assert_no_errors(code_by_module, Require::Everything);
    let handle = handles.get(module_name).unwrap();
    let transaction = state.transaction();
    let module_info = transaction.get_module_info(handle).unwrap();
    let mut module_infos = std::collections::HashMap::new();
    for (name, _) in code_by_module {
        if let Some(handle) = handles.get(*name)
            && let Some(info) = transaction.get_module_info(handle)
        {
            module_infos.insert((*name).to_owned(), info);
        }
    }
    let actions = transaction
        .move_module_member_code_actions(handle, selection, ImportFormat::Absolute)
        .unwrap_or_default();
    let edit_sets: Vec<Vec<(Module, TextRange, String)>> =
        actions.iter().map(|action| action.edits.clone()).collect();
    let titles = actions.iter().map(|action| action.title.clone()).collect();
    (module_info, edit_sets, titles, module_infos)
}

fn compute_make_top_level_actions(
    code: &str,
    selection: TextRange,
) -> (
    ModuleInfo,
    Vec<Vec<(Module, TextRange, String)>>,
    Vec<String>,
) {
    let (handles, state) =
        mk_multi_file_state_assert_no_errors(&[("main", code)], Require::Everything);
    let handle = handles.get("main").unwrap();
    let transaction = state.transaction();
    let module_info = transaction.get_module_info(handle).unwrap();
    let actions = transaction
        .make_local_function_top_level_code_actions(handle, selection, ImportFormat::Absolute)
        .unwrap_or_default();
    let edit_sets: Vec<Vec<(Module, TextRange, String)>> =
        actions.iter().map(|action| action.edits.clone()).collect();
    let titles = actions.iter().map(|action| action.title.clone()).collect();
    (module_info, edit_sets, titles)
}

fn compute_convert_star_import_actions(
    code_by_module: &[(&'static str, &str)],
    module_name: &'static str,
    selection: TextRange,
) -> (
    ModuleInfo,
    Vec<Vec<(Module, TextRange, String)>>,
    Vec<String>,
) {
    let (handles, state) =
        mk_multi_file_state_assert_no_errors(code_by_module, Require::Everything);
    let handle = handles.get(module_name).unwrap();
    let transaction = state.transaction();
    let module_info = transaction.get_module_info(handle).unwrap();
    let actions = transaction
        .convert_star_import_code_actions(handle, selection)
        .unwrap_or_default();
    let edit_sets: Vec<Vec<(Module, TextRange, String)>> =
        actions.iter().map(|action| action.edits.clone()).collect();
    let titles = actions.iter().map(|action| action.title.clone()).collect();
    (module_info, edit_sets, titles)
}

fn compute_convert_import_actions(
    code_by_module: &[(&'static str, &str)],
    module_name: &'static str,
    selection: TextRange,
) -> (
    ModuleInfo,
    Vec<Vec<(Module, TextRange, String)>>,
    Vec<String>,
) {
    let (handles, state) =
        mk_multi_file_state_assert_no_errors(code_by_module, Require::Everything);
    let handle = handles.get(module_name).unwrap();
    let transaction = state.transaction();
    let module_info = transaction.get_module_info(handle).unwrap();
    let actions = transaction
        .convert_import_code_actions(handle, selection)
        .unwrap_or_default();
    let edit_sets: Vec<Vec<(Module, TextRange, String)>> =
        actions.iter().map(|action| action.edits.clone()).collect();
    let titles = actions.iter().map(|action| action.title.clone()).collect();
    (module_info, edit_sets, titles)
}

fn compute_pull_up_actions(
    code: &str,
) -> (
    ModuleInfo,
    Vec<Vec<(Module, TextRange, String)>>,
    Vec<String>,
) {
    let selection = find_marked_range_with(code, "# MOVE-START", "# MOVE-END");
    compute_move_actions(code, selection, |transaction, handle, selection| {
        transaction.pull_members_up_code_actions(handle, selection)
    })
}

fn compute_push_down_actions(
    code: &str,
) -> (
    ModuleInfo,
    Vec<Vec<(Module, TextRange, String)>>,
    Vec<String>,
) {
    let selection = find_marked_range_with(code, "# MOVE-START", "# MOVE-END");
    compute_move_actions(code, selection, |transaction, handle, selection| {
        transaction.push_members_down_code_actions(handle, selection)
    })
}

fn compute_extract_superclass_actions(
    code: &str,
) -> (
    ModuleInfo,
    Vec<Vec<(Module, TextRange, String)>>,
    Vec<String>,
) {
    let selection = find_marked_range_with(code, "# SUPER-START", "# SUPER-END");
    compute_move_actions(code, selection, |transaction, handle, selection| {
        transaction.extract_superclass_code_actions(handle, selection)
    })
}

#[test]
fn basic_test() {
    let report = get_batched_lsp_operations_report_allow_error(
        &[
            ("a", "my_export = 3\n"),
            ("b", "from .a import my_export\n"),
            ("c", "my_export\n# ^"),
            ("d", "my_export = 3\n"),
        ],
        get_test_report,
    );
    // We should suggest imports from both a and d, but not b.
    assert_eq!(
        r#"
# a.py

# b.py

# c.py
1 | my_export
      ^
Code Actions Results:
# Title: Insert import: `from a import my_export`

## Before:
my_export
# ^
## After:
from a import my_export
my_export
# ^
# Title: Insert import: `from d import my_export`

## Before:
my_export
# ^
## After:
from d import my_export
my_export
# ^



# d.py
"#
        .trim(),
        report.trim()
    );
}

#[test]
fn prefer_public_stdlib_module_for_reexports() {
    let report =
        get_batched_lsp_operations_report_allow_error(&[("main", "BytesIO\n# ^")], get_test_report);
    assert_eq!(
        r#"
# main.py
1 | BytesIO
      ^
Code Actions Results:
# Title: Insert import: `from io import BytesIO`

## Before:
BytesIO
# ^
## After:
from io import BytesIO
BytesIO
# ^
# Title: Insert import: `from _io import BytesIO`

## Before:
BytesIO
# ^
## After:
from _io import BytesIO
BytesIO
# ^
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn insertion_test_module_import() {
    let report = get_batched_lsp_operations_report_allow_error(
        &[("my_module", "my_export = 3\n"), ("b", "my_module\n# ^")],
        get_test_report,
    );
    assert_eq!(
        r#"
# my_module.py

# b.py
1 | my_module
      ^
Code Actions Results:
# Title: Insert import: `import my_module`

## Before:
my_module
# ^
## After:
import my_module
my_module
# ^
"#
        .trim(),
        report.trim()
    );
}

#[test]
fn insertion_test_comments() {
    let report = get_batched_lsp_operations_report_allow_error(
        &[
            ("a", "my_export = 3\n"),
            ("b", "# i am a comment\nmy_export\n# ^"),
        ],
        get_test_report,
    );
    // We will insert the import after a comment, which might not be the intended target of the
    // comment. This is not ideal, but we cannot do much better without sophisticated comment
    // attachments.
    assert_eq!(
        r#"
# a.py

# b.py
2 | my_export
      ^
Code Actions Results:
# Title: Insert import: `from a import my_export`

## Before:
# i am a comment
my_export
# ^
## After:
# i am a comment
from a import my_export
my_export
# ^
"#
        .trim(),
        report.trim()
    );
}

#[test]
fn insertion_test_existing_imports() {
    let report = get_batched_lsp_operations_report_allow_error(
        &[
            ("a", "my_export = 3\n"),
            ("b", "from typing import List\nmy_export\n# ^"),
        ],
        get_test_report,
    );
    // Insert before all imports. This might not adhere to existing import sorting code style.
    assert_eq!(
        r#"
# a.py

# b.py
2 | my_export
      ^
Code Actions Results:
# Title: Insert import: `from a import my_export`

## Before:
from typing import List
my_export
# ^
## After:
from a import my_export
from typing import List
my_export
# ^
"#
        .trim(),
        report.trim()
    );
}

#[test]
fn insertion_test_duplicate_imports() {
    let report = get_batched_lsp_operations_report_allow_error(
        &[
            ("a", "my_export = 3\nanother_thing = 4"),
            ("b", "from a import another_thing\nmy_export\n# ^"),
        ],
        get_test_report,
    );
    // The insertion won't attempt to merge imports from the same module.
    // It's not illegal, but it would be nice if we do merge.
    assert_eq!(
        r#"
# a.py

# b.py
2 | my_export
      ^
Code Actions Results:
# Title: Insert import: `from a import my_export`

## Before:
from a import another_thing
my_export
# ^
## After:
from a import my_export
from a import another_thing
my_export
# ^
"#
        .trim(),
        report.trim()
    );
}

#[test]
fn redundant_cast_quickfix() {
    let report = get_batched_lsp_operations_report_allow_error(
        &[(
            "main",
            "from typing import cast\nx: int = 0\nx = cast(int, x)\n#   ^",
        )],
        get_test_report,
    );
    assert_eq!(
        r#"
# main.py
3 | x = cast(int, x)
        ^
Code Actions Results:
# Title: Remove redundant cast

## Before:
from typing import cast
x: int = 0
x = cast(int, x)
#   ^
## After:
from typing import cast
x: int = 0
x = x
#   ^
"#
        .trim(),
        report.trim()
    );
}

#[test]
fn redundant_cast_fix_all() {
    let (handles, state) = mk_multi_file_state(
        &[(
            "main",
            "from typing import cast\nx: int = 0\nx = cast(int, x)\ny = cast(int, x)\n",
        )],
        Require::Exports,
        false,
    );
    let handle = handles.get("main").unwrap();
    let transaction = state.transaction();
    let module_info = transaction.get_module_info(handle).unwrap();
    let edits = transaction
        .redundant_cast_fix_all_edits(handle)
        .unwrap_or_default();
    let updated = apply_refactor_edits_for_module(&module_info, &edits);
    assert_eq!(
        "from typing import cast\nx: int = 0\nx = x\ny = x\n",
        updated
    );
}

fn redundant_cast_action_after(code: &str, cursor_offset: usize) -> Option<String> {
    let (handles, state) = mk_multi_file_state(&[("main", code)], Require::Exports, false);
    let handle = handles.get("main")?;
    let transaction = state.transaction();
    let module_info = transaction.get_module_info(handle)?;
    let position = TextSize::try_from(cursor_offset).ok()?;
    let actions = transaction
        .local_quickfix_code_actions_sorted(
            handle,
            TextRange::new(position, position),
            ImportFormat::Absolute,
        )
        .unwrap_or_default();
    let (_, module, range, patch) = actions
        .into_iter()
        .find(|(title, _, _, _)| title == "Remove redundant cast")?;
    if module.path() != module_info.path() {
        return None;
    }
    let (_before, after) = apply_patch(&module_info, range, patch);
    Some(after)
}

#[test]
fn redundant_cast_parenthesized_expr() {
    let code = "from typing import cast\na: int = 1\nb: int = 2\ncast(int, a + b)\n";
    let cursor_offset = code.find("cast(").unwrap();
    let after = redundant_cast_action_after(code, cursor_offset).unwrap();
    assert_eq!(
        "from typing import cast\na: int = 1\nb: int = 2\n(a + b)\n",
        after
    );
}

#[test]
fn redundant_cast_nested_call() {
    let code = "from typing import cast\na: int = 1\nb: int = 2\nprint(cast(int, a + b))\n";
    let cursor_offset = code.find("cast(").unwrap();
    let after = redundant_cast_action_after(code, cursor_offset).unwrap();
    assert_eq!(
        "from typing import cast\na: int = 1\nb: int = 2\nprint((a + b))\n",
        after
    );
}

#[test]
fn redundant_cast_cursor_inside_args() {
    let code = "from typing import cast\na: int = 1\nb: int = 2\ncast(int, a + b)\n";
    let cursor_offset = code.find("a + b").unwrap();
    let after = redundant_cast_action_after(code, cursor_offset).unwrap();
    assert_eq!(
        "from typing import cast\na: int = 1\nb: int = 2\n(a + b)\n",
        after
    );
}

#[test]
fn redundant_cast_preserves_multiplication_precedence() {
    let code =
        "from typing import cast\nx: int = 1\ny: int = 2\nz: int = 3\nx * cast(int, y + z)\n";
    let cursor_offset = code.find("cast(").unwrap();
    let after = redundant_cast_action_after(code, cursor_offset).unwrap();
    assert_eq!(
        "from typing import cast\nx: int = 1\ny: int = 2\nz: int = 3\nx * (y + z)\n",
        after
    );
}

#[test]
fn test_import_from_stdlib() {
    let report = get_batched_lsp_operations_report_allow_error(
        &[("a", "TypeVar('T')\n# ^")],
        get_test_report,
    );
    // TODO: Ideally `typing` would be preferred over `ast`.
    assert_eq!(
        r#"
# a.py
1 | TypeVar('T')
      ^
Code Actions Results:
# Title: Insert import: `from ast import TypeVar`

## Before:
TypeVar('T')
# ^
## After:
from ast import TypeVar
TypeVar('T')
# ^
# Title: Insert import: `from typing import TypeVar`

## Before:
TypeVar('T')
# ^
## After:
from typing import TypeVar
TypeVar('T')
# ^
"#
        .trim(),
        report.trim()
    );
}

#[test]
fn test_take_deprecation_into_account_in_sorting_of_actions() {
    let report = get_batched_lsp_operations_report_allow_error(
        &[
            (
                "a",
                "from warnings import deprecated\n@deprecated('')\ndef my_func(): pass",
            ),
            ("b", "def my_func(): pass"),
            ("c", "my_func()\n# ^"),
        ],
        get_test_report,
    );
    assert_eq!(
        r#"
# a.py

# b.py

# c.py
1 | my_func()
      ^
Code Actions Results:
# Title: Insert import: `from b import my_func`

## Before:
my_func()
# ^
## After:
from b import my_func
my_func()
# ^
# Title: Insert import: `from a import my_func` (deprecated)

## Before:
my_func()
# ^
## After:
from a import my_func
my_func()
# ^
"#
        .trim(),
        report.trim()
    );
}

#[test]
fn extract_function_basic_refactor() {
    let code = r#"
def process_data(data_list):
    total_sum = 0
    for item in data_list:
        # EXTRACT-START
        squared_value = item * item
        if squared_value > 100:
            print(f"Large value detected: {squared_value}")
        total_sum += squared_value
        # EXTRACT-END
    return total_sum


if __name__ == "__main__":
    data = [1, 5, 12, 8, 15]
    result = process_data(data)
    print(f"The final sum is: {result}")
"#;
    let updated = apply_first_extract_action(code).expect("expected extract refactor action");
    let expected = r#"
def extracted_function(item, total_sum):
    squared_value = item * item
    if squared_value > 100:
        print(f"Large value detected: {squared_value}")
    total_sum += squared_value
    return total_sum

def process_data(data_list):
    total_sum = 0
    for item in data_list:
        # EXTRACT-START
        total_sum = extracted_function(item, total_sum)
        # EXTRACT-END
    return total_sum


if __name__ == "__main__":
    data = [1, 5, 12, 8, 15]
    result = process_data(data)
    print(f"The final sum is: {result}")
"#;
    assert_eq!(expected.trim(), updated.trim());
}

#[test]
fn extract_function_method_scope_preserves_indent() {
    let code = r#"
class Processor:
    def consume(self, item):
        print(item)

    def process(self, data_list):
        for item in data_list:
            # EXTRACT-START
            squared_value = item * item
            if squared_value > 10:
                self.consume(squared_value)
            # EXTRACT-END
        return len(data_list)
"#;
    let updated = apply_first_extract_action(code).expect("expected extract refactor action");
    let expected = r#"
def extracted_function(item, self):
    squared_value = item * item
    if squared_value > 10:
        self.consume(squared_value)

class Processor:
    def consume(self, item):
        print(item)

    def process(self, data_list):
        for item in data_list:
            # EXTRACT-START
            extracted_function(item, self)
            # EXTRACT-END
        return len(data_list)
"#;
    assert_eq!(expected.trim(), updated.trim());
}

#[test]
fn extract_function_produces_method_action() {
    let code = r#"
class Processor:
    def consume(self, item):
        print(item)

    def process(self, data_list):
        for item in data_list:
            # EXTRACT-START
            squared_value = item * item
            if squared_value > 10:
                self.consume(squared_value)
            # EXTRACT-END
        return len(data_list)
"#;
    let (module_info, actions, titles) = compute_extract_actions(code);
    assert_eq!(
        2,
        actions.len(),
        "expected both helper and method extract actions"
    );
    assert!(
        titles
            .get(1)
            .is_some_and(|title| title.contains("method `extracted_method` on `Processor`")),
        "expected second action to target method scope"
    );
    let updated = apply_refactor_edits_for_module(&module_info, &actions[1]);
    let expected = r#"
class Processor:
    def consume(self, item):
        print(item)

    def extracted_method(self, item):
        squared_value = item * item
        if squared_value > 10:
            self.consume(squared_value)

    def process(self, data_list):
        for item in data_list:
            # EXTRACT-START
            self.extracted_method(item)
            # EXTRACT-END
        return len(data_list)
"#;
    assert_eq!(expected.trim(), updated.trim());
}

#[test]
fn extract_function_method_without_self_usage_still_adds_receiver() {
    let code = r#"
class Processor:
    def process(self, data_list):
        for item in data_list:
            # EXTRACT-START
            squared_value = item * item
            print(item)
            # EXTRACT-END
        return len(data_list)
"#;
    let (module_info, actions, _) = compute_extract_actions(code);
    assert_eq!(2, actions.len(), "expected helper and method actions");
    let updated = apply_refactor_edits_for_module(&module_info, &actions[1]);
    let expected = r#"
class Processor:
    def extracted_method(self, item):
        squared_value = item * item
        print(item)

    def process(self, data_list):
        for item in data_list:
            # EXTRACT-START
            self.extracted_method(item)
            # EXTRACT-END
        return len(data_list)
"#;
    assert_eq!(expected.trim(), updated.trim());
}

#[test]
fn extract_function_method_preserves_custom_receiver_name() {
    let code = r#"
class Processor:
    def consume(this, item):
        print(item)

    def process(this, data_list):
        for item in data_list:
            # EXTRACT-START
            squared_value = item * item
            this.consume(squared_value)
            # EXTRACT-END
        return len(data_list)
"#;
    let (module_info, actions, _) = compute_extract_actions(code);
    assert_eq!(2, actions.len(), "expected helper and method actions");
    let updated = apply_refactor_edits_for_module(&module_info, &actions[1]);
    let expected = r#"
class Processor:
    def consume(this, item):
        print(item)

    def extracted_method(this, item):
        squared_value = item * item
        this.consume(squared_value)

    def process(this, data_list):
        for item in data_list:
            # EXTRACT-START
            this.extracted_method(item)
            # EXTRACT-END
        return len(data_list)
"#;
    assert_eq!(expected.trim(), updated.trim());
}

#[test]
fn extract_function_nested_class_method_action() {
    let code = r#"
class Outer:
    class Inner:
        def consume(self, item):
            print(item)

        def process(self, data_list):
            for item in data_list:
                # EXTRACT-START
                squared_value = item * item
                self.consume(squared_value)
                # EXTRACT-END
            return len(data_list)
"#;
    let (module_info, actions, titles) = compute_extract_actions(code);
    assert_eq!(2, actions.len(), "expected helper and method actions");
    assert!(
        titles
            .get(1)
            .is_some_and(|title| title.contains("method `extracted_method` on `Inner`")),
        "expected method action scoped to Inner"
    );
    let updated = apply_refactor_edits_for_module(&module_info, &actions[1]);
    let expected = r#"
class Outer:
    class Inner:
        def consume(self, item):
            print(item)

        def extracted_method(self, item):
            squared_value = item * item
            self.consume(squared_value)

        def process(self, data_list):
            for item in data_list:
                # EXTRACT-START
                self.extracted_method(item)
                # EXTRACT-END
            return len(data_list)
"#;
    assert_eq!(expected.trim(), updated.trim());
}

#[test]
fn extract_variable_basic_refactor() {
    let code = r#"
def process(data):
    total = 0
    for item in data:
        total += (
            # EXTRACT-START
            item * item + 1
            # EXTRACT-END
        )
    return total
"#;
    let updated =
        apply_first_extract_variable_action(code).expect("expected extract variable action");
    let expected = r#"
def process(data):
    total = 0
    for item in data:
        extracted_value = item * item + 1
        total += (
            # EXTRACT-START
            extracted_value
            # EXTRACT-END
        )
    return total
"#;
    assert_eq!(expected.trim(), updated.trim());
}

#[test]
fn invert_boolean_basic_refactor() {
    let code = r#"
def foo():
    abc = True
    return abc
"#;
    let selection = find_nth_range(code, "abc", 2);
    let updated =
        apply_first_invert_boolean_action(code, selection).expect("expected invert-boolean action");
    let expected = r#"
def foo():
    abc = False
    return (not abc)
"#;
    assert_eq!(expected.trim(), updated.trim());
}

#[test]
fn invert_boolean_removes_not() {
    let code = r#"
def foo():
    abc = False
    if not abc:
        return 1
    return 0
"#;
    let selection = find_nth_range(code, "abc", 2);
    let updated =
        apply_first_invert_boolean_action(code, selection).expect("expected invert-boolean action");
    let expected = r#"
def foo():
    abc = True
    if abc:
        return 1
    return 0
"#;
    assert_eq!(expected.trim(), updated.trim());
}

#[test]
fn invert_boolean_rejects_multi_target_assignment() {
    let code = r#"
def foo():
    abc = other = True
    return abc
"#;
    let selection = find_nth_range(code, "abc", 2);
    assert_no_invert_boolean_action(code, selection);
}

#[test]
fn invert_boolean_annotated_assignment() {
    let code = r#"
def foo():
    abc: bool = True
    return abc
"#;
    let selection = find_nth_range(code, "abc", 2);
    let updated =
        apply_first_invert_boolean_action(code, selection).expect("expected invert-boolean action");
    let expected = r#"
def foo():
    abc: bool = False
    return (not abc)
"#;
    assert_eq!(expected.trim(), updated.trim());
}

#[test]
fn invert_boolean_rejects_deleted_variable() {
    let code = r#"
def foo():
    abc = True
    del abc
    return abc
"#;
    let selection = find_nth_range(code, "abc", 3);
    assert_no_invert_boolean_action_allow_errors(code, selection);
}

#[test]
fn invert_boolean_multiple_assignments() {
    let code = r#"
def foo():
    abc = True
    abc = False
    return abc
"#;
    let selection = find_nth_range(code, "abc", 3);
    let updated =
        apply_first_invert_boolean_action(code, selection).expect("expected invert-boolean action");
    let expected = r#"
def foo():
    abc = True
    abc = True
    return (not abc)
"#;
    assert_eq!(expected.trim(), updated.trim());
}

#[test]
fn invert_boolean_nested_expression_keeps_outer_not() {
    let code = r#"
def foo():
    abc = True
    other = True
    return not (abc and other)
"#;
    let selection = find_nth_range(code, "abc", 2);
    let updated =
        apply_first_invert_boolean_action(code, selection).expect("expected invert-boolean action");
    let expected = r#"
def foo():
    abc = False
    other = True
    return not ((not abc) and other)
"#;
    assert_eq!(expected.trim(), updated.trim());
}

#[test]
fn invert_boolean_inverts_unary_not_assignment_value() {
    let code = r#"
def foo(other_var):
    abc = not other_var
    return abc
"#;
    let selection = find_nth_range(code, "abc", 2);
    let updated =
        apply_first_invert_boolean_action(code, selection).expect("expected invert-boolean action");
    let expected = r#"
def foo(other_var):
    abc = other_var
    return (not abc)
"#;
    assert_eq!(expected.trim(), updated.trim());
}

#[test]
fn pull_member_up_basic() {
    let code = r#"
class Super:
    pass

class Sub(Super):
    # MOVE-START
    def foo(self):
        return 1
    # MOVE-END
"#;
    let (module_info, actions, titles) = compute_pull_up_actions(code);
    assert_eq!(vec!["Pull `foo` up to `Super`"], titles);
    let updated = apply_refactor_edits_for_module(&module_info, &actions[0]);
    let expected = r#"
class Super:
    def foo(self):
        return 1

class Sub(Super):
    # MOVE-START
    pass
    # MOVE-END
"#;
    assert_eq!(expected.trim(), updated.trim());
}

#[test]
fn push_member_down_basic() {
    let code = r#"
class Base:
    # MOVE-START
    def foo(self):
        pass
    # MOVE-END

class Child(Base):
    pass
"#;
    let (module_info, actions, titles) = compute_push_down_actions(code);
    assert_eq!(vec!["Push `foo` down to `Child`"], titles);
    let updated = apply_refactor_edits_for_module(&module_info, &actions[0]);
    let expected = r#"
class Base:
    # MOVE-START
    pass
    # MOVE-END

class Child(Base):
    def foo(self):
        pass
"#;
    assert_eq!(expected.trim(), updated.trim());
}

#[test]
fn extract_superclass_basic() {
    let code = r#"
class Foo:
    def a(self):
        return 1
    # SUPER-START
    def b(self):
        return 2
    def c(self):
        return 3
    # SUPER-END
    def d(self):
        return 4
"#;
    let (module_info, actions, titles) = compute_extract_superclass_actions(code);
    assert_eq!(vec!["Extract superclass `BaseFoo`"], titles);
    let updated = apply_refactor_edits_for_module(&module_info, &actions[0]);
    let expected = r#"
class BaseFoo:
    def b(self):
        return 2
    def c(self):
        return 3

class Foo(BaseFoo):
    def a(self):
        return 1
    # SUPER-START
    # SUPER-END
    def d(self):
        return 4
"#;
    assert_eq!(expected.trim(), updated.trim());
}

#[test]
fn extract_superclass_with_metaclass_inserts_pass() {
    let code = r#"
class Base:
    pass

class Meta(type):
    pass

class Foo(Base, metaclass=Meta):
    # SUPER-START
    def only(self):
        pass
    # SUPER-END
"#;
    let (module_info, actions, titles) = compute_extract_superclass_actions(code);
    assert_eq!(vec!["Extract superclass `BaseFoo`"], titles);
    let updated = apply_refactor_edits_for_module(&module_info, &actions[0]);
    let expected = r#"
class Base:
    pass

class Meta(type):
    pass

class BaseFoo:
    def only(self):
        pass

class Foo(Base, BaseFoo, metaclass=Meta):
    # SUPER-START
    pass
    # SUPER-END
"#;
    assert_eq!(expected.trim(), updated.trim());
}

#[test]
fn extract_superclass_with_decorator() {
    let code = r#"
class Foo:
    # SUPER-START
    @property
    def bar(self) -> int:
        return 1
    # SUPER-END
"#;
    let (module_info, actions, titles) = compute_extract_superclass_actions(code);
    assert_eq!(vec!["Extract superclass `BaseFoo`"], titles);
    let updated = apply_refactor_edits_for_module(&module_info, &actions[0]);
    let expected = r#"
class BaseFoo:
    @property
    def bar(self) -> int:
        return 1

class Foo(BaseFoo):
    # SUPER-START
    pass
    # SUPER-END
"#;
    assert_eq!(expected.trim(), updated.trim());
}

#[test]
fn extract_superclass_multiline_method() {
    let code = r#"
class Foo:
    # SUPER-START
    def complex(self):
        x = 1
        y = 2
        if x > 0:
            return x + y
        else:
            return y
    # SUPER-END
"#;
    let (module_info, actions, titles) = compute_extract_superclass_actions(code);
    assert_eq!(vec!["Extract superclass `BaseFoo`"], titles);
    let updated = apply_refactor_edits_for_module(&module_info, &actions[0]);
    let expected = r#"
class BaseFoo:
    def complex(self):
        x = 1
        y = 2
        if x > 0:
            return x + y
        else:
            return y

class Foo(BaseFoo):
    # SUPER-START
    pass
    # SUPER-END
"#;
    assert_eq!(expected.trim(), updated.trim());
}

#[test]
fn extract_superclass_docstring_only_no_action() {
    let code = r#"
class Foo:
    # SUPER-START
    """This class has only a docstring."""
    # SUPER-END
"#;
    let (_, actions, titles) = compute_extract_superclass_actions(code);
    // No action should be offered when there are no extractable members
    assert!(actions.is_empty());
    assert!(titles.is_empty());
}

#[test]
fn extract_superclass_preserves_docstring() {
    let code = r#"
class Foo:
    """Foo's docstring."""
    # SUPER-START
    def method(self):
        pass
    # SUPER-END
"#;
    let (module_info, actions, titles) = compute_extract_superclass_actions(code);
    assert_eq!(vec!["Extract superclass `BaseFoo`"], titles);
    let updated = apply_refactor_edits_for_module(&module_info, &actions[0]);
    let expected = r#"
class BaseFoo:
    def method(self):
        pass

class Foo(BaseFoo):
    """Foo's docstring."""
    # SUPER-START
    pass
    # SUPER-END
"#;
    assert_eq!(expected.trim(), updated.trim());
}

#[test]
fn push_member_down_all_subclasses() {
    let code = r#"
class Base:
    # MOVE-START
    def foo(self):
        pass
    # MOVE-END

class ChildA(Base):
    pass

class ChildB(Base):
    pass
"#;
    let (module_info, actions, titles) = compute_push_down_actions(code);
    assert_eq!(
        vec![
            "Push `foo` down to `ChildA`",
            "Push `foo` down to `ChildB`",
            "Push `foo` down to all subclasses",
        ],
        titles
    );
    let all_idx = titles
        .iter()
        .position(|title| title == "Push `foo` down to all subclasses")
        .expect("missing all-subclasses action");
    let updated = apply_refactor_edits_for_module(&module_info, &actions[all_idx]);
    let expected = r#"
class Base:
    # MOVE-START
    pass
    # MOVE-END

class ChildA(Base):
    def foo(self):
        pass

class ChildB(Base):
    def foo(self):
        pass
"#;
    assert_eq!(expected.trim(), updated.trim());
}

#[test]
fn pull_member_up_with_docstring_replaces_pass() {
    let code = r#"
class Super:
    """Super docstring."""
    pass

class Sub(Super):
    # MOVE-START
    def foo(self):
        return 1
    # MOVE-END
"#;
    let (module_info, actions, titles) = compute_pull_up_actions(code);
    assert_eq!(vec!["Pull `foo` up to `Super`"], titles);
    let updated = apply_refactor_edits_for_module(&module_info, &actions[0]);
    let expected = r#"
class Super:
    """Super docstring."""
    def foo(self):
        return 1

class Sub(Super):
    # MOVE-START
    pass
    # MOVE-END
"#;
    assert_eq!(expected.trim(), updated.trim());
}

#[test]
fn push_member_down_preserves_origin_docstring() {
    let code = r#"
class Base:
    """Base docstring."""
    # MOVE-START
    def foo(self):
        return 1
    # MOVE-END

class Child(Base):
    """Child docstring."""
    pass
"#;
    let (module_info, actions, titles) = compute_push_down_actions(code);
    assert_eq!(vec!["Push `foo` down to `Child`"], titles);
    let updated = apply_refactor_edits_for_module(&module_info, &actions[0]);
    let expected = r#"
class Base:
    """Base docstring."""
    # MOVE-START
    pass
    # MOVE-END

class Child(Base):
    """Child docstring."""
    def foo(self):
        return 1
"#;
    assert_eq!(expected.trim(), updated.trim());
}

#[test]
fn pull_member_up_nested_class() {
    let code = r#"
class Outer:
    class Base:
        pass

    class Child(Base):
        # MOVE-START
        def foo(self):
            return 1
        # MOVE-END
"#;
    let (module_info, actions, titles) = compute_pull_up_actions(code);
    assert_eq!(vec!["Pull `foo` up to `Base`"], titles);
    let updated = apply_refactor_edits_for_module(&module_info, &actions[0]);
    let expected = r#"
class Outer:
    class Base:
        def foo(self):
            return 1

    class Child(Base):
        # MOVE-START
        pass
        # MOVE-END
"#;
    assert_eq!(expected.trim(), updated.trim());
}

#[test]
fn push_member_down_nested_class() {
    let code = r#"
class Outer:
    class Base:
        # MOVE-START
        def foo(self):
            return 1
        # MOVE-END

    class Child(Base):
        pass
"#;
    let (module_info, actions, titles) = compute_push_down_actions(code);
    assert_eq!(vec!["Push `foo` down to `Child`"], titles);
    let updated = apply_refactor_edits_for_module(&module_info, &actions[0]);
    let expected = r#"
class Outer:
    class Base:
        # MOVE-START
        pass
        # MOVE-END

    class Child(Base):
        def foo(self):
            return 1
"#;
    assert_eq!(expected.trim(), updated.trim());
}

#[test]
fn pull_up_class_attribute() {
    let code = r#"
class Base:
    pass

class Child(Base):
    # MOVE-START
    FLAG = 1
    # MOVE-END
"#;
    let (module_info, actions, titles) = compute_pull_up_actions(code);
    assert_eq!(vec!["Pull `FLAG` up to `Base`"], titles);
    let updated = apply_refactor_edits_for_module(&module_info, &actions[0]);
    let expected = r#"
class Base:
    FLAG = 1

class Child(Base):
    # MOVE-START
    pass
    # MOVE-END
"#;
    assert_eq!(expected.trim(), updated.trim());
}

#[test]
fn push_down_class_attribute() {
    let code = r#"
class Base:
    # MOVE-START
    FLAG: int = 1
    # MOVE-END

class Child(Base):
    pass
"#;
    let (module_info, actions, titles) = compute_push_down_actions(code);
    assert_eq!(vec!["Push `FLAG` down to `Child`"], titles);
    let updated = apply_refactor_edits_for_module(&module_info, &actions[0]);
    let expected = r#"
class Base:
    # MOVE-START
    pass
    # MOVE-END

class Child(Base):
    FLAG: int = 1
"#;
    assert_eq!(expected.trim(), updated.trim());
}

#[test]
fn no_pull_up_when_member_exists_in_base() {
    let code = r#"
class Base:
    def foo(self):
        pass

class Child(Base):
    # MOVE-START
    def foo(self):
        pass
    # MOVE-END
"#;
    let (_module_info, _actions, titles) = compute_pull_up_actions(code);
    assert!(titles.is_empty(), "expected no pull-up actions");
}

#[test]
fn no_push_down_when_member_exists_in_subclass() {
    let code = r#"
class Base:
    # MOVE-START
    def foo(self):
        pass
    # MOVE-END

class Child(Base):
    def foo(self):
        pass
"#;
    let (_module_info, _actions, titles) = compute_push_down_actions(code);
    assert!(titles.is_empty(), "expected no push-down actions");
}

#[test]
fn move_module_member_to_sibling() {
    let code_a = r#"
# MOVE-START
def foo():
    return 1
# MOVE-END
"#;
    let code_b = "";
    let selection = find_marked_range_with(code_a, "# MOVE-START", "# MOVE-END");
    let (module_info, actions, titles, module_infos) =
        compute_module_member_move_actions(&[("a", code_a), ("b", code_b)], "a", selection);
    assert_eq!(vec!["Move `foo` to `b`"], titles);
    let updated_a = apply_refactor_edits_for_module(&module_info, &actions[0]);
    let updated_b = apply_refactor_edits_for_module(
        module_infos.get("b").expect("missing module b"),
        &actions[0],
    );
    let expected_a = r#"
# MOVE-START
from b import foo
# MOVE-END
"#;
    let expected_b = r#"
def foo():
    return 1
"#;
    assert_eq!(expected_a.trim(), updated_a.trim());
    assert_eq!(expected_b.trim(), updated_b.trim());
}

#[test]
fn make_local_function_top_level() {
    let code = r#"
def outer():
    # MOVE-START
    def inner(x):
        return x + 1
    # MOVE-END
    return inner(1)
"#;
    let selection = find_marked_range_with(code, "# MOVE-START", "# MOVE-END");
    let (module_info, actions, titles) = compute_make_top_level_actions(code, selection);
    assert_eq!(vec!["Make `inner` top-level"], titles);
    let updated = apply_refactor_edits_for_module(&module_info, &actions[0]);
    let expected = r#"
def outer():
    # MOVE-START
    # MOVE-END
    return inner(1)
def inner(x):
    return x + 1
"#;
    assert_eq!(expected.trim(), updated.trim());
}

#[test]
fn make_local_function_top_level_inserts_pass() {
    let code = r#"
def outer():
    # MOVE-START
    def inner():
        return 1
    # MOVE-END
"#;
    let selection = find_marked_range_with(code, "# MOVE-START", "# MOVE-END");
    let (module_info, actions, titles) = compute_make_top_level_actions(code, selection);
    assert_eq!(vec!["Make `inner` top-level"], titles);
    let updated = apply_refactor_edits_for_module(&module_info, &actions[0]);
    let expected = r#"
def outer():
    # MOVE-START
    pass
def inner():
    return 1
    # MOVE-END
"#;
    assert_eq!(expected.trim(), updated.trim());
}

#[test]
fn make_method_top_level_with_wrapper() {
    let code = r#"
class C:
    # MOVE-START
    def foo(self, x):
        return x + 1
    # MOVE-END
"#;
    let selection = find_marked_range_with(code, "# MOVE-START", "# MOVE-END");
    let (module_info, actions, titles) = compute_make_top_level_actions(code, selection);
    assert_eq!(vec!["Make `foo` top-level"], titles);
    let updated = apply_refactor_edits_for_module(&module_info, &actions[0]);
    let expected = r#"
def foo(self, x):
    return x + 1
class C:
    # MOVE-START
    foo = foo
    # MOVE-END
"#;
    assert_eq!(expected.trim(), updated.trim());
}

#[test]
fn make_staticmethod_top_level_with_wrapper() {
    let code = r#"
class C:
    # MOVE-START
    @staticmethod
    def bar(x):
        return x
    # MOVE-END
"#;
    let selection = find_marked_range_with(code, "# MOVE-START", "# MOVE-END");
    let (module_info, actions, titles) = compute_make_top_level_actions(code, selection);
    assert_eq!(vec!["Make `bar` top-level"], titles);
    let updated = apply_refactor_edits_for_module(&module_info, &actions[0]);
    let expected = r#"
def bar(x):
    return x
class C:
    # MOVE-START
    bar = staticmethod(bar)
    # MOVE-END
"#;
    assert_eq!(expected.trim(), updated.trim());
}

#[test]
fn make_classmethod_top_level_with_wrapper() {
    let code = r#"
class C:
    # MOVE-START
    @classmethod
    def baz(cls, x):
        return x
    # MOVE-END
"#;
    let selection = find_marked_range_with(code, "# MOVE-START", "# MOVE-END");
    let (module_info, actions, titles) = compute_make_top_level_actions(code, selection);
    assert_eq!(vec!["Make `baz` top-level"], titles);
    let updated = apply_refactor_edits_for_module(&module_info, &actions[0]);
    let expected = r#"
def baz(cls, x):
    return x
class C:
    # MOVE-START
    baz = classmethod(baz)
    # MOVE-END
"#;
    assert_eq!(expected.trim(), updated.trim());
}

#[test]
fn convert_star_import_basic() {
    let code_main = r#"
# CONVERT-START
from foo import *  # noqa: F401
# CONVERT-END
a = A
b = B
"#;
    let code_foo = r#"
A = 1
B = 2
C = 3
"#;
    let selection = find_marked_range_with(code_main, "# CONVERT-START", "# CONVERT-END");
    let (module_info, actions, titles) = compute_convert_star_import_actions(
        &[("main", code_main), ("foo", code_foo)],
        "main",
        selection,
    );
    assert_eq!(vec!["Convert to explicit imports from `foo`"], titles);
    let updated = apply_refactor_edits_for_module(&module_info, &actions[0]);
    let expected = r#"
# CONVERT-START
from foo import A, B # noqa: F401
# CONVERT-END
a = A
b = B
"#;
    assert_eq!(expected.trim(), updated.trim());
}

#[test]
fn convert_star_import_relative() {
    let code_main = r#"
# CONVERT-START
from .foo import *
# CONVERT-END
x = A
"#;
    let code_foo = r#"
A = 1
"#;
    let selection = find_marked_range_with(code_main, "# CONVERT-START", "# CONVERT-END");
    let (module_info, actions, titles) = compute_convert_star_import_actions(
        &[("pkg.main", code_main), ("pkg.foo", code_foo)],
        "pkg.main",
        selection,
    );
    assert_eq!(vec!["Convert to explicit imports from `pkg.foo`"], titles);
    let updated = apply_refactor_edits_for_module(&module_info, &actions[0]);
    let expected = r#"
# CONVERT-START
from .foo import A
# CONVERT-END
x = A
"#;
    assert_eq!(expected.trim(), updated.trim());
}

#[test]
fn convert_star_import_selects_correct_import() {
    let code_main = r#"
# CONVERT-START
from foo import *
# CONVERT-END
from bar import *
a = A
b = B
"#;
    let code_foo = r#"
A = 1
"#;
    let code_bar = r#"
B = 2
"#;
    let selection = find_marked_range_with(code_main, "# CONVERT-START", "# CONVERT-END");
    let (module_info, actions, titles) = compute_convert_star_import_actions(
        &[("main", code_main), ("foo", code_foo), ("bar", code_bar)],
        "main",
        selection,
    );
    assert_eq!(vec!["Convert to explicit imports from `foo`"], titles);
    let updated = apply_refactor_edits_for_module(&module_info, &actions[0]);
    let expected = r#"
# CONVERT-START
from foo import A
# CONVERT-END
from bar import *
a = A
b = B
"#;
    assert_eq!(expected.trim(), updated.trim());
}

#[test]
fn convert_star_import_no_action_when_unused() {
    let code_main = r#"
# CONVERT-START
from foo import *
# CONVERT-END
x = 1
"#;
    let code_foo = r#"
A = 1
"#;
    let selection = find_marked_range_with(code_main, "# CONVERT-START", "# CONVERT-END");
    let (_module_info, actions, titles) = compute_convert_star_import_actions(
        &[("main", code_main), ("foo", code_foo)],
        "main",
        selection,
    );
    assert!(actions.is_empty());
    assert!(titles.is_empty());
}

#[test]
fn convert_star_import_multiline() {
    // Multi-line star imports with parentheses should be handled correctly.
    let code_main = r#"
# MULTILINE-START
from foo import (
    *,
)
# MULTILINE-END
x = A
"#;
    let code_foo = r#"
A = 1
"#;
    let selection = find_marked_range_with(code_main, "# MULTILINE-START", "# MULTILINE-END");
    let (module_info, actions, titles) = compute_convert_star_import_actions(
        &[("main", code_main), ("foo", code_foo)],
        "main",
        selection,
    );
    assert_eq!(vec!["Convert to explicit imports from `foo`"], titles);
    let updated = apply_refactor_edits_for_module(&module_info, &actions[0]);
    // The replacement should produce a valid single-line import.
    let expected = r#"
# MULTILINE-START
from foo import A
# MULTILINE-END
x = A
"#;
    assert_eq!(expected.trim(), updated.trim());
}

#[test]
fn convert_star_import_shadowed_name() {
    // When a name from the star import is shadowed by a local assignment,
    // it should not appear in the explicit import list.
    let code_main = r#"
# CONVERT-START
from foo import *
# CONVERT-END
A = 42
print(A)
print(B)
"#;
    let code_foo = r#"
A = 1
B = 2
"#;
    let selection = find_marked_range_with(code_main, "# CONVERT-START", "# CONVERT-END");
    let (module_info, actions, titles) = compute_convert_star_import_actions(
        &[("main", code_main), ("foo", code_foo)],
        "main",
        selection,
    );
    assert_eq!(vec!["Convert to explicit imports from `foo`"], titles);
    let updated = apply_refactor_edits_for_module(&module_info, &actions[0]);
    // Only B should be imported since A is shadowed by a local assignment.
    let expected = r#"
# CONVERT-START
from foo import B
# CONVERT-END
A = 42
print(A)
print(B)
"#;
    assert_eq!(expected.trim(), updated.trim());
}

#[test]
fn extract_variable_name_increments_when_taken() {
    let code = r#"
def compute():
    extracted_value = 10
    result = (
        # EXTRACT-START
        4 * 5
        # EXTRACT-END
    )
    return result
"#;
    let updated =
        apply_first_extract_variable_action(code).expect("expected extract variable action");
    let expected = r#"
def compute():
    extracted_value = 10
    extracted_value_1 = 4 * 5
    result = (
        # EXTRACT-START
        extracted_value_1
        # EXTRACT-END
    )
    return result
"#;
    assert_eq!(expected.trim(), updated.trim());
}

#[test]
fn extract_variable_rejects_empty_selection() {
    let code = r#"
def sink(values):
    # EXTRACT-START
    # EXTRACT-END
    return values
"#;
    assert_no_extract_variable_action(code);
}

#[test]
fn extract_variable_rejects_whitespace_selection() {
    let code = r#"
def sink(values):
    return (
        # EXTRACT-START

        # EXTRACT-END
    )
"#;
    assert_no_extract_variable_action(code);
}

#[test]
fn extract_variable_requires_exact_expression() {
    let code = r#"
def sink(values):
    # EXTRACT-START
    value = values[0]
    # EXTRACT-END
    return value
"#;
    assert_no_extract_variable_action(code);
}

#[test]
fn introduce_parameter_basic_refactor() {
    let code = r#"
def greet(name):
    return (
        # EXTRACT-START
        "Hello " + name
        # EXTRACT-END
    )

def caller():
    greet("Ada")
"#;
    let updated =
        apply_introduce_parameter_action(code, 0).expect("expected introduce-parameter action");
    let expected = r#"
def greet(name, param):
    return (
        # EXTRACT-START
        param
        # EXTRACT-END
    )

def caller():
    greet("Ada", "Hello " + ("Ada"))
"#;
    assert_eq!(expected.trim(), updated.trim());
}

#[test]
fn introduce_parameter_replace_all_occurrences() {
    let code = r#"
def add_one(x):
    value = (
        # EXTRACT-START
        x + 1
        # EXTRACT-END
    )
    return x + 1

def caller():
    add_one(3)
"#;
    let updated = apply_introduce_parameter_action(code, 1)
        .expect("expected introduce-parameter replace-all action");
    let expected = r#"
def add_one(x, param):
    value = (
        # EXTRACT-START
        param
        # EXTRACT-END
    )
    return param

def caller():
    add_one(3, (3) + 1)
"#;
    assert_eq!(expected.trim(), updated.trim());
}

#[test]
fn introduce_parameter_method_callsite_uses_receiver() {
    let code = r#"
class Greeter:
    def __init__(self):
        self.prefix = "Hi "

    def greet(self, name):
        return (
            # EXTRACT-START
            self.prefix + name
            # EXTRACT-END
        )

def caller():
    greeter = Greeter()
    greeter.greet("Ada")
"#;
    let updated =
        apply_introduce_parameter_action(code, 0).expect("expected introduce-parameter action");
    let expected = r#"
class Greeter:
    def __init__(self):
        self.prefix = "Hi "

    def greet(self, name, param):
        return (
            # EXTRACT-START
            param
            # EXTRACT-END
        )

def caller():
    greeter = Greeter()
    greeter.greet("Ada", greeter.prefix + ("Ada"))
"#;
    assert_eq!(expected.trim(), updated.trim());
}

#[test]
fn introduce_parameter_keyword_only_insertion() {
    let code = r#"
def mix(x, *, y):
    return (
        # EXTRACT-START
        x + y
        # EXTRACT-END
    )

def caller():
    mix(1, y=2)
"#;
    let updated =
        apply_introduce_parameter_action(code, 0).expect("expected introduce-parameter action");
    let expected = r#"
def mix(x, *, param, y):
    return (
        # EXTRACT-START
        param
        # EXTRACT-END
    )

def caller():
    mix(1, param=(1) + (2), y=2)
"#;
    assert_eq!(expected.trim(), updated.trim());
}

#[test]
fn introduce_parameter_no_args_callsite() {
    let code = r#"
def magic():
    return (
        # EXTRACT-START
        1 + 2
        # EXTRACT-END
    )

def caller():
    magic()
"#;
    let updated =
        apply_introduce_parameter_action(code, 0).expect("expected introduce-parameter action");
    let expected = r#"
def magic(param):
    return (
        # EXTRACT-START
        param
        # EXTRACT-END
    )

def caller():
    magic(1 + 2)
"#;
    assert_eq!(expected.trim(), updated.trim());
}

#[test]
fn introduce_parameter_staticmethod_callsite() {
    let code = r#"
class Utils:
    @staticmethod
    def join(a, b):
        return (
            # EXTRACT-START
            a + b
            # EXTRACT-END
        )

def caller():
    Utils.join("Hi ", "Ada")
"#;
    let updated =
        apply_introduce_parameter_action(code, 0).expect("expected introduce-parameter action");
    let expected = r#"
class Utils:
    @staticmethod
    def join(a, b, param):
        return (
            # EXTRACT-START
            param
            # EXTRACT-END
        )

def caller():
    Utils.join("Hi ", "Ada", ("Hi ") + ("Ada"))
"#;
    assert_eq!(expected.trim(), updated.trim());
}

#[test]
fn introduce_parameter_mixed_args_callsite() {
    let code = r#"
def add(a, b):
    return (
        # EXTRACT-START
        a + b
        # EXTRACT-END
    )

def caller():
    add(1, b=2)
"#;
    let updated =
        apply_introduce_parameter_action(code, 0).expect("expected introduce-parameter action");
    let expected = r#"
def add(a, b, param):
    return (
        # EXTRACT-START
        param
        # EXTRACT-END
    )

def caller():
    add(1, param=(1) + (2), b=2)
"#;
    assert_eq!(expected.trim(), updated.trim());
}

#[test]
fn introduce_parameter_classmethod_callsite_uses_cls() {
    let code = r#"
class Greeter:
    prefix = "Hi "

    @classmethod
    def greet(cls, name):
        return (
            # EXTRACT-START
            cls.prefix + name
            # EXTRACT-END
        )

def caller():
    Greeter.greet("Ada")
"#;
    let updated =
        apply_introduce_parameter_action(code, 0).expect("expected introduce-parameter action");
    let expected = r#"
class Greeter:
    prefix = "Hi "

    @classmethod
    def greet(cls, name, param):
        return (
            # EXTRACT-START
            param
            # EXTRACT-END
        )

def caller():
    Greeter.greet("Ada", Greeter.prefix + ("Ada"))
"#;
    assert_eq!(expected.trim(), updated.trim());
}

#[test]
fn introduce_parameter_rejects_local_names() {
    let code = r#"
def combine(values):
    offset = 2
    return (
        # EXTRACT-START
        values[0] + offset
        # EXTRACT-END
    )
"#;
    assert_no_introduce_parameter_action(code);
}

#[test]
fn introduce_parameter_rejects_star_args_callsite() {
    let code = r#"
def accept(value):
    return (
        # EXTRACT-START
        value + 1
        # EXTRACT-END
    )

def caller():
    values = [1]
    accept(*values)
"#;
    assert_no_introduce_parameter_action(code);
}

#[test]
fn introduce_parameter_rejects_kwargs_callsite() {
    let code = r#"
def accept(value):
    return (
        # EXTRACT-START
        value + 1
        # EXTRACT-END
    )

def caller():
    values = {"value": 1}
    accept(**values)
"#;
    assert_no_introduce_parameter_action(code);
}

mod extract_field_tests {
    use pretty_assertions::assert_eq;

    use super::Module;
    use super::ModuleInfo;
    use super::Require;
    use super::TextRange;
    use super::apply_refactor_edits_for_module;
    use super::find_marked_range;
    use super::mk_multi_file_state_assert_no_errors;

    fn compute_extract_field_actions(
        code: &str,
    ) -> (
        ModuleInfo,
        Vec<Vec<(Module, TextRange, String)>>,
        Vec<String>,
    ) {
        let (handles, state) =
            mk_multi_file_state_assert_no_errors(&[("main", code)], Require::Everything);
        let handle = handles.get("main").unwrap();
        let transaction = state.transaction();
        let module_info = transaction.get_module_info(handle).unwrap();
        let selection = find_marked_range(module_info.contents());
        let actions = transaction
            .extract_field_code_actions(handle, selection)
            .unwrap_or_default();
        let edit_sets: Vec<Vec<(Module, TextRange, String)>> =
            actions.iter().map(|action| action.edits.clone()).collect();
        let titles = actions.iter().map(|action| action.title.clone()).collect();
        (module_info, edit_sets, titles)
    }

    fn apply_first_extract_field_action(code: &str) -> Option<String> {
        let (module_info, actions, _) = compute_extract_field_actions(code);
        let edits = actions.first()?;
        Some(apply_refactor_edits_for_module(&module_info, edits))
    }

    fn assert_no_extract_field_action(code: &str) {
        let (_, actions, _) = compute_extract_field_actions(code);
        assert!(
            actions.is_empty(),
            "expected no extract-field actions, found {}",
            actions.len()
        );
    }

    #[test]
    fn extract_field_basic_instance_method() {
        let code = r#"
GLOBAL_FACTOR = 3

class Processor:
    """Handles work"""
    def process(self):
        return (
            # EXTRACT-START
            GLOBAL_FACTOR + 2
            # EXTRACT-END
        )
"#;
        let updated =
            apply_first_extract_field_action(code).expect("expected extract field action");
        let expected = r#"
GLOBAL_FACTOR = 3

class Processor:
    """Handles work"""
    extracted_field = GLOBAL_FACTOR + 2
    def process(self):
        return (
            # EXTRACT-START
            self.extracted_field
            # EXTRACT-END
        )
"#;
        assert_eq!(expected.trim(), updated.trim());
    }

    #[test]
    fn extract_field_rejects_method_local_dependencies() {
        let code = r#"
class Collector:
    def process(self, values):
        interim = len(values)
        return (
            # EXTRACT-START
            interim + 1
            # EXTRACT-END
        )
"#;
        assert_no_extract_field_action(code);
    }

    #[test]
    fn extract_field_classmethod_uses_cls_receiver() {
        let code = r#"
GLOBAL = 7

class Builder:
    @classmethod
    def make(cls):
        return (
            # EXTRACT-START
            GLOBAL * 2
            # EXTRACT-END
        )
"#;
        let updated =
            apply_first_extract_field_action(code).expect("expected extract field action");
        let expected = r#"
GLOBAL = 7

class Builder:
    extracted_field = GLOBAL * 2
    @classmethod
    def make(cls):
        return (
            # EXTRACT-START
            cls.extracted_field
            # EXTRACT-END
        )
"#;
        assert_eq!(expected.trim(), updated.trim());
    }

    #[test]
    fn extract_field_nested_class_inserts_into_inner_class() {
        let code = r#"
GLOBAL = 1

class Outer:
    class Inner:
        def compute(self):
            return (
                # EXTRACT-START
                GLOBAL + 2
                # EXTRACT-END
            )
"#;
        let updated =
            apply_first_extract_field_action(code).expect("expected extract field action");
        let expected = r#"
GLOBAL = 1

class Outer:
    class Inner:
        extracted_field = GLOBAL + 2
        def compute(self):
            return (
                # EXTRACT-START
                self.extracted_field
                # EXTRACT-END
            )
"#;
        assert_eq!(expected.trim(), updated.trim());
    }
}

#[test]
fn extract_function_staticmethod_falls_back_to_helper() {
    let code = r#"
class Processor:
    @staticmethod
    def process(item):
        # EXTRACT-START
        squared_value = item * item
        print(squared_value)
        # EXTRACT-END
        return squared_value
"#;
    let (module_info, actions, titles) = compute_extract_actions(code);
    assert_eq!(
        1,
        actions.len(),
        "expected only module-scope helper extract action"
    );
    assert!(
        titles
            .first()
            .is_some_and(|title| title.contains("Extract into helper `")),
        "expected helper extraction title, got {:?}",
        titles
    );
    let updated = apply_refactor_edits_for_module(&module_info, &actions[0]);
    let expected = r#"
def extracted_function(item):
    squared_value = item * item
    print(squared_value)
    return squared_value

class Processor:
    @staticmethod
    def process(item):
        # EXTRACT-START
        squared_value = extracted_function(item)
        # EXTRACT-END
        return squared_value
"#;
    assert_eq!(expected.trim(), updated.trim());
}

#[test]
fn extract_function_classmethod_falls_back_to_helper() {
    let code = r#"
class Processor:
    @classmethod
    def process(cls, item):
        # EXTRACT-START
        squared_value = item * item
        print(squared_value)
        # EXTRACT-END
        return squared_value
"#;
    let (module_info, actions, titles) = compute_extract_actions(code);
    assert_eq!(
        1,
        actions.len(),
        "expected only module-scope helper extract action"
    );
    assert!(
        titles
            .first()
            .is_some_and(|title| title.contains("Extract into helper `")),
        "expected helper extraction title, got {:?}",
        titles
    );
    let updated = apply_refactor_edits_for_module(&module_info, &actions[0]);
    let expected = r#"
def extracted_function(item):
    squared_value = item * item
    print(squared_value)
    return squared_value

class Processor:
    @classmethod
    def process(cls, item):
        # EXTRACT-START
        squared_value = extracted_function(item)
        # EXTRACT-END
        return squared_value
"#;
    assert_eq!(expected.trim(), updated.trim());
}

#[test]
fn extract_function_rejects_empty_selection() {
    let code = r#"
def sink(values):
    for value in values:
        # EXTRACT-START
        # EXTRACT-END
        print(value)
"#;
    assert!(
        apply_first_extract_action(code).is_none(),
        "expected no refactor action for empty selection"
    );
}

#[test]
fn extract_function_rejects_return_statement() {
    let code = r#"
def sink(values):
    # EXTRACT-START
    return values[0]
    # EXTRACT-END
"#;
    assert_no_extract_action(code);
}

#[test]
fn inline_variable_basic_refactor() {
    let code = r#"
def compute():
    value = 1 + 2
    result = value * 3
#            ^
    return result
"#;
    let updated =
        apply_first_inline_variable_action(code).expect("expected inline variable action");
    let expected = r#"
def compute():
    result = (1 + 2) * 3
#            ^
    return result
"#;
    assert_eq!(expected, updated);
}

#[test]
fn inline_method_basic_refactor() {
    let code = r#"
def add(a, b):
    return a + b

def compute():
    total = add(1, 2)
#           ^
    return total
"#;
    let updated = apply_first_inline_method_action(code).expect("expected inline method action");
    let expected = r#"
def add(a, b):
    return a + b

def compute():
    total = (1 + 2)
#           ^
    return total
"#;
    assert_eq!(expected, updated);
}

#[test]
fn inline_method_preserves_needed_parens() {
    // When an argument is a complex expression, it should be wrapped in parens
    let code = r#"
def mul(a, b):
    return a * b

def compute():
    result = mul(1 + 2, 3)
#            ^
    return result
"#;
    let updated = apply_first_inline_method_action(code).expect("expected inline method action");
    let expected = r#"
def mul(a, b):
    return a * b

def compute():
    result = ((1 + 2) * 3)
#            ^
    return result
"#;
    assert_eq!(expected, updated);
}

#[test]
fn inline_method_for_class() {
    let code = r#"
class A:
    def foo(self):
        return 1

    def bar(self):
        self.foo()
#        ^
"#;
    let updated = apply_first_inline_method_action(code).expect("expected inline method action");
    let expected = r#"
class A:
    def foo(self):
        return 1

    def bar(self):
        1
#        ^
"#;
    assert_eq!(expected, updated);
}

#[test]
fn inline_method_for_class_no_action_staticmethod() {
    // @staticmethod methods cannot use self.foo() pattern
    let code = r#"
class A:
    @staticmethod
    def foo():
        return 1

    def bar(self):
        self.foo()
#        ^
"#;
    assert!(apply_first_inline_method_action(code).is_none());
}

#[test]
fn inline_method_for_class_no_action_classmethod() {
    // @classmethod methods cannot use self.foo() pattern
    let code = r#"
class A:
    @classmethod
    def foo(cls):
        return 1

    def bar(self):
        self.foo()
#        ^
"#;
    assert!(apply_first_inline_method_action(code).is_none());
}

#[test]
fn inline_method_for_class_no_action_method_not_found() {
    // Cannot inline a method that doesn't exist in the class
    let code = r#"
class A:
    def bar(self):
        self.foo()
#        ^
"#;
    assert!(apply_first_inline_method_action_allow_errors(code).is_none());
}

#[test]
fn inline_method_for_class_no_action_different_receiver() {
    // Cannot inline when receiver name doesn't match the self parameter
    let code = r#"
class A:
    def foo(self):
        return 1

    def bar(self):
        this.foo()
#        ^
"#;
    assert!(apply_first_inline_method_action_allow_errors(code).is_none());
}

#[test]
fn inline_method_for_nested_class() {
    let code = r#"
class Outer:
    class Inner:
        def foo(self):
            return 1

        def bar(self):
            self.foo()
#            ^
"#;
    let updated = apply_first_inline_method_action(code).expect("expected inline method action");
    let expected = r#"
class Outer:
    class Inner:
        def foo(self):
            return 1

        def bar(self):
            1
#            ^
"#;
    assert_eq!(expected, updated);
}

#[test]
fn inline_parameter_basic_refactor() {
    let code = r#"
def add(a, b):
#          ^
    return a + b

def compute():
    return add(1, 2)
"#;
    let updated =
        apply_first_inline_parameter_action(code).expect("expected inline parameter action");
    let expected = r#"
def add(a):
#          ^
    return a + (2)

def compute():
    return add(1)
"#;
    assert_eq!(expected, updated);
}

#[test]
fn convert_from_import_to_relative() {
    let code_a = r#"
# IMPORT-START
from pkg.b import Foo
# IMPORT-END
"#;
    let code_b = "class Foo: ...\n";
    let selection = find_marked_range_with(code_a, "# IMPORT-START", "# IMPORT-END");
    let (module_info, actions, titles) =
        compute_convert_import_actions(&[("pkg.a", code_a), ("pkg.b", code_b)], "pkg.a", selection);
    assert!(
        titles
            .iter()
            .any(|title| title == "Convert import to relative path"),
        "expected relative import code action"
    );
    assert!(
        titles
            .iter()
            .any(|title| title == "Convert all imports to relative path"),
        "expected convert-all relative code action"
    );
    let idx = titles
        .iter()
        .position(|title| title == "Convert import to relative path")
        .unwrap();
    let updated = apply_refactor_edits_for_module(&module_info, &actions[idx]);
    let expected = r#"
# IMPORT-START
from .b import Foo
# IMPORT-END
"#;
    assert_eq!(expected.trim(), updated.trim());
}

#[test]
fn convert_from_import_to_absolute() {
    let code_a = r#"
# IMPORT-START
from .b import Foo
# IMPORT-END
"#;
    let code_b = "class Foo: ...\n";
    let selection = find_marked_range_with(code_a, "# IMPORT-START", "# IMPORT-END");
    let (module_info, actions, titles) =
        compute_convert_import_actions(&[("pkg.a", code_a), ("pkg.b", code_b)], "pkg.a", selection);
    assert!(
        titles
            .iter()
            .any(|title| title == "Convert import to absolute path"),
        "expected absolute import code action"
    );
    assert!(
        titles
            .iter()
            .any(|title| title == "Convert all imports to absolute path"),
        "expected convert-all absolute code action"
    );
    let idx = titles
        .iter()
        .position(|title| title == "Convert import to absolute path")
        .unwrap();
    let updated = apply_refactor_edits_for_module(&module_info, &actions[idx]);
    let expected = r#"
# IMPORT-START
from pkg.b import Foo
# IMPORT-END
"#;
    assert_eq!(expected.trim(), updated.trim());
}

#[test]
fn convert_import_statement_to_relative() {
    let code_a = r#"
# IMPORT-START
import pkg.b as mod_b
# IMPORT-END
"#;
    let code_b = "class Foo: ...\n";
    let selection = find_marked_range_with(code_a, "# IMPORT-START", "# IMPORT-END");
    let (module_info, actions, titles) =
        compute_convert_import_actions(&[("pkg.a", code_a), ("pkg.b", code_b)], "pkg.a", selection);
    assert!(
        titles
            .iter()
            .any(|title| title == "Convert import to relative path"),
        "expected relative import code action"
    );
    let idx = titles
        .iter()
        .position(|title| title == "Convert import to relative path")
        .unwrap();
    let updated = apply_refactor_edits_for_module(&module_info, &actions[idx]);
    let expected = r#"
# IMPORT-START
from . import b as mod_b
# IMPORT-END
"#;
    assert_eq!(expected.trim(), updated.trim());
}
