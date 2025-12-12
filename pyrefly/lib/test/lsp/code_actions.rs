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
use crate::state::require::Require;
use crate::state::state::State;
use crate::test::util::get_batched_lsp_operations_report_allow_error;
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
    let start_marker = "# EXTRACT-START";
    let end_marker = "# EXTRACT-END";
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
    let selection = find_marked_range(module_info.contents());
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

fn assert_no_extract_action(code: &str) {
    let (_, actions, _) = compute_extract_actions(code);
    assert!(
        actions.is_empty(),
        "expected no extract-function actions, found {}",
        actions.len()
    );
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
#[ignore = "multiple insertion point choices not yet supported"]
fn extract_function_offers_inner_function_option() {
    let code = r#"
def outer(xs):
    # EXTRACT-START
    running = 0
    for x in xs:
        running += x
    # EXTRACT-END
    return running
"#;
    let (_, _, titles) = compute_extract_actions(code);
    assert!(
        titles.iter().any(|title| title.contains("module scope")),
        "expected at least one extract action when control flow is simple"
    );
}
