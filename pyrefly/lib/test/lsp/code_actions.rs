/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use pretty_assertions::assert_eq;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;

use crate::module::module_info::ModuleInfo;
use crate::state::handle::Handle;
use crate::state::state::State;
use crate::test::util::get_batched_lsp_operations_report_allow_error;

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

fn get_test_report_insert_import(state: &State, handle: &Handle, position: TextSize) -> String {
    let mut report = "Code Actions Results:\n".to_owned();
    let transaction = state.transaction();
    for (title, info, range, patch) in transaction
        .local_quickfix_code_actions(handle, TextRange::new(position, position))
        .into_iter()
        .flatten()
        // NOTE: We do NOT patch buffer with ignore comment code action that is
        // tested separately
        .filter(|(title, _, _, _)| !title.contains("ignore"))
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

fn get_test_report_ignore_code_action(
    state: &State,
    handle: &Handle,
    position: TextSize,
) -> String {
    let mut report = "Code Actions Results:\n".to_owned();
    let transaction = state.transaction();
    for (title, info, range, patch) in transaction
        .local_quickfix_code_actions(handle, TextRange::new(position, position))
        .into_iter()
        .flatten()
        // NOTE: We ONLY patch buffer with ignore comment code action so
        // we can test it separately
        .filter(|(title, _, _, _)| title.contains("ignore"))
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

#[test]
fn add_ignore_comment() {
    let report = get_batched_lsp_operations_report_allow_error(
        &[
            ("a", "my_export = 3\nanother_thing = 4"),
            ("b", "from a import another_thing\nmy_export\n# ^"),
            (
                "c",
                r#"
def func(x: int) -> int:
    pass
func(y)
#    ^"#,
            ),
            (
                "d",
                r#"
def func(x: int) -> int:
#                    ^
    pass
func(y)"#,
            ),
        ],
        get_test_report_ignore_code_action,
    );
    assert_eq!(
        r#"
# a.py

# b.py
2 | my_export
      ^
Code Actions Results:
# Title: Suppress UnknownName with ignore comment

## Before:
from a import another_thing
my_export
# ^
## After:
from a import another_thing
# pyrefly: ignore[unknown-name]
my_export
# ^



# c.py
4 | func(y)
         ^
Code Actions Results:
# Title: Suppress UnknownName with ignore comment

## Before:

def func(x: int) -> int:
    pass
func(y)
#    ^
## After:

def func(x: int) -> int:
    pass
# pyrefly: ignore[unknown-name]
func(y)
#    ^



# d.py
2 | def func(x: int) -> int:
                         ^
Code Actions Results:
# Title: Suppress BadReturn with ignore comment

## Before:

def func(x: int) -> int:
#                    ^
    pass
func(y)
## After:

# pyrefly: ignore[bad-return]
def func(x: int) -> int:
#                    ^
    pass
func(y)
"#
        .trim(),
        report.trim()
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
        get_test_report_insert_import,
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
fn insertion_test_comments() {
    let report = get_batched_lsp_operations_report_allow_error(
        &[
            ("a", "my_export = 3\n"),
            ("b", "# i am a comment\nmy_export\n# ^"),
        ],
        get_test_report_insert_import,
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
        get_test_report_insert_import,
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
        get_test_report_insert_import,
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
