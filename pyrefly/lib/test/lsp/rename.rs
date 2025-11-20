/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use itertools::Itertools;
use pretty_assertions::assert_eq;
use pyrefly_build::handle::Handle;
use ruff_text_size::TextSize;

use crate::state::state::State;
use crate::test::util::code_frame_of_source_at_range;
use crate::test::util::get_batched_lsp_operations_report;

fn get_test_report(state: &State, handle: &Handle, position: TextSize) -> String {
    let transaction = state.transaction();
    let ranges = transaction.find_local_references(handle, position);
    let module_info = transaction.get_module_info(handle).unwrap();
    format!(
        "Rename locations:\n{}",
        ranges
            .into_iter()
            .map(|range| code_frame_of_source_at_range(module_info.contents(), range))
            .join("\n")
    )
}

#[test]
fn test_rename_parameter_updates_keyword_arguments() {
    let code = r#"
def greet(name, message):
    """Greet someone with a message."""
    print(f"{message}, {name}!")
    return name

def another_func():
    result = greet(name="Alice", message="Hello")
#                  ^
    return result
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
8 |     result = greet(name="Alice", message="Hello")
                       ^
Rename locations:
2 | def greet(name, message):
              ^^^^
4 |     print(f"{message}, {name}!")
                            ^^^^
5 |     return name
               ^^^^
8 |     result = greet(name="Alice", message="Hello")
                       ^^^^
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn test_rename_parameter_only_updates_correct_function() {
    let code = r#"
def func1(name, message):
    """First function with name parameter."""
    print(f"{message}, {name}!")
    return name

def func2(name, value):
    """Second function with same name parameter."""
    print(f"Value: {value}, Name: {name}")
    return name

def caller():
    result1 = func1(name="Alice", message="Hello")
#                   ^
    result2 = func2(name="Bob", value=42)
    return result1, result2
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
13 |     result1 = func1(name="Alice", message="Hello")
                         ^
Rename locations:
2 | def func1(name, message):
              ^^^^
4 |     print(f"{message}, {name}!")
                            ^^^^
5 |     return name
               ^^^^
13 |     result1 = func1(name="Alice", message="Hello")
                         ^^^^
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn test_rename_function_parameter_updates_call_sites() {
    let code = r#"
def greet(name, message):
#         ^
    """Greet someone with a message."""
    print(f"{message}, {name}!")
    return name

def caller():
    result1 = greet(name="Alice", message="Hello")
    result2 = greet(message="Hi", name="Bob")
    result3 = greet(name="Charlie", message="Hey")
    return result1, result2, result3
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
2 | def greet(name, message):
              ^
Rename locations:
2 | def greet(name, message):
              ^^^^
5 |     print(f"{message}, {name}!")
                            ^^^^
6 |     return name
               ^^^^
9 |     result1 = greet(name="Alice", message="Hello")
                        ^^^^
10 |     result2 = greet(message="Hi", name="Bob")
                                       ^^^^
11 |     result3 = greet(name="Charlie", message="Hey")
                         ^^^^
"#
        .trim(),
        report.trim(),
    );
}
