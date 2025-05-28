/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use lsp_types::CompletionItem;
use pretty_assertions::assert_eq;
use ruff_text_size::TextSize;

use crate::state::handle::Handle;
use crate::state::state::State;
use crate::test::util::get_batched_lsp_operations_report_allow_error;

fn get_test_report(state: &State, handle: &Handle, position: TextSize) -> String {
    let mut report = "Completion Results:".to_owned();
    for CompletionItem {
        label,
        detail,
        kind,
        ..
    } in state.transaction().completion(handle, position)
    {
        report.push_str("\n- (");
        report.push_str(&format!("{:?}", kind.unwrap()));
        report.push_str(") ");
        report.push_str(&label);
        if let Some(detail) = detail {
            report.push_str(": ");
            report.push_str(&detail);
        }
    }
    report
}

#[test]
fn dot_complete_basic_test() {
    let code = r#"
class Foo:
    x: int
foo = Foo()
foo.
#   ^
class Bar(Foo):
    y: int
bar = Bar()
bar.
#   ^
"#;
    let report = get_batched_lsp_operations_report_allow_error(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
5 | foo.
        ^
Completion Results:
- (Field) x: int

10 | bar.
         ^
Completion Results:
- (Field) y: int
- (Field) x: int
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn dot_complete_rankded_test() {
    let code = r#"
class Foo:
    _private: bool
    __special__: str
    y: int
    x: int

foo = Foo()
foo.
#   ^
"#;
    let report = get_batched_lsp_operations_report_allow_error(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
9 | foo.
        ^
Completion Results:
- (Field) y: int
- (Field) x: int
- (Field) _private: bool
- (Field) __special__: str
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn variable_complete_basic_test() {
    let code = r#"
def foo():
  xxxx = 3
  x
# ^
  def bar():
    yyyy = 4;
    y
#   ^
"#;
    let report = get_batched_lsp_operations_report_allow_error(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
4 |   x
      ^
Completion Results:
- (Variable) xxxx: Literal[3]
- (Variable) bar: () -> None
- (Variable) foo: () -> None

8 |     y
        ^
Completion Results:
- (Variable) yyyy: Literal[4]
- (Variable) xxxx: Literal[3]
- (Variable) bar: () -> None
- (Variable) foo: () -> None
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn complete_multi_module() {
    let code = r#"
import lib

def foo(x: lib.Foo):
  x.
#   ^
"#;

    let lib = r#"
# This file needs to be much longer than main, in order to provoke a crash.
# Therefore, we pad it with a bunch of nonsense. This is the first line.
# This is the second line.
from typing import overload

class Foo:
    @property
    @overload
    def magic(x, y) -> bool:
        return True
    @overload
    def magic(x, y, z) -> int:
        return True
"#;

    let report = get_batched_lsp_operations_report_allow_error(
        &[("main", code), ("lib", lib)],
        get_test_report,
    );
    assert_eq!(
        r#"
# main.py
5 |   x.
        ^
Completion Results:
- (Field) magic


# lib.py
"#
        .trim(),
        report.trim(),
    );
}

// TODO(kylei): ruff's ast gives us names = ["imp"] for `from foo imp`
#[test]
fn from_import_imp_test() {
    let foo_code = r#"
imperial_guard = "cool"
"#;
    let main_code = r#"
from foo imp
#          ^
"#;
    let report = get_batched_lsp_operations_report_allow_error(
        &[("main", main_code), ("foo", foo_code)],
        get_test_report,
    );
    assert_eq!(
        r#"
# main.py
2 | from foo imp
               ^
Completion Results:
- (Variable) imperial_guard


# foo.py
"#
        .trim(),
        report.trim(),
    );
}
// TODO(kylei): ruff's ast gives us names = [] for `from foo import <>`
#[test]
fn from_import_empty_test() {
    let foo_code = r#"
imperial_guard = "cool"
"#;
    let main_code = r#"
from foo import 
#              ^
"#;
    let report = get_batched_lsp_operations_report_allow_error(
        &[("main", main_code), ("foo", foo_code)],
        get_test_report,
    );
    assert_eq!(
        r#"
# main.py
2 | from foo import 
                   ^
Completion Results:


# foo.py
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn from_import_basic() {
    let foo_code = r#"
imperial_guard = "cool"
"#;
    let main_code = r#"
from foo import imperial
#          ^           ^
"#;
    let report = get_batched_lsp_operations_report_allow_error(
        &[("main", main_code), ("foo", foo_code)],
        get_test_report,
    );
    assert_eq!(
        r#"
# main.py
2 | from foo import imperial
               ^
Completion Results:

2 | from foo import imperial
                           ^
Completion Results:
- (Variable) imperial_guard


# foo.py
"#
        .trim(),
        report.trim(),
    );
}
