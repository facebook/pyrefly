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
        insert_text,
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
        if let Some(insert_text) = insert_text {
            report.push_str(" inserting `");
            report.push_str(&insert_text);
            report.push('`');
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
fn dot_complete_types_test() {
    let code = r#"
class Foo:
    x: int
    def method(self): ...
    @staticmethod
    def static_method(): ...
    @classmethod
    def class_method(cls): ...
foo = Foo()
foo.
#   ^
"#;
    let report = get_batched_lsp_operations_report_allow_error(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
10 | foo.
         ^
Completion Results:
- (Field) x: int
- (Method) method: BoundMethod[Foo, (self: Self@Foo) -> None]
- (Function) static_method: () -> None
- (Method) class_method: BoundMethod[type[Foo], (cls: type[Self@Foo]) -> None]
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
- (Function) bar: () -> None
- (Variable) xxxx: Literal[3]
- (Function) foo: () -> None

8 |     y
        ^
Completion Results:
- (Variable) yyyy: Literal[4]
- (Variable) xxxx: Literal[3]
- (Function) bar: () -> None
- (Function) foo: () -> None
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

#[test]
fn kwargs_completion_basic() {
    let code = r#"
def foo(a: int, b: str): ...
xyz = 5
foo(x
#    ^
"#;
    let report = get_batched_lsp_operations_report_allow_error(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
4 | foo(x
         ^
Completion Results:
- (Variable) a=: int
- (Variable) b=: str
- (Function) foo: (a: int, b: str) -> None
- (Variable) xyz: Literal[5]
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn kwargs_completion_with_existing_args() {
    let code = r#"
def foo(a: int, b: str, c: bool): ...
foo(1, 
#      ^
"#;
    let report = get_batched_lsp_operations_report_allow_error(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
3 | foo(1, 
           ^
Completion Results:
- (Variable) a=: int
- (Variable) b=: str
- (Variable) c=: bool
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn kwargs_completion_method() {
    let code = r#"
class Foo:
    def method(self, x: int, y: str): ...

foo = Foo()
foo.method(
#          ^
"#;
    let report = get_batched_lsp_operations_report_allow_error(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
6 | foo.method(
               ^
Completion Results:
- (Variable) x=: int
- (Variable) y=: str
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn kwargs_completion_kwonly_params() {
    let code = r#"
def foo(a: int, *, b: str, c: bool): ...
foo(
#   ^
"#;
    let report = get_batched_lsp_operations_report_allow_error(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
3 | foo(
        ^
Completion Results:
- (Variable) a=: int
- (Variable) b=: str
- (Variable) c=: bool
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn kwargs_completion_mixed_params() {
    let code = r#"
def foo(a: int, b: str = "default", *, c: bool, d: float = 1.0): ...
foo(
#   ^
"#;
    let report = get_batched_lsp_operations_report_allow_error(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
3 | foo(
        ^
Completion Results:
- (Variable) a=: int
- (Variable) b=: str
- (Variable) c=: bool
- (Variable) d=: float
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn kwargs_completion_no_self_param() {
    let code = r#"
class Foo:
    def test(self, x: int, y: str): ...

Foo().test(
#          ^
"#;
    let report = get_batched_lsp_operations_report_allow_error(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
5 | Foo().test(
               ^
Completion Results:
- (Variable) x=: int
- (Variable) y=: str
"#
        .trim(),
        report.trim(),
    );
}

// todo(kylei): completion on constructor
#[test]
fn kwargs_completion_constructor() {
    let code = r#"
class Foo:
    def __init__(self, x: int, y: str): ...

Foo(
#   ^
"#;
    let report = get_batched_lsp_operations_report_allow_error(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
5 | Foo(
        ^
Completion Results:
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn kwargs_completion_nested_call() {
    let code = r#"
def outer(a: int): ...
def inner(b: str): ...
outer(inner(
#           ^
"#;
    let report = get_batched_lsp_operations_report_allow_error(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
4 | outer(inner(
                ^
Completion Results:
- (Variable) b=: str
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn kwargs_completion_no_completions_for_non_function() {
    let code = r#"
x = 42
x(
# ^
"#;
    let report = get_batched_lsp_operations_report_allow_error(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
3 | x(
      ^
Completion Results:
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn builtins_doesnt_autoimport() {
    let code = r#"
isins
#    ^
"#;
    let report = get_batched_lsp_operations_report_allow_error(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
2 | isins
         ^
Completion Results:
- (Function) isinstance
- (Function) timerfd_settime_ns: from os import timerfd_settime_ns

- (Function) distributions: from importlib.metadata import distributions

- (Function) packages_distributions: from importlib.metadata import packages_distributions

- (Class) MissingHeaderBodySeparatorDefect: from email.errors import MissingHeaderBodySeparatorDefect

- (Function) fix_missing_locations: from ast import fix_missing_locations

- (Class) FirstHeaderLineIsContinuationDefect: from email.errors import FirstHeaderLineIsContinuationDefect
"#
        .trim(),
        report.trim(),
    );
}

// todo(kylei): completion on literal
#[test]
fn completion_literal() {
    let code = r#"
from typing import Literal
def foo(x: Literal['foo']): ...
foo(
#   ^
"#;
    let report = get_batched_lsp_operations_report_allow_error(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
4 | foo(
        ^
Completion Results:
- (Variable) x=: Literal['foo']
"#
        .trim(),
        report.trim(),
    );
}

// todo(kylei): completion on known dict values
// Pyright completes "a", "b"
#[test]
fn completion_dict() {
    let code = r#"
x = {"a": 3, "b", 4}
x["
# ^
"#;
    let report = get_batched_lsp_operations_report_allow_error(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
3 | x["
      ^
Completion Results:
- (Variable) x: dict[int | str, int]
"#
        .trim(),
        report.trim(),
    );
}

// todo(kylei): kwarg completion on overload
#[test]
fn kwargs_completion_literal() {
    let code = r#"
from typing import Literal, overload
@overload
def foo(x: int):
    print(x)
@overload
def foo(y: bool):
    print(y)
foo(
#   ^
"#;
    let report = get_batched_lsp_operations_report_allow_error(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
9 | foo(
        ^
Completion Results:
"#
        .trim(),
        report.trim(),
    );
}
