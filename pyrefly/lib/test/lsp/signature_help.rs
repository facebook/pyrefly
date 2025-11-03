/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use itertools::Itertools;
use lsp_types::ParameterLabel;
use lsp_types::SignatureHelp;
use lsp_types::SignatureInformation;
use pretty_assertions::assert_eq;
use pyrefly_build::handle::Handle;
use ruff_text_size::TextSize;

use crate::state::state::State;
use crate::test::util::get_batched_lsp_operations_report_allow_error;

fn get_test_report(state: &State, handle: &Handle, position: TextSize) -> String {
    if let Some(SignatureHelp {
        signatures,
        active_signature,
        active_parameter,
    }) = state.transaction().get_signature_help_at(handle, position)
    {
        let active_signature_result = if let Some(active) = active_signature {
            format!(" active={active}")
        } else {
            "".to_owned()
        };
        let active_parameter_result = if let Some(active) = active_parameter {
            format!(", active parameter={active}")
        } else {
            "".to_owned()
        };
        let signatures_result = signatures
            .into_iter()
            .map(
                |SignatureInformation {
                     label,
                     documentation: _,
                     parameters,
                     active_parameter,
                 }| {
                    format!(
                        "- {}{}{}",
                        label,
                        if let Some(params) = parameters {
                            format!(
                                ", parameters=[{}]",
                                params
                                    .into_iter()
                                    .map(|p| match p.label {
                                        ParameterLabel::Simple(s) => s,
                                        _ => unreachable!(),
                                    })
                                    .join(", ")
                            )
                        } else {
                            "".to_owned()
                        },
                        if let Some(active) = active_parameter {
                            format!(", active parameter = {active}")
                        } else {
                            "".to_owned()
                        }
                    )
                },
            )
            .join("\n");
        // format!("Signature Help Result:{active_signature_result}\n{signatures_result}")
        format!(
            "Signature Help Result:{active_signature_result}{active_parameter_result}\n{signatures_result}"
        )
    } else {
        "Signature Help: None".to_owned()
    }
}

#[test]
fn simple_function_test() {
    let code = r#"
def f(a: str, b: int, c: bool) -> None: ...

f()
# ^
f("", )
#    ^
f("",3, )
#      ^
f("",3,True)
#      ^
"#;
    let report = get_batched_lsp_operations_report_allow_error(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
4 | f()
      ^
Signature Help Result: active=0, active parameter=0
- def f(
    a: str,
    b: int,
    c: bool
) -> None, parameters=[a: str, b: int, c: bool]

6 | f("", )
         ^
Signature Help Result: active=0, active parameter=1
- def f(
    a: str,
    b: int,
    c: bool
) -> None, parameters=[a: str, b: int, c: bool]

8 | f("",3, )
           ^
Signature Help Result: active=0, active parameter=2
- def f(
    a: str,
    b: int,
    c: bool
) -> None, parameters=[a: str, b: int, c: bool]

10 | f("",3,True)
            ^
Signature Help Result: active=0, active parameter=2
- def f(
    a: str,
    b: int,
    c: bool
) -> None, parameters=[a: str, b: int, c: bool]
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn simple_incomplete_function_call_test() {
    let code = r#"
def f(a: str) -> None: ...

f(
# ^
"#;
    let report = get_batched_lsp_operations_report_allow_error(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
4 | f(
      ^
Signature Help Result: active=0, active parameter=0
- def f(a: str) -> None, parameters=[a: str]
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn simple_function_nested_test() {
    let code = r#"
def f(a: str) -> None: ...
def g(b: int) -> None: ...

f()
# ^
f(g())
#   ^
"#;
    let report = get_batched_lsp_operations_report_allow_error(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
5 | f()
      ^
Signature Help Result: active=0, active parameter=0
- def f(a: str) -> None, parameters=[a: str]

7 | f(g())
        ^
Signature Help Result: active=0, active parameter=0
- def g(b: int) -> None, parameters=[b: int]
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn simple_method_test() {
    let code = r#"
class Foo:
  def f(self, a: str, b: int, c: bool) -> None: ...

foo = Foo()
foo.f()
#     ^
foo.f("", )
#        ^
foo.f("",3, )
#          ^
foo.f("",3,True)
#          ^
"#;
    let report = get_batched_lsp_operations_report_allow_error(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
6 | foo.f()
          ^
Signature Help Result: active=0, active parameter=0
- def f(
    self: Foo,
    a: str,
    b: int,
    c: bool
) -> None, parameters=[a: str, b: int, c: bool]

8 | foo.f("", )
             ^
Signature Help Result: active=0, active parameter=1
- def f(
    self: Foo,
    a: str,
    b: int,
    c: bool
) -> None, parameters=[a: str, b: int, c: bool]

10 | foo.f("",3, )
                ^
Signature Help Result: active=0, active parameter=2
- def f(
    self: Foo,
    a: str,
    b: int,
    c: bool
) -> None, parameters=[a: str, b: int, c: bool]

12 | foo.f("",3,True)
                ^
Signature Help Result: active=0, active parameter=2
- def f(
    self: Foo,
    a: str,
    b: int,
    c: bool
) -> None, parameters=[a: str, b: int, c: bool]
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn overloaded_function_test() {
    let code = r#"
from typing import overload


@overload
def overloaded_func(a: str) -> bool: ...
@overload
def overloaded_func(a: int, b: bool) -> str: ...
def overloaded_func():
    pass


overloaded_func()
#               ^
overloaded_func(1, )
#                 ^
overloaded_func(1, T)
#                  ^
"#;
    let report = get_batched_lsp_operations_report_allow_error(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
13 | overloaded_func()
                     ^
Signature Help Result: active=0, active parameter=0
- (a: str) -> bool, parameters=[a: str]
- (
    a: int,
    b: bool
) -> str, parameters=[a: int, b: bool]

15 | overloaded_func(1, )
                       ^
Signature Help Result: active=0, active parameter=1
- (a: str) -> bool, parameters=[a: str]
- (
    a: int,
    b: bool
) -> str, parameters=[a: int, b: bool]

17 | overloaded_func(1, T)
                        ^
Signature Help Result: active=1, active parameter=1
- (a: str) -> bool, parameters=[a: str]
- (
    a: int,
    b: bool
) -> str, parameters=[a: int, b: bool]
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn overloaded_method_test() {
    let code = r#"
from typing import overload


class Foo:
    @overload
    def overloaded_meth(self, a: str) -> bool: ...
    @overload
    def overloaded_meth(self, a: int, b: bool) -> str: ...
    def overloaded_meth(self):
        pass


foo = Foo()
foo.overloaded_meth()
#                   ^
foo.overloaded_meth(1, )
#                      ^
foo.overloaded_meth(1, F)
#                      ^
"#;
    let report = get_batched_lsp_operations_report_allow_error(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
15 | foo.overloaded_meth()
                         ^
Signature Help Result: active=0, active parameter=0
- (
    self: Foo,
    a: str
) -> bool, parameters=[a: str]
- (
    self: Foo,
    a: int,
    b: bool
) -> str, parameters=[a: int, b: bool]

17 | foo.overloaded_meth(1, )
                            ^
Signature Help Result: active=0, active parameter=1
- (
    self: Foo,
    a: str
) -> bool, parameters=[a: str]
- (
    self: Foo,
    a: int,
    b: bool
) -> str, parameters=[a: int, b: bool]

19 | foo.overloaded_meth(1, F)
                            ^
Signature Help Result: active=1, active parameter=1
- (
    self: Foo,
    a: str
) -> bool, parameters=[a: str]
- (
    self: Foo,
    a: int,
    b: bool
) -> str, parameters=[a: int, b: bool]
"#
        .trim(),
        report.trim(),
    );
}
