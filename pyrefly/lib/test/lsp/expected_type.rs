/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Tests for `get_expected_type_at`, which reports the type a checked
//! expression is expected to have at a given position. The expected type is
//! recorded at every `check_and_return_type` call site, so it is available for
//! annotated assignments, attribute/subscript targets with declared types,
//! `return`, `yield`, and `TypedDict` values. Positions that are checked
//! through other paths (call arguments, augmented assignment, overload
//! resolution, lambda bodies, `for` iterables) currently report no expected
//! type; those cases are documented here so the coverage is explicit.

use pretty_assertions::assert_eq;
use pyrefly_build::handle::Handle;
use ruff_text_size::TextSize;

use crate::state::state::State;
use crate::test::util::get_batched_lsp_operations_report_allow_error;

fn get_test_report(state: &State, handle: &Handle, position: TextSize) -> String {
    match state.transaction().get_expected_type_at(handle, position) {
        Some(t) => format!("Expected Type: `{t}`"),
        None => "Expected Type: None".to_owned(),
    }
}

#[test]
fn test_expected_type_assign() {
    let code = r#"
x: int = "hello"
#        ^
y: int = 1
#        ^
z: int | str = 1.5
#              ^
w: int = None
#        ^
a: list[int] = []
#              ^
b: dict[str, int] = {}
#                   ^
"#;
    let report = get_batched_lsp_operations_report_allow_error(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
2 | x: int = "hello"
             ^
Expected Type: `int`

4 | y: int = 1
             ^
Expected Type: `int`

6 | z: int | str = 1.5
                   ^
Expected Type: `int | str`

8 | w: int = None
             ^
Expected Type: `int`

10 | a: list[int] = []
                    ^
Expected Type: `list[int]`

12 | b: dict[str, int] = {}
                         ^
Expected Type: `dict[str, int]`
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn test_expected_type_annotated_name() {
    let code = r#"
x: int
x = "hello"
#   ^
"#;
    let report = get_batched_lsp_operations_report_allow_error(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
3 | x = "hello"
        ^
Expected Type: `int`
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn test_expected_type_return() {
    let code = r#"
def f() -> str:
    return 1
    #      ^
def g() -> int:
    return 1
    #      ^
"#;
    let report = get_batched_lsp_operations_report_allow_error(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
3 |     return 1
               ^
Expected Type: `str`

6 |     return 1
               ^
Expected Type: `int`
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn test_expected_type_attribute() {
    let code = r#"
class C:
    x: int
c = C()
c.x = "hello"
#     ^
c.x = 1
#     ^
"#;
    let report = get_batched_lsp_operations_report_allow_error(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
5 | c.x = "hello"
          ^
Expected Type: `int`

7 | c.x = 1
          ^
Expected Type: `int`
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn test_expected_type_yield() {
    let code = r#"
from typing import Generator
def f() -> Generator[int, None, None]:
    yield "hello"
    #      ^
def g() -> Generator[int, None, None]:
    yield from ["hello"]
    #             ^
"#;
    let report = get_batched_lsp_operations_report_allow_error(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
4 |     yield "hello"
               ^
Expected Type: `int`

7 |     yield from ["hello"]
                      ^
Expected Type: `Generator[int, None, Unknown]`
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn test_expected_type_typed_dict() {
    let code = r#"
from typing import TypedDict
class TD(TypedDict):
    x: int
d: TD = {"x": "hello"}
#             ^
"#;
    let report = get_batched_lsp_operations_report_allow_error(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
5 | d: TD = {"x": "hello"}
                  ^
Expected Type: `int`
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn test_expected_type_call_arg() {
    let code = r#"
def f(x: int) -> None:
    pass
f("hello")
#  ^
f(1)
# ^
def h(x: int, y: str) -> None:
    pass
h(x=1, y=2)
#        ^
def i(x: int) -> int:
    return x
def j(y: str) -> None:
    pass
j(i("hello"))
#    ^
"#;
    let report = get_batched_lsp_operations_report_allow_error(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
4 | f("hello")
       ^
Expected Type: `int`

6 | f(1)
      ^
Expected Type: `int`

10 | h(x=1, y=2)
              ^
Expected Type: `str`

16 | j(i("hello"))
          ^
Expected Type: `int`
"#
        .trim(),
        report.trim(),
    );
}

// The cases below are checked through paths that do not (yet) record an
// expected type, so they report `None`. They document the current coverage gap.

#[test]
fn test_expected_type_aug_subscript_not_recorded() {
    let code = r#"
x: int = 1
x += "hello"
#     ^
y: list[int] = []
y[0] = "hello"
#      ^
"#;
    let report = get_batched_lsp_operations_report_allow_error(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
3 | x += "hello"
          ^
Expected Type: None

6 | y[0] = "hello"
           ^
Expected Type: `Iterable[int]`
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn test_expected_type_overload() {
    // `1.5` matches neither overload, but call-argument prediction reports the
    // first overload's parameter type as a best guess of what was expected.
    let code = r#"
from typing import overload
@overload
def f(x: int) -> int: ...
@overload
def f(x: str) -> str: ...
def f(x):
    return x
f(1.5)
# ^
"#;
    let report = get_batched_lsp_operations_report_allow_error(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
9 | f(1.5)
      ^
Expected Type: `int`
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn test_expected_type_lambda_arg() {
    // The lambda is an argument to `f`, so its expected type is the parameter
    // type `Callable[[int], None]`. A position inside the lambda body has no
    // recorded expected type of its own (operator operands aren't recorded), so
    // walking outward reports the enclosing lambda argument's expected type.
    let code = r#"
from typing import Callable
def f(g: Callable[[int], None]) -> None:
    pass
f(lambda x: x + "hello")
#               ^
"#;
    let report = get_batched_lsp_operations_report_allow_error(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
5 | f(lambda x: x + "hello")
                    ^
Expected Type: `(int) -> None`
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn test_expected_type_for_loop_not_recorded() {
    let code = r#"
for x in []:
#        ^
    pass
for y in [1, 2, 3]:
#        ^
    pass
"#;
    let report = get_batched_lsp_operations_report_allow_error(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
2 | for x in []:
             ^
Expected Type: None

5 | for y in [1, 2, 3]:
             ^
Expected Type: None
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn test_no_expected_type() {
    let code = r#"
x = 1
#   ^
1 + 1
# ^
def f():
    #^
    pass
"#;
    let report = get_batched_lsp_operations_report_allow_error(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
2 | x = 1
        ^
Expected Type: None

4 | 1 + 1
      ^
Expected Type: `int`

6 | def f():
         ^
Expected Type: None
"#
        .trim(),
        report.trim(),
    );
}
