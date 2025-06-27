/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use pretty_assertions::assert_eq;
use ruff_text_size::TextSize;

use crate::state::handle::Handle;
use crate::state::state::State;
use crate::test::util::get_batched_lsp_operations_report;
use crate::test::util::get_batched_lsp_operations_report_allow_error;

fn get_test_report(state: &State, handle: &Handle, position: TextSize) -> String {
    if let Some(t) = state.transaction().get_type_at(handle, position) {
        format!("Hover Result: `{}`", t)
    } else {
        "Hover Result: None".to_owned()
    }
}

#[test]
fn basic_test() {
    let code = r#"
from typing import Literal
#        ^     ^        ^
def f(x: list[int], y: str, z: Literal[42]):
#   ^               ^       ^
    return x
#          ^
yyy = f([1, 2, 3], "test", 42)
#     ^
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
2 | from typing import Literal
             ^
Hover Result: `Module[typing]`

2 | from typing import Literal
                   ^
Hover Result: None

2 | from typing import Literal
                            ^
Hover Result: `type[Literal]`

4 | def f(x: list[int], y: str, z: Literal[42]):
        ^
Hover Result: `(x: list[int], y: str, z: Literal[42]) -> list[int]`

4 | def f(x: list[int], y: str, z: Literal[42]):
                        ^
Hover Result: `str`

4 | def f(x: list[int], y: str, z: Literal[42]):
                                ^
Hover Result: `Literal[42]`

6 |     return x
               ^
Hover Result: `list[int]`

8 | yyy = f([1, 2, 3], "test", 42)
          ^
Hover Result: `(x: list[int], y: str, z: Literal[42]) -> list[int]`
"#
        .trim(),
        report.trim(),
    );
}

// TODO(kylei): redefinitions should work. they are especially common in try/except blocks
#[test]
fn redefinition_test() {
    let code = r#"
def f(): ...
#   ^
def f(): ...
#   ^
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
2 | def f(): ...
        ^
Hover Result: None

4 | def f(): ...
        ^
Hover Result: None
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn import_test() {
    let code = r#"
import typing
#        ^
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
2 | import typing
             ^
Hover Result: `Module[typing]`
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn import_alias_test() {
    let code = r#"
import typing as t
#                ^
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
2 | import typing as t
                     ^
Hover Result: `Module[typing]`
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn duplicate_import_test() {
    let code = r#"
from typing import List
import typing
#        ^
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
3 | import typing
             ^
Hover Result: `Module[typing]`
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn dead_code_tests() {
    let code = r#"
if 1 == 0:
  def f():
  #   ^
      pass

  x = 3
# ^
  x
# ^
  f
# ^
if False:
  def f():
  #   ^
      pass

  x = 3
# ^
  x
# ^
  f
# ^
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
3 |   def f():
          ^
Hover Result: None

7 |   x = 3
      ^
Hover Result: None

9 |   x
      ^
Hover Result: None

11 |   f
       ^
Hover Result: None

14 |   def f():
           ^
Hover Result: None

18 |   x = 3
       ^
Hover Result: None

20 |   x
       ^
Hover Result: None

22 |   f
       ^
Hover Result: None
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn attribute_tests() {
    let code = r#"
class MyClass:
  x = 5
c1 = MyClass()
c1.x
#  ^

class MyClassWithImplicitField:
  def __init__(self, name: str):
    self.name = name

c2 = MyClassWithImplicitField("")
c2.name
#  ^

class ExtendsMyClass(MyClass):
  y = 6
c3 = ExtendsMyClass()
c3.x
#  ^
c3.y
#  ^

class Union1:
  x = 5

class Union2:
  x = 6

c4: Union1 | Union2 = Union1()
c4.x
#  ^
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
5 | c1.x
       ^
Hover Result: `int`

13 | c2.name
        ^
Hover Result: `str`

19 | c3.x
        ^
Hover Result: `int`

21 | c3.y
        ^
Hover Result: `int`

31 | c4.x
        ^
Hover Result: `int`
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn var_expansion_test() {
    let code = r#"
x = 5
while True:
  x = x + 1
y = x
#   ^
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
5 | y = x
        ^
Hover Result: `int`
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn overloaded_functions_test() {
    let code = r#"
from typing import overload

@overload
def overloaded_func(a: str) -> bool: ...
@overload
def overloaded_func(a: int, b: bool) -> str: ...
def overloaded_func():
    pass

overloaded_func("")
# ^
overloaded_func(1, True)
# ^
overloaded_func(False)
# ^

"#;
    let report = get_batched_lsp_operations_report_allow_error(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
11 | overloaded_func("")
       ^
Hover Result: `(a: str) -> bool`

13 | overloaded_func(1, True)
       ^
Hover Result: `(a: int, b: bool) -> str`

15 | overloaded_func(False)
       ^
Hover Result: `Overload[(a: str) -> bool, (a: int, b: bool) -> str]`
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn overloaded_methods_test() {
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
foo.overloaded_meth("")
#       ^
foo.overloaded_meth(1, True)
#       ^
foo.overloaded_meth(False)
#       ^
"#;
    let report = get_batched_lsp_operations_report_allow_error(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
13 | foo.overloaded_meth("")
             ^
Hover Result: `(self: Foo, a: str) -> bool`

15 | foo.overloaded_meth(1, True)
             ^
Hover Result: `(self: Foo, a: int, b: bool) -> str`

17 | foo.overloaded_meth(False)
             ^
Hover Result: `BoundMethod[Foo, Overload[(self: Self@Foo, a: str) -> bool, (self: Self@Foo, a: int, b: bool) -> str]]`
"#
        .trim(),
        report.trim(),
    );
}
