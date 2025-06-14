/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use pretty_assertions::assert_eq;
use ruff_text_size::TextSize;

use crate::module::module_info::TextRangeWithModuleInfo;
use crate::state::handle::Handle;
use crate::state::state::State;
use crate::test::util::code_frame_of_source_at_range;
use crate::test::util::get_batched_lsp_operations_report;
use crate::test::util::get_batched_lsp_operations_report_allow_error;

fn get_test_report(state: &State, handle: &Handle, position: TextSize) -> String {
    if let Some(TextRangeWithModuleInfo { module_info, range }) =
        state.transaction().goto_definition(handle, position)
    {
        format!(
            "Definition Result:\n{}",
            code_frame_of_source_at_range(module_info.contents(), range)
        )
    } else {
        "Definition Result: None".to_owned()
    }
}

#[test]
fn ignored_test() {
    let code = r#"
x = 1 # go-to-definition is unsupported for literals
#   ^
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
2 | x = 1 # go-to-definition is unsupported for literals
        ^
Definition Result: None
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn no_crash_on_dead_branch_test() {
    let code = r#"
from typing import TYPE_CHECKING

if not TYPE_CHECKING:
    x = 1
#   ^
    y = x
#       ^
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    // We are just testing that gotodef won't crash on these examples
    assert_eq!(
        r#"
# main.py
5 |     x = 1
        ^
Definition Result: None

7 |     y = x
            ^
Definition Result: None
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn basic_test() {
    let code = r#"
from typing import Literal

def f(x: list[int], y: str, z: Literal[42]):
#   ^                           ^
    return x

yyy = f([1, 2, 3], "test", 42)
#     ^

class A: pass
class B(A): pass
#       ^
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
4 | def f(x: list[int], y: str, z: Literal[42]):
        ^
Definition Result:
4 | def f(x: list[int], y: str, z: Literal[42]):
        ^

4 | def f(x: list[int], y: str, z: Literal[42]):
                                    ^
Definition Result:
250 | Literal: _SpecialForm
      ^^^^^^^

8 | yyy = f([1, 2, 3], "test", 42)
          ^
Definition Result:
4 | def f(x: list[int], y: str, z: Literal[42]):
        ^

12 | class B(A): pass
             ^
Definition Result:
11 | class A: pass
           ^
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn narrow_test() {
    let code = r#"
def f(x: int | None) -> int:
    if x is None:
        raise ValueError("x is None")
    return 0 if x else 1
#               ^
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
5 |     return 0 if x else 1
                    ^
Definition Result:
2 | def f(x: int | None) -> int:
          ^
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn narrow_test_consecutive() {
    let code = r#"
def f(x: list[int]) -> None:
    x[0] = 0
    x[1] = 1
#   ^

class C:
    y: int
    z: int
def g(x: C) -> None:
    x.y = 0
    x.z = 1
#   ^
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
4 |     x[1] = 1
        ^
Definition Result:
2 | def f(x: list[int]) -> None:
          ^

12 |     x.z = 1
         ^
Definition Result:
10 | def g(x: C) -> None:
           ^
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn shadowed_def_test0() {
    let code = r#"
def test() -> None:
  x = 0
# ^
  x = 1
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
3 |   x = 0
      ^
Definition Result:
3 |   x = 0
      ^
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn shadowed_def_test1() {
    let code = r#"
def test(flag: bool) -> None:
  x = 0
  if flag:
    x = 1
#   ^
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
5 |     x = 1
        ^
Definition Result:
5 |     x = 1
        ^
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn function_and_class_name_test() {
    let code = r#"
def foo() -> None:
#   ^
  pass

class Foo:
#     ^
  def bar(self) -> int:
#     ^
    return 42
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
2 | def foo() -> None:
        ^
Definition Result:
2 | def foo() -> None:
        ^^^

6 | class Foo:
          ^
Definition Result:
6 | class Foo:
          ^^^

8 |   def bar(self) -> int:
          ^
Definition Result:
8 |   def bar(self) -> int:
          ^^^
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn exception_handler_name_test() {
    let code = r#"
def test(flag: bool) -> None:
  try:
    1 / 0
  except Exception as e:
#                     ^
    pass
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
5 |   except Exception as e:
                          ^
Definition Result:
5 |   except Exception as e:
                          ^
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn pattern_match_test() {
    let code = r#"
from typing import Any
class Foo:
  x: int

def test(o: Any) -> None:
  match o:
    case Foo(x=0): pass
# NOTE(grievejia): The keyword case doesn't work currently because of a visitor bug in Ruff
    case [*args]: pass
#          ^
    case {**kwargs}: pass
#           ^
    case _ as y: pass
#             ^
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
10 |     case [*args]: pass
                ^
Definition Result:
10 |     case [*args]: pass
                ^^^^

12 |     case {**kwargs}: pass
                 ^
Definition Result:
12 |     case {**kwargs}: pass
                 ^^^^^^

14 |     case _ as y: pass
                   ^
Definition Result:
14 |     case _ as y: pass
                   ^
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn keyword_argument_test_function() {
    let code = r#"
def foo(x: int, y: str) -> None: pass
def bar(x: int, *, y: str) -> None: pass
def baz(x: int, /) -> None: pass

def test() -> None:
  foo(0, y="foo")
#        ^
  bar(1, y="bar")
#        ^
  baz(x=0)
#     ^
"#;
    let report = get_batched_lsp_operations_report_allow_error(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
7 |   foo(0, y="foo")
             ^
Definition Result:
2 | def foo(x: int, y: str) -> None: pass
                    ^

9 |   bar(1, y="bar")
             ^
Definition Result:
3 | def bar(x: int, *, y: str) -> None: pass
                       ^

11 |   baz(x=0)
           ^
Definition Result:
4 | def baz(x: int, /) -> None: pass
        ^^^
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn keyword_argument_test_method() {
    let code = r#"
class Foo:
    def foo(self, x: int, y: str) -> None:
        pass
    def bar(self) -> None:
        pass

def test(a: Foo) -> None:
    a.foo(0, y="foo")
#            ^
    a.bar(x=1)
#         ^
"#;
    let report = get_batched_lsp_operations_report_allow_error(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
9 |     a.foo(0, y="foo")
                 ^
Definition Result:
3 |     def foo(self, x: int, y: str) -> None:
                              ^

11 |     a.bar(x=1)
               ^
Definition Result:
5 |     def bar(self) -> None:
            ^^^
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn keyword_argument_multi_file() {
    let code_fuction_provider = r#"
def foo(x: int, y: str) -> None:
    pass
"#;
    let code = r#"
from .my_func import foo
foo(0, y="foo")
#      ^
"#;
    let report = get_batched_lsp_operations_report(
        &[("main", code), ("my_func", code_fuction_provider)],
        get_test_report,
    );
    assert_eq!(
        r#"
# main.py
3 | foo(0, y="foo")
           ^
Definition Result:
2 | def foo(x: int, y: str) -> None:
                    ^


# my_func.py
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn named_import_tests() {
    let code_import_provider: &str = r#"
from typing import Literal

def f(x: list[int], y: str, z: Literal[42]):
    return x
"#;
    let code_test: &str = r#"
from typing import Literal
from .import_provider import f
# ^         ^                ^

foo: Literal[1] = 1
#        ^
bar = f([1], "", 42)
#     ^
"#;

    let report = get_batched_lsp_operations_report(
        &[
            ("main", code_test),
            ("import_provider", code_import_provider),
        ],
        get_test_report,
    );
    assert_eq!(
        r#"
# main.py
3 | from .import_provider import f
      ^
Definition Result: None

3 | from .import_provider import f
                ^
Definition Result:
1 | 
    ^

3 | from .import_provider import f
                                 ^
Definition Result:
4 | def f(x: list[int], y: str, z: Literal[42]):
        ^

6 | foo: Literal[1] = 1
             ^
Definition Result:
250 | Literal: _SpecialForm
      ^^^^^^^

8 | bar = f([1], "", 42)
          ^
Definition Result:
4 | def f(x: list[int], y: str, z: Literal[42]):
        ^


# import_provider.py
"#
        .trim(),
        report.trim()
    );
}

#[test]
fn aliased_import_tests() {
    let code_import_provider: &str = r#"
# top of module
class Foo: pass
    "#;
    let code_test: &str = r#"
from import_provider import Foo as F
#                                  ^
import import_provider as ip
#                         ^

def f(x: ip.Foo, y: F):
#        ^          ^
    return x
"#;

    let report = get_batched_lsp_operations_report(
        &[
            ("main", code_test),
            ("import_provider", code_import_provider),
        ],
        get_test_report,
    );
    assert_eq!(
        r#"
# main.py
2 | from import_provider import Foo as F
                                       ^
Definition Result:
3 | class Foo: pass
          ^^^

4 | import import_provider as ip
                              ^
Definition Result:
1 | 
    ^

7 | def f(x: ip.Foo, y: F):
             ^
Definition Result:
1 | 
    ^

7 | def f(x: ip.Foo, y: F):
                        ^
Definition Result:
3 | class Foo: pass
          ^^^


# import_provider.py
"#
        .trim(),
        report.trim()
    );
}

#[test]
fn incorrect_import_tests() {
    let code_test: &str = r#"
from .....import_provider import baz
baz
# ^
"#;
    let report =
        get_batched_lsp_operations_report_allow_error(&[("main", code_test)], get_test_report);
    assert_eq!(
        r#"
# main.py
3 | baz
      ^
Definition Result:
2 | from .....import_provider import baz
                                     ^^^
"#
        .trim(),
        report.trim()
    )
}

#[test]
fn star_import_tests() {
    let code_import_provider: &str = r#"
def f():
    pass
"#;
    let code_test: &str = r#"
from .import_provider import *

bar = f() # should jump to definition in import_provider
#     ^
"#;

    let report = get_batched_lsp_operations_report(
        &[
            ("main", code_test),
            ("import_provider", code_import_provider),
        ],
        get_test_report,
    );
    assert_eq!(
        r#"
# main.py
4 | bar = f() # should jump to definition in import_provider
          ^
Definition Result:
2 | def f():
        ^


# import_provider.py
"#
        .trim(),
        report.trim()
    );
}

#[test]
fn inline_import_test() {
    let code_import_provider: &str = r#"
def f():
        pass"#;
    let code_test: &str = r#"
def foo() -> None:
    from .import_provider import f
    bar = f()
    #     ^
    "#;
    let report = get_batched_lsp_operations_report(
        &[
            ("main", code_test),
            ("import_provider", code_import_provider),
        ],
        get_test_report,
    );
    assert_eq!(
        r#"
# main.py
4 |     bar = f()
              ^
Definition Result:
2 | def f():
        ^


# import_provider.py
"#
        .trim(),
        report.trim()
    );
}

#[test]
fn import_of_import_test() {
    let code_import_provider2: &str = r#"
def f():
        pass"#;
    let code_import_provider: &str = r#"
from .import_provider2 import f
"#;
    let code_test: &str = r#"
from .import_provider import *

bar = f() # should jump to definition in import_provider2
#     ^
    "#;
    let report = get_batched_lsp_operations_report(
        &[
            ("main", code_test),
            ("import_provider", code_import_provider),
            ("import_provider2", code_import_provider2),
        ],
        get_test_report,
    );
    assert_eq!(
        r#"
# main.py
4 | bar = f() # should jump to definition in import_provider2
          ^
Definition Result:
2 | def f():
        ^


# import_provider.py

# import_provider2.py
"#
        .trim(),
        report.trim()
    );
}

#[test]
fn unresolved_named_import_test() {
    let code: &str = r#"
from .unresolved_import import f

bar = f()
#     ^
"#;

    let report = get_batched_lsp_operations_report_allow_error(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
4 | bar = f()
          ^
Definition Result:
2 | from .unresolved_import import f
                                   ^
"#
        .trim(),
        report.trim()
    );
}

#[test]
fn unresolved_star_import_test() {
    let code: &str = r#"
from .unresolved_import import *

bar = f()
#     ^
"#;

    let report = get_batched_lsp_operations_report_allow_error(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
4 | bar = f()
          ^
Definition Result: None

"#
        .trim(),
        report.trim()
    );
}

#[test]
fn goto_def_dead_code() {
    let code: &str = r#"
if False:
    x
#   ^
"#;

    let report = get_batched_lsp_operations_report_allow_error(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
3 |     x
        ^
Definition Result: None
"#
        .trim(),
        report.trim()
    );
}

#[test]
fn multi_definition_test() {
    let code = r#"
if True:
    xxxx = 1
else:
    xxxx = 2
xxxx # it's reasonable to only return the first def, but also reasonable to return both defs
# ^
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
6 | xxxx # it's reasonable to only return the first def, but also reasonable to return both defs
      ^
Definition Result:
3 |     xxxx = 1
        ^^^^
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn untyped_param_test() {
    let code = r#"
def f(untyped):
  untyped
#    ^
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
3 |   untyped
         ^
Definition Result:
2 | def f(untyped):
          ^^^^^^^
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn global_reference_test() {
    let code = r#"
x = 3
def f(untyped):
  global x
  x
# ^
"#;
    let report = get_batched_lsp_operations_report_allow_error(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
5 |   x
      ^
Definition Result:
2 | x = 3
    ^
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn reassigned_test() {
    let code = r#"
xxxx = 1
xxxx = 2
xxxx # should jump to the most recent definition
# ^
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
4 | xxxx # should jump to the most recent definition
      ^
Definition Result:
3 | xxxx = 2
    ^^^^
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn param_def_test() {
    let code = r#"
def f(x, /, y, *, z):
#     ^     ^     ^
  pass
def g(*args, **kwargs):
#      ^       ^
  pass
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
2 | def f(x, /, y, *, z):
          ^
Definition Result:
2 | def f(x, /, y, *, z):
          ^

2 | def f(x, /, y, *, z):
                ^
Definition Result:
2 | def f(x, /, y, *, z):
                ^

2 | def f(x, /, y, *, z):
                      ^
Definition Result:
2 | def f(x, /, y, *, z):
                      ^

5 | def g(*args, **kwargs):
           ^
Definition Result:
5 | def g(*args, **kwargs):
           ^^^^

5 | def g(*args, **kwargs):
                   ^
Definition Result:
5 | def g(*args, **kwargs):
                   ^^^^^^
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn generics_test() {
    let code = r#"
def f[T](input: T):
#     ^         ^
  pass
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
2 | def f[T](input: T):
          ^
Definition Result:
2 | def f[T](input: T):
          ^

2 | def f[T](input: T):
                    ^
Definition Result:
2 | def f[T](input: T):
          ^
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn property_test() {
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

dict = {"foo": '', "bar": 3}
dict["foo"]
#      ^
dict["bar"]
#      ^
"#;
    // TODO: property go-to-definition is unsupported
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
5 | c1.x
       ^
Definition Result:
3 |   x = 5
      ^

13 | c2.name
        ^
Definition Result:
10 |     self.name = name
              ^^^^

19 | c3.x
        ^
Definition Result:
3 |   x = 5
      ^

21 | c3.y
        ^
Definition Result:
17 |   y = 6
       ^

31 | c4.x
        ^
Definition Result: None

35 | dict["foo"]
            ^
Definition Result: None

37 | dict["bar"]
            ^
Definition Result: None
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn cross_module_property_test() {
    let code_class_provider = r#"
class MyClass:
  x = 5
"#;
    let code = r#"
from .my_class import MyClass
c1 = MyClass()
c1.x
#  ^
"#;
    let report = get_batched_lsp_operations_report(
        &[("main", code), ("my_class", code_class_provider)],
        get_test_report,
    );
    assert_eq!(
        r#"
# main.py
4 | c1.x
       ^
Definition Result:
3 |   x = 5
      ^


# my_class.py
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn module_attribute_test() {
    let code_class_provider = r#"
class MyClass:
  x = 5
"#;
    let code = r#"
import my_class
my_class.MyClass
#        ^
"#;
    let report = get_batched_lsp_operations_report(
        &[("main", code), ("my_class", code_class_provider)],
        get_test_report,
    );
    assert_eq!(
        r#"
# main.py
3 | my_class.MyClass
             ^
Definition Result:
2 | class MyClass:
          ^^^^^^^


# my_class.py
"#
        .trim(),
        report.trim(),
    );
}
