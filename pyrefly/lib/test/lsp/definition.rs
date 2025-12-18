/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use itertools::Itertools as _;
use pretty_assertions::assert_eq;
use pyrefly_build::handle::Handle;
use pyrefly_python::module::TextRangeWithModule;
use ruff_text_size::TextSize;

use crate::state::state::State;
use crate::test::util::code_frame_of_source_at_range;
use crate::test::util::get_batched_lsp_operations_report;
use crate::test::util::get_batched_lsp_operations_report_allow_error;

fn get_test_report(state: &State, handle: &Handle, position: TextSize) -> String {
    let defs = state.transaction().goto_definition(handle, position);
    if !defs.is_empty() {
        defs.into_iter()
            .map(
                |TextRangeWithModule {
                     module: module_info,
                     range,
                 }| {
                    format!(
                        "Definition Result:\n{}",
                        code_frame_of_source_at_range(module_info.contents(), range)
                    )
                },
            )
            .join("\n")
    } else {
        "Definition Result: None".to_owned()
    }
}

fn get_test_report_do_not_jump_through_renamed_import(
    state: &State,
    handle: &Handle,
    position: TextSize,
) -> String {
    if let Some(TextRangeWithModule {
        module: module_info,
        range,
    }) = state
        .transaction()
        .goto_definition_do_not_jump_through_renamed_import(handle, position)
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
    println!("REPORT=>{}<=", report);
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
249 | Literal: _SpecialForm
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
fn branch_with_pinning_test() {
    let code = r#"
def test(flag: bool, x: str | None) -> None:
    if flag:
        y = "derp"
        x = y
    x
#   ^
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    // Why didn't it go to the parameter `x`? That's because we don't explore all branches of Phi nodes
    // in gotodef yet
    assert_eq!(
        r#"
# main.py
6 |     x
        ^
Definition Result:
5 |         x = y
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
fn keyword_argument_test_multiple_methods() {
    let code = r#"
class A:
    def foo(self, x: int, y: str) -> None:
        pass
class B:
    def foo(self, y: str, x: int) -> None:
        pass

def test(u: A | B) -> None:
    u.foo(x=0, y="foo")
#              ^
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
10 |     u.foo(x=0, y="foo")
                    ^
Definition Result:
3 |     def foo(self, x: int, y: str) -> None:
                              ^
Definition Result:
6 |     def foo(self, y: str, x: int) -> None:
                      ^
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn keyword_argument_multi_file() {
    let code_function_provider = r#"
def foo(x: int, y: str) -> None:
    pass
"#;
    let code = r#"
from .my_func import foo
foo(0, y="foo")
#      ^
"#;
    let report = get_batched_lsp_operations_report(
        &[("main", code), ("my_func", code_function_provider)],
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
249 | Literal: _SpecialForm
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

    let report = get_batched_lsp_operations_report(
        &[
            ("main", code_test),
            ("import_provider", code_import_provider),
        ],
        get_test_report_do_not_jump_through_renamed_import,
    );
    assert_eq!(
        r#"
# main.py
2 | from import_provider import Foo as F
                                       ^
Definition Result:
2 | from import_provider import Foo as F
                                       ^

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
2 | from import_provider import Foo as F
                                       ^


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
fn legacy_typevar_test() {
    let code = r#"
from pathlib import Path

class A[T, U]:
    pass

class B:
    pass

def f(x: A[B, Path]) -> None:
#          ^  ^
    pass
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
10 | def f(x: A[B, Path]) -> None:
                ^
Definition Result:
7 | class B:
          ^

10 | def f(x: A[B, Path]) -> None:
                   ^
Definition Result:
173 | class Path(PurePath):
            ^^^^
"#
        .trim(),
        report.trim(),
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

25 | dict["foo"]
            ^
Definition Result:
3618 |     def __getitem__(self, key: _KT, /) -> _VT:
               ^^^^^^^^^^^

27 | dict["bar"]
            ^
Definition Result:
3618 |     def __getitem__(self, key: _KT, /) -> _VT:
               ^^^^^^^^^^^
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn import_module_test() {
    let code_x_init = r#"# x/__init__.py
def f():
    pass
"#;
    let code_x_y = r#"# x/y.py
def g():
    pass
"#;
    let code = r#"
import x.y
def test():
    x.f()
#   ^
"#;
    let report = get_batched_lsp_operations_report(
        &[("main", code), ("x", code_x_init), ("x.y", code_x_y)],
        get_test_report,
    );
    assert_eq!(
        r#"
# main.py
4 |     x.f()
        ^
Definition Result:
1 | # x/__init__.py
    ^


# x.py

# x.y.py
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

#[test]
fn union_attribute_access_test() {
    let code = r#"
class A:
    x: int = 0

class B:
    x: str = "abc"

def test(y: A | B) -> int | str:
    return y.x
           # ^
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
9 |     return y.x
                 ^
Definition Result:
3 |     x: int = 0
        ^
Definition Result:
6 |     x: str = "abc"
        ^
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn union_method_access_test() {
    let code = r#"
class State:
    def get_location(self) -> str: ...

class NewYork:
    def get_location(self) -> str:
        return "10016"

class Massachusetts:
    def get_location(self) -> str:
        return "02108"

def find_location(x: Massachusetts | NewYork):
    x.get_location()
      # ^

def find_location2(x: NewYork | Massachusetts):
    x.get_location()
      # ^
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
14 |     x.get_location()
             ^
Definition Result:
10 |     def get_location(self) -> str:
             ^^^^^^^^^^^^
Definition Result:
6 |     def get_location(self) -> str:
            ^^^^^^^^^^^^

18 |     x.get_location()
             ^
Definition Result:
10 |     def get_location(self) -> str:
             ^^^^^^^^^^^^
Definition Result:
6 |     def get_location(self) -> str:
            ^^^^^^^^^^^^
"#
        .trim(),
        report.trim(),
    );
}

// todo(kylei) go-to definition on x should go to the definition
#[test]
fn global_keyword() {
    let code = r#"
x = 5
def test():
    global x
    #      ^
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
4 |     global x
               ^
Definition Result: None
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn module_attr_access_should_follow_exports_not_dunder_all_test() {
    let code_name_provider = r#"
__all__ = []
x: int = 42
"#;
    let code = r#"
import foo
foo.x
#   ^
"#;
    let report = get_batched_lsp_operations_report(
        &[("main", code), ("foo", code_name_provider)],
        get_test_report,
    );
    assert_eq!(
        r#"
# main.py
3 | foo.x
        ^
Definition Result:
3 | x: int = 42
    ^


# foo.py
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn renamed_reexport() {
    let lib2 = r#"
def foo() -> None: ...
"#;
    let lib = r#"
from lib2 import foo as foo_renamed
"#;
    let code = r#"
from lib import foo_renamed
#                    ^
"#;
    let report = get_batched_lsp_operations_report(
        &[("main", code), ("lib", lib), ("lib2", lib2)],
        get_test_report,
    );
    assert_eq!(
        r#"
# main.py
2 | from lib import foo_renamed
                         ^
Definition Result:
2 | def foo() -> None: ...
        ^^^


# lib.py

# lib2.py
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn renamed_assignment_reexport() {
    let lib2 = r#"
def foo() -> None: ...
"#;
    let lib = r#"
from lib2 import foo
foo_renamed = foo
"#;
    let code = r#"
from lib import foo_renamed
#                    ^
"#;
    let report = get_batched_lsp_operations_report(
        &[("main", code), ("lib", lib), ("lib2", lib2)],
        get_test_report,
    );
    assert_eq!(
        r#"
# main.py
2 | from lib import foo_renamed
                         ^
Definition Result:
3 | foo_renamed = foo
    ^^^^^^^^^^^


# lib.py

# lib2.py
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn operator_comparison_overload() {
    let code = r#"
class My:
    def __lt__(self, other: object) -> bool:
        return True
    def __le__(self, other: object) -> bool:
        return True
    def __gt__(self, other: object) -> bool:
        return True
    def __ge__(self, other: object) -> bool:
        return True

a = My()
b = My()
result1 = a < b
#           ^
result2 = a <= b
#            ^
result3 = a > b
#           ^
result4 = a >= b
#            ^
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
14 | result1 = a < b
                 ^
Definition Result:
3 |     def __lt__(self, other: object) -> bool:
            ^^^^^^

16 | result2 = a <= b
                  ^
Definition Result:
5 |     def __le__(self, other: object) -> bool:
            ^^^^^^

18 | result3 = a > b
                 ^
Definition Result:
7 |     def __gt__(self, other: object) -> bool:
            ^^^^^^

20 | result4 = a >= b
                  ^
Definition Result:
9 |     def __ge__(self, other: object) -> bool:
            ^^^^^^
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn unreachable_branch() {
    let code = r#"
x = 5
if False:
    print(x)
    #     ^
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
4 |     print(x)
              ^
Definition Result: None

"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn operator_binop_overload() {
    let code = r#"
class My:
    def __add__(self, other: object) -> "My":
        return self
    def __sub__(self, other: object) -> "My":
        return self
    def __mul__(self, other: object) -> "My":
        return self
    def __truediv__(self, other: object) -> "My":
        return self

a = My()
b = My()
result1 = a + b
#           ^
result2 = a - b
#           ^
result3 = a * b
#           ^
result4 = a / b
#           ^
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
14 | result1 = a + b
                 ^
Definition Result:
3 |     def __add__(self, other: object) -> "My":
            ^^^^^^^

16 | result2 = a - b
                 ^
Definition Result:
5 |     def __sub__(self, other: object) -> "My":
            ^^^^^^^

18 | result3 = a * b
                 ^
Definition Result:
7 |     def __mul__(self, other: object) -> "My":
            ^^^^^^^

20 | result4 = a / b
                 ^
Definition Result:
9 |     def __truediv__(self, other: object) -> "My":
            ^^^^^^^^^^^
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn operator_unaryop_overload() {
    let code = r#"
class My:
    def __neg__(self) -> "My":
        return self
    def __pos__(self) -> "My":
        return self
    def __invert__(self) -> "My":
        return self

a = My()
result1 = -a
#         ^
result2 = +a
#         ^
result3 = ~a
#         ^
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
11 | result1 = -a
               ^
Definition Result:
3 |     def __neg__(self) -> "My":
            ^^^^^^^

13 | result2 = +a
               ^
Definition Result:
5 |     def __pos__(self) -> "My":
            ^^^^^^^

15 | result3 = ~a
               ^
Definition Result:
7 |     def __invert__(self) -> "My":
            ^^^^^^^^^^
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn goto_def_on_none_test() {
    let code = r#"
x = None
#   ^
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    // None should resolve to builtins.pyi
    assert!(report.contains("Definition Result:"));
    assert!(report.contains("None"));
}
