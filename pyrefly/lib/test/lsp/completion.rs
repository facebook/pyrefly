/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use lsp_types::CompletionItem;
use lsp_types::CompletionItemKind;
use pretty_assertions::assert_eq;
use ruff_text_size::TextSize;

use crate::state::handle::Handle;
use crate::state::state::State;
use crate::test::util::get_batched_lsp_operations_report_allow_error;

#[derive(Default)]
struct ResultsFilter {
    include_keywords: bool,
    include_builtins: bool,
}

fn get_default_test_report() -> impl Fn(&State, &Handle, TextSize) -> String {
    get_test_report(ResultsFilter::default())
}

fn get_test_report(filter: ResultsFilter) -> impl Fn(&State, &Handle, TextSize) -> String {
    move |state: &State, handle: &Handle, position: TextSize| {
        let mut report = "Completion Results:".to_owned();
        for CompletionItem {
            label,
            detail,
            kind,
            insert_text,
            data,
            ..
        } in state.transaction().completion(handle, position)
        {
            if (filter.include_keywords || kind != Some(CompletionItemKind::KEYWORD))
                && (filter.include_builtins || data != Some(serde_json::json!("builtin")))
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
        }
        report
    }
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
    let report =
        get_batched_lsp_operations_report_allow_error(&[("main", code)], get_default_test_report());
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
- (Field) x: int
- (Field) y: int
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
    let report =
        get_batched_lsp_operations_report_allow_error(&[("main", code)], get_default_test_report());
    assert_eq!(
        r#"
# main.py
10 | foo.
         ^
Completion Results:
- (Method) class_method: (cls: type[Self@Foo]) -> None
- (Method) method: (self: Self@Foo) -> None
- (Function) static_method: () -> None
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
    let report =
        get_batched_lsp_operations_report_allow_error(&[("main", code)], get_default_test_report());
    assert_eq!(
        r#"
# main.py
9 | foo.
        ^
Completion Results:
- (Field) x: int
- (Field) y: int
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
  y
# ^
  f
# ^
  b
# ^
  def bar():
    yyyy = 4;
    x
#   ^
    y
#   ^
    f
#   ^
    b
#   ^
"#;
    let report =
        get_batched_lsp_operations_report_allow_error(&[("main", code)], get_default_test_report());
    assert_eq!(
        r#"
# main.py
4 |   x
      ^
Completion Results:
- (Variable) xxxx: Literal[3]

6 |   y
      ^
Completion Results:

8 |   f
      ^
Completion Results:
- (Function) foo: () -> None

10 |   b
       ^
Completion Results:
- (Function) bar: () -> None

14 |     x
         ^
Completion Results:
- (Variable) xxxx: Literal[3]

16 |     y
         ^
Completion Results:
- (Variable) yyyy: Literal[4]

18 |     f
         ^
Completion Results:
- (Function) foo: () -> None

20 |     b
         ^
Completion Results:
- (Function) bar: () -> None
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn variable_with_globals_complete_test() {
    let code = r#"
FileExistsOrNot = 1
FileExist
#        ^
"#;
    let report = get_batched_lsp_operations_report_allow_error(
        &[("main", code)],
        get_test_report(ResultsFilter {
            include_builtins: true,
            ..Default::default()
        }),
    );
    assert_eq!(
        r#"
# main.py
3 | FileExist
             ^
Completion Results:
- (Class) FileExistsError
- (Variable) FileExistsOrNot: Literal[1]
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
        get_default_test_report(),
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
        get_default_test_report(),
    );
    assert_eq!(
        r#"
# main.py
2 | from foo imp
               ^
Completion Results:
- (Variable) imperial_guard
- (Variable) __annotations__
- (Variable) __builtins__
- (Variable) __cached__
- (Variable) __debug__
- (Variable) __dict__
- (Variable) __doc__
- (Variable) __file__
- (Variable) __loader__
- (Variable) __name__
- (Variable) __package__
- (Variable) __path__
- (Variable) __spec__


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
        get_default_test_report(),
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
        get_default_test_report(),
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
- (Variable) __annotations__
- (Variable) __builtins__
- (Variable) __cached__
- (Variable) __debug__
- (Variable) __dict__
- (Variable) __doc__
- (Variable) __file__
- (Variable) __loader__
- (Variable) __name__
- (Variable) __package__
- (Variable) __path__
- (Variable) __spec__


# foo.py
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn from_import_relative() {
    let foo_code = r#"
imperial_guard = "cool"
"#;
    let main_code = r#"
from .foo import imperial
#                       ^
"#;
    let report = get_batched_lsp_operations_report_allow_error(
        &[("main", main_code), ("foo", foo_code)],
        get_default_test_report(),
    );
    assert_eq!(
        r#"
# main.py
2 | from .foo import imperial
                            ^
Completion Results:
- (Variable) imperial_guard
- (Variable) __annotations__
- (Variable) __builtins__
- (Variable) __cached__
- (Variable) __debug__
- (Variable) __dict__
- (Variable) __doc__
- (Variable) __file__
- (Variable) __loader__
- (Variable) __name__
- (Variable) __package__
- (Variable) __path__
- (Variable) __spec__


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
    let report =
        get_batched_lsp_operations_report_allow_error(&[("main", code)], get_default_test_report());
    assert_eq!(
        r#"
# main.py
4 | foo(x
         ^
Completion Results:
- (Variable) a=: int
- (Variable) b=: str
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
    let report =
        get_batched_lsp_operations_report_allow_error(&[("main", code)], get_default_test_report());
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
    let report =
        get_batched_lsp_operations_report_allow_error(&[("main", code)], get_default_test_report());
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
    let report =
        get_batched_lsp_operations_report_allow_error(&[("main", code)], get_default_test_report());
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
    let report =
        get_batched_lsp_operations_report_allow_error(&[("main", code)], get_default_test_report());
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
    let report =
        get_batched_lsp_operations_report_allow_error(&[("main", code)], get_default_test_report());
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

#[test]
fn kwargs_completion_dunder_new() {
    let code = r#"
from typing import Self
class Foo:
    def __new__(cls, x: int, y: str) -> Self: ...

Foo(
#   ^
"#;
    let report =
        get_batched_lsp_operations_report_allow_error(&[("main", code)], get_default_test_report());
    assert_eq!(
        r#"
# main.py
6 | Foo(
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
fn kwargs_completion_dunder_new_incompatible() {
    let code = r#"
class Foo:
    def __new__(cls, x: int, y: str) -> None: ...

Foo(
#   ^
"#;
    let report =
        get_batched_lsp_operations_report_allow_error(&[("main", code)], get_default_test_report());
    assert_eq!(
        r#"
# main.py
5 | Foo(
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
fn kwargs_completion_constructor() {
    let code = r#"
class Foo:
    def __init__(self, x: int, y: str): ...

Foo(
#   ^
"#;
    let report =
        get_batched_lsp_operations_report_allow_error(&[("main", code)], get_default_test_report());
    assert_eq!(
        r#"
# main.py
5 | Foo(
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
fn kwargs_completion_dunder_call_metaclass_constructor() {
    let code = r#"
class Meta(type):
    def __call__(cls, a: int) -> None: ...
class Foo(metaclass=Meta):
    def __init__(self): ...

Foo(
#   ^
"#;
    let report =
        get_batched_lsp_operations_report_allow_error(&[("main", code)], get_default_test_report());
    assert_eq!(
        r#"
# main.py
7 | Foo(
        ^
Completion Results:
- (Variable) a=: int
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
    let report =
        get_batched_lsp_operations_report_allow_error(&[("main", code)], get_default_test_report());
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
    let report =
        get_batched_lsp_operations_report_allow_error(&[("main", code)], get_default_test_report());
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
    let report = get_batched_lsp_operations_report_allow_error(
        &[("main", code)],
        get_test_report(ResultsFilter {
            include_builtins: true,
            ..Default::default()
        }),
    );
    assert_eq!(
        r#"
# main.py
2 | isins
         ^
Completion Results:
- (Function) isinstance
- (Class) FirstHeaderLineIsContinuationDefect: from email.errors import FirstHeaderLineIsContinuationDefect

- (Class) MissingHeaderBodySeparatorDefect: from email.errors import MissingHeaderBodySeparatorDefect

- (Function) distributions: from importlib.metadata import distributions

- (Function) fix_missing_locations: from ast import fix_missing_locations

- (Function) packages_distributions: from importlib.metadata import packages_distributions

- (Function) timerfd_settime_ns: from os import timerfd_settime_ns
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
    let report =
        get_batched_lsp_operations_report_allow_error(&[("main", code)], get_default_test_report());
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
    let report =
        get_batched_lsp_operations_report_allow_error(&[("main", code)], get_default_test_report());
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

#[test]
fn kwargs_completion_overload_basic() {
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
    let report =
        get_batched_lsp_operations_report_allow_error(&[("main", code)], get_default_test_report());
    assert_eq!(
        r#"
# main.py
9 | foo(
        ^
Completion Results:
- (Variable) x=: int
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn kwargs_completion_overload_correct() {
    let code = r#"
from typing import Literal, overload
@overload
def foo(y: bool, z: bool):
print(y)
@overload
def foo(x: int, y: str):
    print(x)
foo(1, 
#      ^
"#;
    let report =
        get_batched_lsp_operations_report_allow_error(&[("main", code)], get_default_test_report());
    assert_eq!(
        r#"
# main.py
9 | foo(1, 
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
fn no_keywords_on_dot_complete() {
    let code = r#"
class Foo: ...
Foo.
#   ^
"#;
    let report = get_batched_lsp_operations_report_allow_error(
        &[("main", code)],
        get_test_report(ResultsFilter {
            include_keywords: true,
            ..Default::default()
        }),
    );
    assert_eq!(
        r#"
# main.py
3 | Foo.
        ^
Completion Results:
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn import_completions_on_builtins() {
    let code = r#"
import typ
#         ^
"#;
    let report = get_batched_lsp_operations_report_allow_error(
        &[("main", code)],
        get_test_report(ResultsFilter {
            include_keywords: true,
            ..Default::default()
        }),
    );
    assert_eq!(
        r#"
# main.py
2 | import typ
              ^
Completion Results:
- (Module) types: types
- (Module) typing: typing
- (Module) typing_extensions: typing_extensions
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn completion_on_empty_line() {
    let code = r#"
def test():
    xyz = 5
    
#   ^
"#;
    let report = get_batched_lsp_operations_report_allow_error(
        &[("main", code)],
        get_test_report(ResultsFilter {
            include_keywords: true,
            include_builtins: true,
        }),
    );
    assert_eq!(
        r#"
# main.py
4 |     
        ^
Completion Results:
- (Class) ArithmeticError
- (Class) AssertionError
- (Class) AttributeError
- (Class) BaseException
- (Class) BaseExceptionGroup
- (Class) BlockingIOError
- (Class) BrokenPipeError
- (Class) BufferError
- (Class) BytesWarning
- (Class) ChildProcessError
- (Class) ConnectionAbortedError
- (Class) ConnectionError
- (Class) ConnectionRefusedError
- (Class) ConnectionResetError
- (Class) DeprecationWarning
- (Class) EOFError
- (Variable) Ellipsis
- (Class) EncodingWarning
- (Variable) EnvironmentError
- (Class) Exception
- (Class) ExceptionGroup
- (Keyword) False
- (Class) FileExistsError
- (Class) FileNotFoundError
- (Class) FloatingPointError
- (Class) FutureWarning
- (Class) GeneratorExit
- (Variable) IOError
- (Class) ImportError
- (Class) ImportWarning
- (Class) IndentationError
- (Class) IndexError
- (Class) InterruptedError
- (Class) IsADirectoryError
- (Class) KeyError
- (Class) KeyboardInterrupt
- (Class) LookupError
- (Class) MemoryError
- (Class) ModuleNotFoundError
- (Class) NameError
- (Keyword) None
- (Class) NotADirectoryError
- (Variable) NotImplemented
- (Class) NotImplementedError
- (Class) OSError
- (Class) OverflowError
- (Class) PendingDeprecationWarning
- (Class) PermissionError
- (Class) ProcessLookupError
- (Class) PythonFinalizationError
- (Class) RecursionError
- (Class) ReferenceError
- (Class) ResourceWarning
- (Class) RuntimeError
- (Class) RuntimeWarning
- (Class) StopAsyncIteration
- (Class) StopIteration
- (Class) SyntaxError
- (Class) SyntaxWarning
- (Class) SystemError
- (Class) SystemExit
- (Class) TabError
- (Class) TimeoutError
- (Keyword) True
- (Class) TypeError
- (Class) UnboundLocalError
- (Class) UnicodeDecodeError
- (Class) UnicodeEncodeError
- (Class) UnicodeError
- (Class) UnicodeTranslateError
- (Class) UnicodeWarning
- (Class) UserWarning
- (Class) ValueError
- (Class) Warning
- (Class) ZeroDivisionError
- (Function) abs
- (Function) aiter
- (Function) all
- (Keyword) and
- (Function) anext
- (Function) any
- (Function) ascii
- (Keyword) assert
- (Keyword) async
- (Keyword) await
- (Function) bin
- (Class) bool
- (Keyword) break
- (Function) breakpoint
- (Class) bytearray
- (Class) bytes
- (Function) callable
- (Keyword) case
- (Function) chr
- (Keyword) class
- (Class) classmethod
- (Function) compile
- (Class) complex
- (Keyword) continue
- (Variable) copyright
- (Variable) credits
- (Keyword) def
- (Keyword) del
- (Function) delattr
- (Class) dict
- (Function) dir
- (Function) divmod
- (Keyword) elif
- (Variable) ellipsis
- (Keyword) else
- (Class) enumerate
- (Function) eval
- (Keyword) except
- (Function) exec
- (Variable) exit
- (Class) filter
- (Keyword) finally
- (Class) float
- (Keyword) for
- (Function) format
- (Keyword) from
- (Class) frozenset
- (Class) function
- (Function) getattr
- (Keyword) global
- (Function) globals
- (Function) hasattr
- (Function) hash
- (Variable) help
- (Function) hex
- (Function) id
- (Keyword) if
- (Keyword) import
- (Keyword) in
- (Function) input
- (Class) int
- (Keyword) is
- (Function) isinstance
- (Function) issubclass
- (Function) iter
- (Keyword) lambda
- (Function) len
- (Variable) license
- (Class) list
- (Function) locals
- (Class) map
- (Keyword) match
- (Function) max
- (Class) memoryview
- (Function) min
- (Function) next
- (Keyword) nonlocal
- (Keyword) not
- (Class) object
- (Function) oct
- (Function) open
- (Keyword) or
- (Function) ord
- (Keyword) pass
- (Function) pow
- (Function) print
- (Class) property
- (Variable) quit
- (Keyword) raise
- (Class) range
- (Function) repr
- (Keyword) return
- (Class) reversed
- (Function) round
- (Class) set
- (Function) setattr
- (Class) slice
- (Function) sorted
- (Class) staticmethod
- (Class) str
- (Function) sum
- (Class) super
- (Function) test: () -> None
- (Keyword) try
- (Class) tuple
- (Keyword) type
- (Function) vars
- (Keyword) while
- (Keyword) with
- (Keyword) yield
- (Class) zip
- (Variable) _AddableT1
- (Variable) _AddableT2
- (Variable) _AwaitableT
- (Variable) _AwaitableT_co
- (Variable) _BaseExceptionT
- (Variable) _BaseExceptionT_co
- (Variable) _ClassInfo
- (Variable) _E_contra
- (Variable) _ExceptionT
- (Variable) _ExceptionT_co
- (Class) _FormatMapMapping
- (Class) _GetItemIterable
- (Variable) _I
- (Variable) _IntegerFormats
- (Variable) _KT
- (Variable) _LiteralInteger
- (Variable) _M_contra
- (Variable) _NegativeInteger
- (Class) _NotImplementedType
- (Variable) _Opener
- (Variable) _P
- (Variable) _PositiveInteger
- (Variable) _R_co
- (Variable) _S
- (Variable) _StartT_co
- (Variable) _StepT_co
- (Variable) _StopT_co
- (Variable) _SupportsAnextT_co
- (Variable) _SupportsNextT_co
- (Class) _SupportsPow2
- (Class) _SupportsPow3
- (Class) _SupportsPow3NoneOnly
- (Class) _SupportsRound1
- (Class) _SupportsRound2
- (Variable) _SupportsSomeKindOfPow
- (Variable) _SupportsSumNoDefaultT
- (Class) _SupportsSumWithNoDefaultGiven
- (Class) _SupportsSynchronousAnext
- (Class) _SupportsWriteAndFlush
- (Variable) _T
- (Variable) _T1
- (Variable) _T2
- (Variable) _T3
- (Variable) _T4
- (Variable) _T5
- (Variable) _T_co
- (Variable) _T_contra
- (Class) _TranslateTable
- (Variable) _VT
- (Constant) __annotations__
- (Function) __build_class__
- (Constant) __builtins__
- (Constant) __cached__
- (Constant) __debug__
- (Constant) __dict__
- (Constant) __doc__
- (Constant) __file__
- (Function) __import__
- (Constant) __loader__
- (Constant) __name__
- (Constant) __package__
- (Constant) __path__
- (Constant) __spec__"#
            .trim(),
        report.trim(),
    );
}

#[test]
fn redeclaration() {
    let code = r#"
fff = 1
fff = fff + 1
ff
#^
"#;
    let report =
        get_batched_lsp_operations_report_allow_error(&[("main", code)], get_default_test_report());
    assert_eq!(
        r#"
# main.py
4 | ff
     ^
Completion Results:
- (Variable) fff: int
"#
        .trim(),
        report.trim(),
    );
}
