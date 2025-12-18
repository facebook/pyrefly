/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use lsp_types::CompletionItem;
use lsp_types::CompletionItemKind;
use pretty_assertions::assert_eq;
use pyrefly_build::handle::Handle;
use ruff_text_size::TextSize;

use crate::state::lsp::ImportFormat;
use crate::state::require::Require;
use crate::state::state::State;
use crate::state::state::Transaction;
use crate::test::util::extract_cursors_for_test;
use crate::test::util::get_batched_lsp_operations_report;
use crate::test::util::get_batched_lsp_operations_report_allow_error;
use crate::test::util::mk_multi_file_state;

// Some assertions compare literal completion dumps. pretty_assertions decorates
// diffs with ANSI escapes, so we strip them to keep the expected strings stable.
fn strip_ansi(input: &str) -> String {
    let mut result = String::with_capacity(input.len());
    let mut chars = input.chars();
    while let Some(ch) = chars.next() {
        if ch == '\u{1b}' {
            for next in chars.by_ref() {
                if next == 'm' {
                    break;
                }
            }
        } else {
            result.push(ch);
        }
    }
    result
}

#[derive(Default)]
struct ResultsFilter {
    include_keywords: bool,
    include_builtins: bool,
}

#[test]
fn completion_magic_methods_in_class() {
    let code = r#"
class Foo:
    def __
#       ^
"#;
    let report =
        get_batched_lsp_operations_report_allow_error(&[("main", code)], get_default_test_report());
    let trimmed = report.trim();
    for expected in [
        "- (Method) __eq__",
        "- (Method) __len__",
        "- (Method) __str__",
        "- (Method) __hash__",
    ] {
        assert!(
            trimmed.contains(expected),
            "missing {expected} in completions:\n{trimmed}"
        );
    }
}

fn get_default_test_report() -> impl Fn(&State, &Handle, TextSize) -> String {
    get_test_report(ResultsFilter::default(), ImportFormat::Absolute)
}

fn get_test_report(
    filter: ResultsFilter,
    import_format: ImportFormat,
) -> impl Fn(&State, &Handle, TextSize) -> String {
    move |state: &State, handle: &Handle, position: TextSize| {
        let mut report = "Completion Results:".to_owned();
        for CompletionItem {
            label,
            detail,
            kind,
            insert_text,
            data,
            tags,
            text_edit,
            documentation,
            ..
        } in state
            .transaction()
            .completion(handle, position, import_format, true)
        {
            let is_deprecated = if let Some(tags) = tags {
                tags.contains(&lsp_types::CompletionItemTag::DEPRECATED)
            } else {
                false
            };
            if (filter.include_keywords || kind != Some(CompletionItemKind::KEYWORD))
                && (filter.include_builtins || data != Some(serde_json::json!("builtin")))
            {
                report.push_str("\n- (");
                report.push_str(&format!("{:?}", kind.unwrap()));
                report.push_str(") ");
                if is_deprecated {
                    report.push_str("[DEPRECATED] ");
                }
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
                if let Some(text_edit) = text_edit {
                    report.push_str(" with text edit: ");
                    report.push_str(&format!("{:?}", &text_edit));
                }
                if let Some(documentation) = documentation {
                    report.push('\n');
                    match documentation {
                        lsp_types::Documentation::String(s) => {
                            report.push_str(&s);
                        }
                        lsp_types::Documentation::MarkupContent(content) => {
                            report.push_str(&content.value);
                        }
                    }
                }
            }
        }
        report
    }
}

fn dict_field_labels(txn: &Transaction<'_>, handle: &Handle, position: TextSize) -> Vec<String> {
    txn.completion(handle, position, ImportFormat::Absolute, true)
        .into_iter()
        .filter(|item| item.kind == Some(CompletionItemKind::FIELD))
        .map(|item| item.label)
        .collect()
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
fn dict_key_completion_from_literal() {
    let code = r#"
def use_dict():
    data = {"foo": 1, "bar": 2}
    data[""]
#        ^
"#;
    let report =
        get_batched_lsp_operations_report_allow_error(&[("main", code)], get_default_test_report());
    let report = strip_ansi(&report);
    assert!(
        report.trim().contains(
            r#"
# main.py
4 |     data[""]
             ^
Completion Results:
- (Field) bar: Literal[2]
- (Field) foo: Literal[1]
"#
            .trim()
        )
    );
}

#[test]
fn dict_key_completion_from_nested_literal() {
    let code = r#"
def nested():
    config = {"user": {"name": "Alice"}}
    config["user"][""]
#                  ^
"#;
    let (handles, state) = mk_multi_file_state(&[("main", code)], Require::indexing(), true);
    let handle = handles.get("main").unwrap();
    let position = extract_cursors_for_test(code)[0];
    let txn = state.transaction();
    let labels = dict_field_labels(&txn, handle, position);
    assert_eq!(labels, vec!["name".to_owned()]);
}

#[test]
fn dict_key_completion_from_outer_literal() {
    let code = r#"
def nested():
    config = {"user": {"name": "Alice"}}
    config["user"][""]
#           ^
"#;
    let (handles, state) = mk_multi_file_state(&[("main", code)], Require::indexing(), true);
    let handle = handles.get("main").unwrap();
    let position = extract_cursors_for_test(code)[0];
    let txn = state.transaction();
    let labels = dict_field_labels(&txn, handle, position);
    assert_eq!(labels, vec!["user".to_owned()]);
}

#[test]
fn dict_key_completion_from_typed_dict() {
    let code = r#"
from typing import TypedDict

class User(TypedDict):
    name: str
    age: int

u: User
u[""]
#  ^
"#;
    let report =
        get_batched_lsp_operations_report_allow_error(&[("main", code)], get_default_test_report());
    let report = strip_ansi(&report);
    assert!(
        report.trim().contains(
            r#"
# main.py
9 | u[""]
       ^
Completion Results:
- (Field) age: int
- (Field) name: str
"#
            .trim()
        )
    );
}

#[test]
fn dot_complete_with_deprecated_method() {
    let code = r#"
from warnings import deprecated
class Foo:
    x: int
    @deprecated("this is not ok")
    def not_ok(self): ...
    @deprecated("this is also not ok")
    @property
    def also_not_ok(self) -> int: ...
foo = Foo()
foo.
#   ^
"#;
    let report =
        get_batched_lsp_operations_report_allow_error(&[("main", code)], get_default_test_report());
    assert_eq!(
        r#"
# main.py
11 | foo.
         ^
Completion Results:
- (Field) x: int
- (Field) [DEPRECATED] also_not_ok: int
- (Method) [DEPRECATED] not_ok: def not_ok(self: Foo) -> None: ...
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn completion_deprecated_top_level_function() {
    let code = r#"
from typing import *
from warnings import deprecated

@deprecated("this is deprecated")
def test1() -> None:
    ...

def test2() -> None:
    ...

te
# ^
"#;
    let report =
        get_batched_lsp_operations_report_allow_error(&[("main", code)], get_default_test_report());
    assert_eq!(
        r#"
# main.py
12 | te
       ^
Completion Results:
- (Class) deprecated: type[deprecated]
- (Function) test2: () -> None
- (Function) [DEPRECATED] test1: () -> None
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn complete_deprecated_class() {
    let code = r#"
from warnings import deprecated
@deprecated("this class is deprecated")
class MyDeprecatedClass: pass
MyDe
#   ^
"#;
    let report =
        get_batched_lsp_operations_report_allow_error(&[("main", code)], get_default_test_report());
    assert_eq!(
        r#"
# main.py
5 | MyDe
        ^
Completion Results:
- (Class) [DEPRECATED] MyDeprecatedClass: type[MyDeprecatedClass]
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
- (Method) class_method: def class_method(cls: type[Foo]) -> None: ...
- (Method) method: def method(self: Foo) -> None: ...
- (Function) static_method: def static_method() -> None: ...
- (Field) x: int
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn dot_complete_ranked_test() {
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
fn variable_complete_with_deprecated_function() {
    let code = r#"
from warnings import deprecated
@deprecated("this is not ok")
def not_ok(): ...
def foo():
  n
# ^
"#;
    let report =
        get_batched_lsp_operations_report_allow_error(&[("main", code)], get_default_test_report());
    assert_eq!(
        r#"
# main.py
6 |   n
      ^
Completion Results:
- (Function) [DEPRECATED] not_ok: () -> None
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
        get_test_report(
            ResultsFilter {
                include_builtins: true,
                ..Default::default()
            },
            ImportFormat::Absolute,
        ),
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
- (Field) magic: Unknown


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
fn from_import_deprecated() {
    let foo_code = r#"
from warnings import deprecated

def func_ok():
    ...
@deprecated("this is not ok")
def func_not_ok():
    ...
"#;
    let main_code = r#"
from foo import func
#          ^        ^
"#;
    let report = get_batched_lsp_operations_report_allow_error(
        &[("main", main_code), ("foo", foo_code)],
        get_default_test_report(),
    );
    assert_eq!(
        r#"
# main.py
2 | from foo import func
               ^
Completion Results:

2 | from foo import func
                        ^
Completion Results:
- (Variable) deprecated
- (Variable) func_ok
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
- (Variable) [DEPRECATED] func_not_ok


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
fn attribute_completion_in_function_call() {
    let code = r#"
class Foo:
    x: int

def test(abcdefg: int) -> None:
    pass

f = Foo()

test(f.
#      ^
"#;
    let report =
        get_batched_lsp_operations_report_allow_error(&[("main", code)], get_default_test_report());
    assert_eq!(
        r#"
# main.py
10 | test(f.
            ^
Completion Results:
- (Field) x: int
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
        get_test_report(
            ResultsFilter {
                include_builtins: true,
                ..Default::default()
            },
            ImportFormat::Absolute,
        ),
    );
    assert_eq!(
        r#"
# main.py
2 | isins
         ^
Completion Results:
- (Function) isinstance
- (Class) DivisionImpossible: from decimal import DivisionImpossible

- (Class) FirstHeaderLineIsContinuationDefect: from email.errors import FirstHeaderLineIsContinuationDefect

- (Class) MissingHeaderBodySeparatorDefect: from email.errors import MissingHeaderBodySeparatorDefect

- (Function) disjoint_base: from typing_extensions import disjoint_base

- (Function) distributions: from importlib.metadata import distributions

- (Function) fix_missing_locations: from ast import fix_missing_locations

- (Function) packages_distributions: from importlib.metadata import packages_distributions

- (Function) timerfd_settime_ns: from os import timerfd_settime_ns

- (Module) typing_extensions: import typing_extensions
"#
        .trim(),
        report.trim(),
    );
}

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
- (Value) 'foo': Literal['foo'] inserting `'foo'`
- (Variable) x=: Literal['foo']
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn completion_literal_with_escape_chars() {
    let code = r#"
from typing import Literal
def foo(x: Literal['\a', '\b', '\f', '\n', '\r', '\t', '\v', '\\', '"', "'"]): ...
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
- (Value) '"': Literal['"'] inserting `'"'`
- (Value) '\'': Literal['\''] inserting `'\''`
- (Value) '\\': Literal['\\'] inserting `'\\'`
- (Value) '\a': Literal['\a'] inserting `'\a'`
- (Value) '\b': Literal['\b'] inserting `'\b'`
- (Value) '\f': Literal['\f'] inserting `'\f'`
- (Value) '\n': Literal['\n'] inserting `'\n'`
- (Value) '\r': Literal['\r'] inserting `'\r'`
- (Value) '\t': Literal['\t'] inserting `'\t'`
- (Value) '\v': Literal['\v'] inserting `'\v'`
- (Variable) x=: Literal['\a', '\b', '\t', '\n', '\v', '\f', '\r', '"', '\'', '\\']
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn completion_literal_with_escape_chars_inside() {
    let code = r#"
from typing import Literal
def foo(x: Literal["a\nb"]): ...
foo("
#    ^
"#;
    let report =
        get_batched_lsp_operations_report_allow_error(&[("main", code)], get_default_test_report());
    assert_eq!(
        r#"
# main.py
4 | foo("
         ^
Completion Results:
- (Value) 'a\nb': Literal['a\nb'] inserting `a
b`"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn completion_literal_union() {
    let code = r#"
from typing import Literal, Union
def foo(x: Union[Literal['foo'] | Literal['bar']]): ...
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
- (Value) 'bar': Literal['bar'] inserting `'bar'`
- (Value) 'foo': Literal['foo'] inserting `'foo'`
- (Variable) x=: Literal['bar', 'foo']
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn completion_literal_union_being_typed() {
    let code = r#"
from typing import Literal, Union
def foo(x: Union[Literal['foo'] | Literal['bar']]): ...
foo('
#    ^
"#;
    let report =
        get_batched_lsp_operations_report_allow_error(&[("main", code)], get_default_test_report());
    assert_eq!(
        r#"
# main.py
4 | foo('
         ^
Completion Results:
- (Value) 'bar': Literal['bar'] inserting `bar`
- (Value) 'foo': Literal['foo'] inserting `foo`
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn completion_literal_union_alias() {
    let code = r#"
from typing import Literal, Union
MyUnion = Union[Literal['foo'], Literal['bar']]
def foo(x: MyUnion): ...
foo(
#   ^

"#;
    let report =
        get_batched_lsp_operations_report_allow_error(&[("main", code)], get_default_test_report());
    assert_eq!(
        r#"
# main.py
5 | foo(
        ^
Completion Results:
- (Value) 'bar': Literal['bar'] inserting `'bar'`
- (Value) 'foo': Literal['foo'] inserting `'foo'`
- (Variable) x=: Literal['bar', 'foo']
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn completion_literal_union_multiple_types() {
    let code = r#"
from typing import Literal, Union, LiteralString
def foo(x: Union[Literal['foo'] | Literal[1] | LiteralString]): ...
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
- (Value) 1: Literal[1] inserting `1`
- (Variable) x=: Literal[1] | LiteralString
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn completion_literal_nested() {
    let code = r#"
from typing import Literal, Union
class Foo: ...
def foo(x: Union[Union[Literal['foo']] | Literal[1] | Foo]): ...
foo(
#   ^
"#;
    let report =
        get_batched_lsp_operations_report_allow_error(&[("main", code)], get_default_test_report());
    assert_eq!(
        r#"
# main.py
5 | foo(
        ^
Completion Results:
- (Value) 'foo': Literal['foo'] inserting `'foo'`
- (Value) 1: Literal[1] inserting `1`
- (Variable) x=: Literal['foo', 1] | Foo
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn completion_literal_do_not_duplicate_quotes() {
    let code = r#"
from typing import Literal, Union
class Foo: ...
def foo(x: Union[Union[Literal['foo']] | Literal[1] | Foo]): ...
foo(''
#    ^
"#;
    let report =
        get_batched_lsp_operations_report_allow_error(&[("main", code)], get_default_test_report());
    assert_eq!(
        r#"
# main.py
5 | foo(''
         ^
Completion Results:
- (Value) 'foo': Literal['foo'] inserting `foo`
- (Value) 1: Literal[1] inserting `1`
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
x = {"a": 3, "b": 4}
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
        get_test_report(
            ResultsFilter {
                include_keywords: true,
                ..Default::default()
            },
            ImportFormat::Absolute,
        ),
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
fn dot_compete_union() {
    let code = r#"
class A:
    x: int
    y: int

class B:
    y: str
    z: str

def foo(x: A | B) -> None:
    x.
#     ^
"#;
    let report =
        get_batched_lsp_operations_report_allow_error(&[("main", code)], get_default_test_report());
    assert_eq!(
        r#"
# main.py
11 |     x.
           ^
Completion Results:
- (Field) x: int
- (Field) y: int | str
- (Field) z: str
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn dot_compete_override() {
    let code = r#"
class A:
    def foo(self) -> int | str: ...

class B(A):
    def foo(self) -> int: ...

def foo(x: B) -> None:
    x.
#     ^
"#;
    let report =
        get_batched_lsp_operations_report_allow_error(&[("main", code)], get_default_test_report());
    assert_eq!(
        r#"
# main.py
9 |     x.
          ^
Completion Results:
- (Method) foo: def foo(self: B) -> int: ...
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn dot_complete_super() {
    let code = r#"
class A:
    x: int

class B:
    y: str

class C(A, B):
    def foo(self):
        super().
#               ^
"#;
    let report =
        get_batched_lsp_operations_report_allow_error(&[("main", code)], get_default_test_report());
    assert_eq!(
        r#"
# main.py
10 |         super().
                     ^
Completion Results:
- (Field) x: int
- (Field) y: str
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
        get_test_report(
            ResultsFilter {
                include_keywords: true,
                ..Default::default()
            },
            ImportFormat::Absolute,
        ),
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
fn autoimport_relative_on_builtins() {
    let code = r#"
T = foooooo
#       ^
"#;
    let report = get_batched_lsp_operations_report_allow_error(
        &[("main", code), ("bar", "foooooo = 1")],
        get_test_report(Default::default(), ImportFormat::Relative),
    );
    assert_eq!(
        r#"
# main.py
2 | T = foooooo
            ^
Completion Results:
- (Variable) foooooo: from .bar import foooooo



# bar.py
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn autoimport_relative_on_deprecated() {
    let code = r#"
T = foooooo
#       ^
"#;
    let bar_code = r#"
from warnings import deprecated
@deprecated("this is not ok")
def foooooo():
    ...
"#;
    let report = get_batched_lsp_operations_report_allow_error(
        &[("main", code), ("bar", bar_code)],
        get_test_report(Default::default(), ImportFormat::Relative),
    );
    assert_eq!(
        r#"
# main.py
2 | T = foooooo
            ^
Completion Results:
- (Function) [DEPRECATED] foooooo: from .bar import foooooo



# bar.py
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn autoimport_completions_on_builtins() {
    let code = r#"
T = Literal
#       ^
"#;
    let report = get_batched_lsp_operations_report_allow_error(
        &[("main", code)],
        get_test_report(Default::default(), ImportFormat::Relative),
    );
    assert_eq!(
        r#"
# main.py
2 | T = Literal
            ^
Completion Results:
- (Variable) AnyOrLiteralStr: from _typeshed import AnyOrLiteralStr

- (Variable) Literal: from typing import Literal

- (Variable) Literal: from typing_extensions import Literal

- (Variable) LiteralString: from typing import LiteralString

- (Variable) StrOrLiteralStr: from _typeshed import StrOrLiteralStr
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn autoimport_completions_set_label_details() {
    let code = r#"
T = foooooo
#       ^
"#;
    let files = [("main", code), ("bar", "foooooo = 1")];
    let (handles, state) = mk_multi_file_state(&files, Require::indexing(), false);
    let handle = handles.get("main").unwrap();
    let position = extract_cursors_for_test(code)[0];

    // Label details supported
    let completions =
        state
            .transaction()
            .completion(handle, position, ImportFormat::Absolute, true);
    let autoimport = completions
        .into_iter()
        .find(|item| item.label == "foooooo")
        .expect("expected foooooo to be in completions");
    let label_details = autoimport
        .label_details
        .expect("auto import completion should include label details");
    assert_eq!(label_details.detail.as_deref(), Some(" (import bar)"));
    assert_eq!(label_details.description.as_deref(), Some("bar"));

    // Label details unsupported
    let completions =
        state
            .transaction()
            .completion(handle, position, ImportFormat::Absolute, false);
    completions
        .into_iter()
        .find(|item| item.label == "foooooo")
        .expect("expected foooooo to be in completions");
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
        get_test_report(
            ResultsFilter {
                include_keywords: true,
                include_builtins: true,
            },
            ImportFormat::Absolute,
        ),
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

#[test]
fn from_import_keyword_completion() {
    let code = r#"
from lib imp
#          ^
"#;

    let lib = r#"
class Foo: pass
"#;

    let report = get_batched_lsp_operations_report_allow_error(
        &[("main", code), ("lib", lib)],
        get_test_report(
            ResultsFilter {
                include_keywords: true,
                ..Default::default()
            },
            ImportFormat::Absolute,
        ),
    );
    assert_eq!(
        r#"
# main.py
2 | from lib imp
               ^
Completion Results:
- (Variable) Foo
- (Keyword) import
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


# lib.py
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn kwarg_completion_in_str() {
    let code = r#"
def foo(x: str): ...
foo(x="x")
#      ^"#;
    let report = get_batched_lsp_operations_report(
        &[("main", code)],
        get_test_report(
            ResultsFilter {
                include_keywords: true,
                ..Default::default()
            },
            ImportFormat::Absolute,
        ),
    );
    assert_eq!(
        r#"
# main.py
3 | foo(x="x")
           ^
Completion Results:
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn completion_with_imported_class_docstring() {
    let lib = r#"
class Foo:
    """This is a Foo class with useful functionality."""
    pass
"#;
    let main = r#"
from lib import Foo
F
#^
"#;
    let report = get_batched_lsp_operations_report_allow_error(
        &[("main", main), ("lib", lib)],
        get_default_test_report(),
    );
    assert_eq!(
        r#"
# main.py
3 | F
     ^
Completion Results:
- (Class) Foo: type[Foo]
This is a Foo class with useful functionality.


# lib.py
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn completion_with_imported_function_docstring() {
    let lib = r#"
def bar():
    """This function does something useful."""
    pass
"#;
    let main = r#"
from lib import bar
b
#^
"#;
    let report = get_batched_lsp_operations_report_allow_error(
        &[("main", main), ("lib", lib)],
        get_default_test_report(),
    );
    assert_eq!(
        r#"
# main.py
3 | b
     ^
Completion Results:
- (Function) bar: () -> None
This function does something useful.


# lib.py
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn completion_with_local_class_docstring() {
    let code = r#"
class Foo:
    """This is a local Foo class with useful functionality."""
    pass
F
#^
"#;
    let report =
        get_batched_lsp_operations_report_allow_error(&[("main", code)], get_default_test_report());
    assert_eq!(
        r#"
# main.py
5 | F
     ^
Completion Results:
- (Class) Foo: type[Foo]
This is a local Foo class with useful functionality.
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn completion_with_local_function_docstring() {
    let code = r#"
def bar():
    """This function does something useful."""
    pass
b
#^
"#;
    let report =
        get_batched_lsp_operations_report_allow_error(&[("main", code)], get_default_test_report());
    assert_eq!(
        r#"
# main.py
5 | b
     ^
Completion Results:
- (Function) bar: () -> None
This function does something useful.
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn dot_complete_with_method_docstring() {
    let code = r#"
class Foo:
    def method(self) -> int:
        """This is a method docstring."""
        return 42

f = Foo()
f.
# ^
"#;
    let report =
        get_batched_lsp_operations_report_allow_error(&[("main", code)], get_default_test_report());
    assert_eq!(
        r#"
# main.py
8 | f.
      ^
Completion Results:
- (Method) method: def method(self: Foo) -> int: ...
This is a method docstring.
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn dot_complete_with_multiple_methods_docstring() {
    let code = r#"
class Foo:
    def first(self) -> int:
        """First method documentation."""
        return 1

    def second(self) -> str:
        """Second method documentation."""
        return "hello"

f = Foo()
f.
# ^
"#;
    let report =
        get_batched_lsp_operations_report_allow_error(&[("main", code)], get_default_test_report());
    assert_eq!(
        r#"
# main.py
12 | f.
       ^
Completion Results:
- (Method) first: def first(self: Foo) -> int: ...
First method documentation.
- (Method) second: def second(self: Foo) -> str: ...
Second method documentation.
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn dot_complete_with_property_docstring() {
    let code = r#"
class Foo:
    @property
    def value(self) -> int:
        """Property with documentation."""
        return 42

f = Foo()
f.
# ^
"#;
    let report =
        get_batched_lsp_operations_report_allow_error(&[("main", code)], get_default_test_report());
    assert_eq!(
        r#"
# main.py
9 | f.
      ^
Completion Results:
- (Field) value: int
Property with documentation.
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn dot_complete_mixed_with_and_without_docstring() {
    let code = r#"
class Foo:
    x: int

    def documented(self) -> str:
        """This has documentation."""
        return "doc"

    def undocumented(self) -> int:
        return 123

f = Foo()
f.
# ^
"#;
    let report =
        get_batched_lsp_operations_report_allow_error(&[("main", code)], get_default_test_report());
    assert_eq!(
        r#"
# main.py
13 | f.
       ^
Completion Results:
- (Method) documented: def documented(self: Foo) -> str: ...
This has documentation.
- (Method) undocumented: def undocumented(self: Foo) -> int: ...
- (Field) x: int
"#
        .trim(),
        report.trim(),
    );
}

// Regression test for https://github.com/facebook/pyrefly/issues/1257
// Because the base type for completion is passed to Type::for_display,
// which converts all unsolved Var to Var::ZERO, we were running into an
// unexpected Var::ZERO in attribute lookup, leading to a panic.
#[test]
fn dot_complete_var_crash_regression() {
    let code = r#"
class C[T]:
    def m(self) -> None: ...

    @property
    def p(self) -> T: ...

def f[T]() -> C[T]: ...
f().
#   ^
"#;
    let report =
        get_batched_lsp_operations_report_allow_error(&[("main", code)], get_default_test_report());
    assert_eq!(
        r#"
# main.py
9 | f().
        ^
Completion Results:
- (Method) m: def m(self: C[Unknown]) -> None: ...
- (Field) p: Unknown
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn test_no_completion_in_comments() {
    let code = r#"
# This is a comment
#      ^
import sys
x = sys.version
#       ^
"#;
    let (handles, state) = mk_multi_file_state(&[("main", code)], Require::indexing(), true);
    let handle = handles.get("main").unwrap();
    let positions = extract_cursors_for_test(code);
    let txn = state.transaction();

    // Position 0: inside comment - should return empty
    let comment_completions = txn.completion(handle, positions[0], ImportFormat::Absolute, true);
    assert!(
        comment_completions.is_empty(),
        "Expected no completions in comment, but got {} completions",
        comment_completions.len()
    );

    // Position 1: normal code - should return completions
    let normal_completions = txn.completion(handle, positions[1], ImportFormat::Absolute, true);
    assert!(
        !normal_completions.is_empty(),
        "Expected completions in normal code but got none"
    );
}

#[test]
fn bound_method_completions_include_descriptor_attributes() {
    // Make sure completions work for bound methods from custom descriptors.
    // See: https://github.com/facebook/pyrefly/issues/821
    let code = r#"
from __future__ import annotations
from typing import Callable

class CachedMethod:
    def __init__(self, fn: Callable[[Constraint], int]) -> None:
        self._fn = fn

    def __get__(self, obj: Constraint | None, owner: type[Constraint]) -> CachedMethod:
        return self

    def __call__(self, obj: Constraint) -> int:
        return self._fn(obj)

    def clear_cache(self, obj: Constraint) -> None: ...

def cache_on_self(fn: Callable[[Constraint], int]) -> CachedMethod:
    return CachedMethod(fn)

class Constraint:
    @cache_on_self
    def pointwise_read_writes(self) -> int:
        return 0

    def use_cached_method(self) -> None:
        self.pointwise_read_writes.
#                                  ^
"#;
    let (handles, state) = mk_multi_file_state(&[("main", code)], Require::indexing(), false);
    let handle = handles.get("main").unwrap();
    let position = extract_cursors_for_test(code)[0];
    let txn = state.transaction();
    let completions = txn.completion(handle, position, ImportFormat::Absolute, true);

    let completion_labels: Vec<_> = completions.iter().map(|c| c.label.as_str()).collect();

    // The descriptor's `clear_cache` method should appear in completions
    assert!(
        completion_labels.contains(&"clear_cache"),
        "Expected `clear_cache` in completions for bound method from custom descriptor.\n\
         This attribute is defined on CachedMethod and should be accessible.\n\
         Got completions: {:?}",
        completion_labels
    );
}
