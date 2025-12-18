/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use lsp_types::Hover;
use lsp_types::HoverContents;
use pretty_assertions::assert_eq;
use pyrefly_build::handle::Handle;
use ruff_text_size::TextSize;

use crate::lsp::wasm::hover::get_hover;
use crate::state::state::State;
use crate::test::util::get_batched_lsp_operations_report;

fn get_test_report(state: &State, handle: &Handle, position: TextSize) -> String {
    match get_hover(&state.transaction(), handle, position, true) {
        Some(Hover {
            contents: HoverContents::Markup(markup),
            ..
        }) => markup.value,
        _ => "None".to_owned(),
    }
}

#[test]
fn bound_methods_test() {
    let code = r#"
class Foo:
   def meth(self):
        pass

foo = Foo()
foo.meth()
#   ^
xyz = [foo.meth]
#^
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert!(report.contains("(method) meth: def meth(self: Foo) -> None: ..."));
    assert!(report.contains("(variable) xyz: list[(self: Foo) -> None]"));
    assert!(
        report.contains("Go to [list]"),
        "Expected 'Go to [list]' link, got: {}",
        report
    );
    assert!(
        report.contains("builtins.pyi"),
        "Expected link to builtins.pyi, got: {}",
        report
    );
}

#[test]
fn renamed_reexport_shows_original_name() {
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
```python
(function) foo: def foo() -> None: ...
```


# lib.py

# lib2.py
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn hover_shows_unpacked_kwargs_fields() {
    let code = r#"
from typing import TypedDict, Unpack

class Payload(TypedDict):
    foo: int
    bar: str
    baz: bool | None

def takes(**kwargs: Unpack[Payload]) -> None:
    ...

takes(foo=1, bar="x", baz=None)
#^
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
12 | takes(foo=1, bar="x", baz=None)
      ^
```python
(function) takes: def takes(
    *,
    foo: int,
    bar: str,
    baz: bool | None,
    **kwargs: Unpack[Payload]
) -> None: ...
```
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn hover_on_callable_instance_uses_dunder_call_signature() {
    let code = r#"
class Greeter:
    def __call__(self, name: str, repeat: int = 1) -> str: ...

greeter = Greeter()
greeter("hi")
#^
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert!(
        report.contains("__call__"),
        "Expected hover to refer to __call__, got: {report}"
    );
    assert!(
        report.contains("name: str"),
        "Expected hover to show parameter 'name', got: {report}"
    );
    assert!(
        report.contains("repeat: int = 1"),
        "Expected hover to show optional parameter, got: {report}"
    );
}

#[test]
fn hover_over_inline_ignore_comment() {
    let code = r#"
a: int = "test"  # pyrefly: ignore
#                                ^
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
2 | a: int = "test"  # pyrefly: ignore
                                     ^
**Suppressed Error**

`bad-assignment`: `Literal['test']` is not assignable to `int`
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn hover_over_ignore_on_function_call() {
    let code = r#"
def foo(x: str) -> None:
    pass

x: int = foo("hello")  # pyrefly: ignore
#                                     ^
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    // Should show the suppressed error from function call assignment
    assert!(report.contains("**Suppressed Error"));
    assert!(report.contains("`bad-assignment`"));
}

#[test]
fn hover_over_generic_type_ignore() {
    let code = r#"
a: int = "test"  # type: ignore
#                            ^
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
2 | a: int = "test"  # type: ignore
                                 ^
**Suppressed Error**

`bad-assignment`: `Literal['test']` is not assignable to `int`
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn hover_over_string_with_hash_character() {
    let code = r#"
x = "hello # world"  # pyrefly: ignore
#                                    ^
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    // The # inside the string should be ignored, only the comment # matters
    // Since there's no error on this line, should show "No errors suppressed"
    assert!(report.contains("No errors suppressed"));
}

#[test]
fn hover_over_ignore_with_no_actual_errors() {
    let code = r#"
x: int = 5  # pyrefly: ignore[bad-return]
#                                       ^
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert!(report.contains("No errors suppressed"));
}

#[test]
fn hover_shows_parameter_doc_for_keyword_argument() {
    let code = r#"
def foo(x: int, y: int) -> None:
    """
    Args:
        x: documentation for x
        y: documentation for y
    """
    ...

foo(x=1, y=2)
#   ^
foo(x=1, y=2)
#        ^
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert!(
        report.contains("**Parameter `x`**"),
        "Expected parameter documentation for x, got: {report}"
    );
    assert!(report.contains("documentation for x"));
    assert!(report.contains("**Parameter `y`**"));
    assert!(report.contains("documentation for y"));
}

#[test]
fn hover_returns_none_for_docstring_literals() {
    let code = r#"
def foo():
    """Function docstring."""
#    ^
    return 1
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
3 |     """Function docstring."""
         ^
None
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn hover_shows_parameter_doc_with_multiline_description() {
    let code = r#"
def foo(param: int) -> None:
    """
    Args:
        param: This is a long parameter description
            that spans multiple lines
            with detailed information
    """
    ...

foo(param=1)
#   ^
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert!(report.contains("**Parameter `param`**"));
    assert!(report.contains("This is a long parameter description"));
    assert!(report.contains("that spans multiple lines"));
    assert!(report.contains("with detailed information"));
}

#[test]
fn hover_on_parameter_definition_shows_doc() {
    let code = r#"
def foo(param: int) -> None:
    """
    Args:
        param: documentation for param
    """
    print(param)
#         ^
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert!(
        report.contains("**Parameter `param`**"),
        "Expected parameter doc when hovering on parameter usage, got: {report}"
    );
    assert!(report.contains("documentation for param"));
}

#[test]
fn hover_parameter_doc_with_type_annotations_in_docstring() {
    let code = r#"
def foo(x, y):
    """
    Args:
        x (int): an integer parameter
        y (str): a string parameter
    """
    ...

foo(x=1, y="hello")
#   ^
foo(x=1, y="hello")
#        ^
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert!(report.contains("**Parameter `x`**"));
    assert!(report.contains("an integer parameter"));
    assert!(report.contains("**Parameter `y`**"));
    assert!(report.contains("a string parameter"));
}

#[test]
fn hover_shows_docstring_for_dataclass_field() {
    let code = r#"
from dataclasses import dataclass

@dataclass
class Widget:
    name: str
    """Name of the widget."""
    box: str
    """The box containing the widget."""

widget = Widget("foo", "bar")
widget.box
#      ^
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert!(
        report.contains("The box containing the widget."),
        "Expected dataclass field docstring to appear in hover, got: {report}"
    );
}

#[test]
fn hover_parameter_doc_with_complex_types() {
    let code = r#"
from typing import Optional, List, Dict

def foo(data: Optional[List[Dict[str, int]]]) -> None:
    """
    Args:
        data: complex nested type parameter
    """
    ...

foo(data=[])
#   ^
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert!(report.contains("**Parameter `data`**"));
    assert!(report.contains("complex nested type parameter"));
}

#[test]
fn hover_over_overloaded_binary_operator_shows_dunder_name() {
    let code = r#"
from typing import overload

class Matrix:
    @overload
    def __matmul__(self, other: Matrix) -> Matrix: ...
    @overload
    def __matmul__(self, other: int) -> Matrix: ...
    def __matmul__(self, other) -> Matrix: ...

lhs = Matrix()
rhs = Matrix()
lhs @ rhs
#   ^
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
13 | lhs @ rhs
         ^
```python
(method) __matmul__: def __matmul__(
    self: Matrix,
    other: Matrix
) -> Matrix: ...
```
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn hover_over_getitem_operator_shows_dunder_name() {
    let code = r#"
class Container:
    def __getitem__(self, idx: int) -> int: ...

c = Container()
c [0]
# ^
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert!(
        report.contains("6 | c [0]"),
        "Expected code frame to include subscript line, got: {report}"
    );
    assert!(
        report.contains("\n      ^\n```python"),
        "Expected caret to precede hover block, got: {report}"
    );
    assert!(
        report.contains(
            "```python\n(method) __getitem__: def __getitem__(\n    self: Container,\n    idx: int\n) -> int: ...\n```"
        ),
        "Expected __getitem__ signature in hover, got: {report}"
    );
}

#[test]
fn hover_over_setitem_operator_shows_dunder_name() {
    let code = r#"
class Container:
    def __setitem__(self, idx: int, value: str) -> None: ...

c = Container()
c [0] = "foo"
# ^
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert!(
        report.contains("6 | c [0] = \"foo\""),
        "Expected code frame to include assignment subscript, got: {report}"
    );
    assert!(
        report.contains(
            "```python\n(method) __setitem__: def __setitem__(\n    self: Container,\n    idx: int,\n    value: str\n) -> None: ...\n```"
        ),
        "Expected __setitem__ signature in hover, got: {report}"
    );
}

#[test]
fn hover_over_delitem_operator_shows_dunder_name() {
    let code = r#"
class Container:
    def __delitem__(self, idx: int) -> None: ...

c = Container()
del c [0]
#     ^
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert!(
        report.contains("6 | del c [0]"),
        "Expected code frame to include delete subscript, got: {report}"
    );
    assert!(
        report.contains(
            "```python\n(method) __delitem__: def __delitem__(\n    self: Container,\n    idx: int\n) -> None: ...\n```"
        ),
        "Expected __delitem__ signature in hover, got: {report}"
    );
}

#[test]
fn hover_over_getitem_without_space_doesnt_show_signature() {
    let code = r#"
class Container:
    def __getitem__(self, idx: int) -> int: ...

c = Container()
c[0]
#^ ^
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
6 | c[0]
     ^
```python
(variable) c: Container
```

6 | c[0]
       ^
```python
(attribute) __getitem__: Literal[0]
```
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn hover_over_binding_in_brackets_without_space_works() {
    let code = r#"
class Container:
    def __getitem__(self, idx: int) -> int: ...

idx_var = 0
c = Container()
c[idx_var]
#  ^
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
7 | c[idx_var]
       ^
```python
(variable) idx_var: Literal[0]
```
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn hover_over_code_with_ignore_shows_type() {
    let code = r#"
a: int = "test"  # pyrefly: ignore
#^
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    // Should show the type of 'a', not the suppressed error
    assert!(
        report.contains("int"),
        "Expected type hover, got: {}",
        report
    );
    assert!(
        !report.contains("Suppressed"),
        "Should not show suppressed error when hovering over code"
    );
}

#[test]
fn builtin_types_have_definition_links() {
    let code = r#"
x: str = "hello"
#^
y: int = 42
#^
z: list[int] = []
#^
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert!(
        report.contains("Go to [str]"),
        "Expected 'Go to [str]' link for str type, got: {}",
        report
    );
    assert!(
        report.contains("Go to [int]"),
        "Expected 'Go to [int]' link for int type, got: {}",
        report
    );
    assert!(
        report.contains("Go to") && report.contains("[list]"),
        "Expected 'Go to' link with [list] for list type, got: {}",
        report
    );

    assert!(
        report.contains("builtins.pyi"),
        "Expected links to builtins.pyi, got: {}",
        report
    );
}

#[test]
fn constant_kind_for_caps_test() {
    let code = r#"
XYZ = 5
# ^
xyz = 5
# ^
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
2 | XYZ = 5
      ^
```python
(constant) XYZ: Literal[5]
```

4 | xyz = 5
      ^
```python
(variable) xyz: Literal[5]
```
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn hover_on_callable_instance_attribute_access() {
    let code = r#"
class Greeter:
    attr: int = 1
    def __call__(self, name: str) -> str: ...

greeter = Greeter()
greeter.attr
#^^^^^^
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    // Should show Greeter type (variable), NOT the __call__ signature
    // The cursor is on 'greeter', usage is attribute access, not call.
    // However, get_hover usually tries to resolve the expression.
    // If we hover on 'greeter' in 'greeter.attr', we expect 'Greeter'.
    // If we hover on 'attr', we expect 'int'.
    // The test framework extracts the range marked by ^.
    assert!(
        report.contains("variable") || report.contains("parameter") || report.contains("Greeter")
    );
    assert!(!report.contains("__call__"));
}
