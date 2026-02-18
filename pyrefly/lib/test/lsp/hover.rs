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
use crate::test::util::get_batched_lsp_operations_report_allow_error;

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
fn hover_on_module_function_shows_function() {
    let lib = r#"
def foo() -> None: ...
"#;
    let code = r#"
import lib

lib.foo()
#    ^
"#;
    let report =
        get_batched_lsp_operations_report(&[("main", code), ("lib", lib)], get_test_report);
    assert!(
        report.contains("(function) foo"),
        "Expected function label, got: {report}"
    );
    assert!(
        !report.contains("(method) foo"),
        "Did not expect method label, got: {report}"
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
fn hover_shows_type_sources_for_narrow_and_first_use() {
    let code = r#"
def f(x: int | None) -> None:
    if x is None:
        return
    y = []
    y.append(1)
    x
#   ^
    y
#   ^
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    let report = normalize_typeshed_paths(&report);
    assert_eq!(
        r#"
# main.py
7 |     x
        ^
```python
(parameter) x: int
```
---
**Type source**
- Narrowed by condition at 3:13: `x is not None`


Go to [int](file:///tmp/pyrefly_bundled_typeshed/builtins.pyi#L420,7)

9 |     y
        ^
```python
(variable) y: list[int]
```
---
**Type source**
- Inferred from first use at 6:5: `y.append(1)`


Go to [int](file:///tmp/pyrefly_bundled_typeshed/builtins.pyi#L420,7) | [list](file:///tmp/pyrefly_bundled_typeshed/builtins.pyi#L3351,7)
"#
        .trim(),
        report.trim(),
    );
}

fn normalize_typeshed_paths(report: &str) -> String {
    const PREFIX: &str = "file:///tmp/pyrefly_bundled_typeshed_";
    let mut normalized = String::with_capacity(report.len());
    let mut rest = report;
    while let Some(idx) = rest.find(PREFIX) {
        normalized.push_str(&rest[..idx]);
        rest = &rest[idx + PREFIX.len()..];
        let Some(slash_idx) = rest.find('/') else {
            normalized.push_str(PREFIX);
            normalized.push_str(rest);
            return normalized;
        };
        normalized.push_str("file:///tmp/pyrefly_bundled_typeshed");
        rest = &rest[slash_idx..];
    }
    normalized.push_str(rest);
    normalized
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

#[test]
fn hover_on_import_same_name_alias_first_token_test() {
    let lib = r#"
def func() -> None: ...
"#;
    let code = r#"
from lib import func as func
#                ^
"#;
    let report =
        get_batched_lsp_operations_report(&[("main", code), ("lib", lib)], get_test_report);
    assert_eq!(
        r#"
# main.py
2 | from lib import func as func
                     ^
```python
(function) func: def func() -> None: ...
```


# lib.py
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn hover_on_import_same_name_alias_second_token_test() {
    let lib = r#"
def func() -> None: ...
"#;
    let code = r#"
from lib import func as func
#                        ^
"#;
    let report =
        get_batched_lsp_operations_report(&[("main", code), ("lib", lib)], get_test_report);
    assert_eq!(
        r#"
# main.py
2 | from lib import func as func
                             ^
```python
(function) func: def func() -> None: ...
```


# lib.py
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn hover_on_import_different_name_alias_first_token_test() {
    let lib = r#"
def bar() -> None: ...
"#;
    let code = r#"
from lib import bar as baz
#                ^
"#;
    let report =
        get_batched_lsp_operations_report(&[("main", code), ("lib", lib)], get_test_report);
    assert_eq!(
        r#"
# main.py
2 | from lib import bar as baz
                     ^
```python
(function) bar: def bar() -> None: ...
```


# lib.py
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn hover_on_import_different_name_alias_second_token_test() {
    let lib = r#"
def bar() -> None: ...
"#;
    let code = r#"
from lib import bar as baz
#                       ^
"#;
    let report =
        get_batched_lsp_operations_report(&[("main", code), ("lib", lib)], get_test_report);
    assert_eq!(
        r#"
# main.py
2 | from lib import bar as baz
                            ^
```python
(function) bar: def bar() -> None: ...
```


# lib.py
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn hover_on_first_component_of_multi_part_import() {
    let mymod_init = r#"# mymod/__init__.py
def version() -> str: ...
"#;
    let mymod_submod_init = r#"# mymod/submod/__init__.py
class Foo: ...
"#;
    let code = r#"
import mymod.submod
#       ^
"#;
    let report = get_batched_lsp_operations_report(
        &[
            ("main", code),
            ("mymod", mymod_init),
            ("mymod.submod", mymod_submod_init),
        ],
        get_test_report,
    );
    assert!(
        report.contains("(module) mymod:"),
        "Expected hover to show 'mymod', got: {report}"
    );
    assert!(
        !report.contains("(module) mymod.submod:"),
        "Hover should not show 'mymod.submod' when hovering over 'mymod', got: {report}"
    );
}

#[test]
fn hover_on_middle_component_of_multi_part_import() {
    let mymod_init = r#"# mymod/__init__.py
def version() -> str: ...
"#;
    let mymod_submod_init = r#"# mymod/submod/__init__.py
class Foo: ...
"#;
    let mymod_submod_deep_init = r#"# mymod/submod/deep/__init__.py
class Bar: ...
"#;
    let code = r#"
from mymod.submod.deep import Bar
#            ^
"#;
    let report = get_batched_lsp_operations_report(
        &[
            ("main", code),
            ("mymod", mymod_init),
            ("mymod.submod", mymod_submod_init),
            ("mymod.submod.deep", mymod_submod_deep_init),
        ],
        get_test_report,
    );
    assert!(
        report.contains("(module) mymod.submod:"),
        "Expected hover to show 'mymod.submod', got: {report}"
    );
    assert!(
        !report.contains("(module) mymod.submod.deep:"),
        "Hover should not show 'mymod.submod.deep' when hovering over 'submod', got: {report}"
    );
}

#[test]
fn hover_on_first_component_when_intermediate_module_missing() {
    // Only mymod.submod exists, not mymod itself
    let mymod_submod_init = r#"# mymod/submod/__init__.py
class Foo: ...
"#;
    let code = r#"
import mymod.submod
#       ^
"#;
    let report = get_batched_lsp_operations_report(
        &[("main", code), ("mymod.submod", mymod_submod_init)],
        get_test_report,
    );
    // When clicking on 'mymod' in 'mymod.submod', hover shows the full identifier
    // 'mymod.submod' even though mymod itself doesn't exist
    assert!(
        report.contains("mymod.submod: Module[mymod]"),
        "Expected hover to show full module name 'mymod.submod', got: {report}"
    );
}

#[test]
fn hover_on_middle_component_when_intermediate_module_missing() {
    // Only mymod and mymod.submod.deep exist, not mymod.submod
    let mymod_init = r#"# mymod/__init__.py
def version() -> str: ...
"#;
    let mymod_submod_deep_init = r#"# mymod/submod/deep/__init__.py
class Bar: ...
"#;
    let code = r#"
from mymod.submod.deep import Bar
#            ^
"#;
    let report = get_batched_lsp_operations_report(
        &[
            ("main", code),
            ("mymod", mymod_init),
            ("mymod.submod.deep", mymod_submod_deep_init),
        ],
        get_test_report,
    );
    // When clicking on 'submod' in 'mymod.submod.deep', hover shows the full identifier
    // 'mymod.submod.deep' even though mymod.submod itself doesn't exist
    assert!(
        report.contains("mymod.submod.deep: Module[mymod]"),
        "Expected hover to show full module name 'mymod.submod.deep', got: {report}"
    );
}

#[test]
fn hover_on_constructor_shows_instance_type() {
    let code = r#"
class Person:
    def __init__(self, name: str, age: int) -> None: ...

Person()
#^
"#;
    let report = get_batched_lsp_operations_report_allow_error(&[("main", code)], get_test_report);
    assert!(
        report
            .contains("def Person(\n    self: Person,\n    name: str,\n    age: int\n) -> Person"),
        "Expected constructor hover to show complete signature with -> Person, got: {report}"
    );
}

#[test]
fn hover_over_in_operator_shows_contains_dunder() {
    let code = r#"
class Container:
    def __contains__(self, item: int) -> bool: ...

c = Container()
1 in c
# ^
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    // The hover should show the __contains__ method signature
    assert!(
        report.contains("self: Container") && report.contains("item: int"),
        "Expected hover to show __contains__ method signature, got: {report}"
    );
}

#[test]
fn hover_on_constructor_with_arguments() {
    let code = r#"
class Person:
    def __init__(self, name: str, age: int) -> None: ...

Person("Alice", 25)
#^
"#;
    let report = get_batched_lsp_operations_report_allow_error(&[("main", code)], get_test_report);
    assert!(
        report.contains("-> Person"),
        "Expected constructor hover to show -> Person, got: {report}"
    );
}

#[test]
fn hover_over_in_keyword_in_for_loop() {
    let code = r#"
for x in [1, 2, 3]:
#     ^
    pass
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    // The hover should show the iteration keyword with iterable type
    assert!(
        report.contains("(keyword) in") && report.contains("Iteration over"),
        "Expected hover to show iteration keyword info, got: {report}"
    );
}

#[test]
fn hover_on_direct_init_call_shows_none() {
    let code = r#"
class Person:
    def __init__(self, name: str) -> None: ...

p = Person.__new__(Person)
Person.__init__(p, "Alice")
#        ^
"#;
    let report = get_batched_lsp_operations_report_allow_error(&[("main", code)], get_test_report);
    assert!(
        report.contains("-> None"),
        "Expected direct __init__ call to show -> None, got: {report}"
    );
    assert!(
        !report.contains("-> Person") || report.contains("__init__"),
        "Direct __init__ call should show -> None, got: {report}"
    );
}

#[test]
fn hover_over_in_keyword_in_list_comprehension() {
    let code = r#"
result = [x for x in [1, 2, 3] if x in [1]]
#                 ^
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    // The first 'in' is iteration, expect iteration keyword hover info
    assert!(
        report.contains("(keyword) in") && report.contains("Iteration over"),
        "Expected hover for iteration 'in' in comprehension, got: {report}"
    );
}

#[test]
fn hover_on_method_call_unchanged() {
    let code = r#"
class Foo:
    def method(self) -> str: ...

foo = Foo()
foo.method()
#     ^
"#;
    let report = get_batched_lsp_operations_report_allow_error(&[("main", code)], get_test_report);
    assert!(
        report.contains("-> str"),
        "Expected method hover to show -> str, got: {report}"
    );
}

#[test]
fn hover_on_argument_shows_argument_type() {
    let code = r#"
class Person:
    def __init__(self, name: str) -> None: ...

Person("Alice")
#        ^
"#;
    let report = get_batched_lsp_operations_report_allow_error(&[("main", code)], get_test_report);
    // Hovering over a string literal shows its literal type
    assert!(
        report.contains("Literal['Alice']") || report.contains("str"),
        "Expected argument hover to show literal type or str, got: {report}"
    );
    // The argument hover should not show the constructor signature
    assert!(
        !report.contains("__init__") || !report.contains("name: str"),
        "Argument hover should show argument type, not constructor, got: {report}"
    );
}

#[test]
fn hover_on_generic_constructor() {
    let code = r#"
from typing import Generic, TypeVar

T = TypeVar("T")

class Box(Generic[T]):
    def __init__(self, value: T) -> None: ...

Box[str]("hello")
#^
"#;
    let report = get_batched_lsp_operations_report_allow_error(&[("main", code)], get_test_report);
    assert!(
        report.contains("Box[str]"),
        "Expected generic constructor to show Box[str], got: {report}"
    );
}

#[test]
fn hover_over_in_keyword_for_membership_in_comprehension() {
    let code = r#"
result = [x for x in [1, 2, 3] if x in [1]]
#                                   ^
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    // The second 'in' is membership test - should show __contains__ signature
    assert!(
        report.contains("__contains__"),
        "Expected hover for membership 'in' to show __contains__, got: {report}"
    );
}

/// Test for the exact example from issue #1926: [x for x in x if x in [1]]
/// This verifies both uses of `in` show appropriate contextual hover.
#[test]
fn hover_over_in_keyword_issue_1926_example() {
    // First `in` - iteration syntax (for clause)
    let code_iteration = r#"
x = [1, 2, 3]
result = [x for x in x if x in [1]]
#                 ^
"#;
    let report = get_batched_lsp_operations_report(&[("main", code_iteration)], get_test_report);
    assert!(
        report.contains("(keyword) in") && report.contains("Iteration over"),
        "First 'in' should show iteration hover, got: {report}"
    );

    // Second `in` - membership testing operator
    let code_membership = r#"
x = [1, 2, 3]
result = [x for x in x if x in [1]]
#                           ^
"#;
    let report = get_batched_lsp_operations_report(&[("main", code_membership)], get_test_report);
    // For membership test, we expect to see __contains__ method
    assert!(
        report.contains("__contains__"),
        "Second 'in' should show __contains__ hover, got: {report}"
    );
}
