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
    **kwargs: Unpack[TypedDict[Payload]]
) -> None: ...
```
"#
        .trim(),
        report.trim(),
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
(method) __matmul__: (
    self: Matrix,
    other: Matrix
) -> Matrix
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
