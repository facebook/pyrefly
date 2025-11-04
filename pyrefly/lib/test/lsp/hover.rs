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
    match get_hover(&state.transaction(), handle, position) {
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
    assert_eq!(
        r#"
# main.py
7 | foo.meth()
        ^
```python
(attribute) meth: def meth(self: Foo) -> None: ...
```

9 | xyz = [foo.meth]
     ^
```python
(variable) xyz: list[(self: Foo) -> None]
```
"#
        .trim(),
        report.trim(),
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
