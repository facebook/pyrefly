/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use pretty_assertions::assert_eq;
use pyrefly_build::handle::Handle;
use ruff_text_size::TextSize;

use crate::lsp::non_wasm::external_provider::compute_qualified_name;
use crate::state::lsp::FindPreference;
use crate::state::state::State;
use crate::test::util::get_batched_lsp_operations_report;

fn get_test_report(state: &State, handle: &Handle, position: TextSize) -> String {
    let transaction = state.transaction();
    let defs = transaction.find_definition(handle, position, FindPreference::default());
    let Some(definition) = defs.into_iter().next() else {
        return "Qualified Name: no definition found".to_owned();
    };
    match compute_qualified_name(&transaction, handle, &definition) {
        Some(name) => format!("Qualified Name: `{name}`"),
        None => "Qualified Name: None".to_owned(),
    }
}

#[test]
fn module_import() {
    let code = r#"
import os
#      ^
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
2 | import os
           ^
Qualified Name: `os`
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn function() {
    let code = r#"
def foo(x: int) -> str:
    return str(x)
y = foo
#   ^
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
4 | y = foo
        ^
Qualified Name: `main.foo`
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn class() {
    let code = r#"
class MyClass:
    pass
y = MyClass
#   ^
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
4 | y = MyClass
        ^
Qualified Name: `main.MyClass`
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn method() {
    let code = r#"
class MyClass:
    def my_method(self) -> None:
        pass
MyClass.my_method
#       ^
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
5 | MyClass.my_method
            ^
Qualified Name: `main.MyClass.my_method`
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn variable_with_type_annotation() {
    let code = r#"
x: int = 42
y = x
#   ^
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
3 | y = x
        ^
Qualified Name: `main.x`
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn cross_module_function() {
    let code_a = r#"
def helper() -> int:
    return 1
"#;
    let code_main = r#"
from a import helper
y = helper
#   ^
"#;
    let report =
        get_batched_lsp_operations_report(&[("a", code_a), ("main", code_main)], get_test_report);
    assert_eq!(
        r#"
# a.py

# main.py
3 | y = helper
        ^
Qualified Name: `a.helper`
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn cross_module_class() {
    let code_a = r#"
class Foo:
    pass
"#;
    let code_main = r#"
from a import Foo
y = Foo
#   ^
"#;
    let report =
        get_batched_lsp_operations_report(&[("a", code_a), ("main", code_main)], get_test_report);
    assert_eq!(
        r#"
# a.py

# main.py
3 | y = Foo
        ^
Qualified Name: `a.Foo`
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn nested_class() {
    let code = r#"
class Outer:
    class Inner:
        pass
y = Outer.Inner
#         ^
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
5 | y = Outer.Inner
              ^
Qualified Name: `main.Outer.Inner`
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn nested_class_method() {
    let code = r#"
class Outer:
    class Inner:
        def my_method(self) -> None:
            pass
Outer.Inner.my_method
#           ^
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
6 | Outer.Inner.my_method
                ^
Qualified Name: `main.Outer.Inner.my_method`
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn nested_function() {
    let code = r#"
def outer():
    def inner():
        pass
    inner
#   ^
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
5 |     inner
        ^
Qualified Name: `main.outer.<locals>.inner`
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn deeply_nested_function() {
    let code = r#"
def a():
    def b():
        def c():
            pass
        c
#       ^
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
6 |         c
            ^
Qualified Name: `main.a.<locals>.b.<locals>.c`
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn function_in_class_method() {
    let code = r#"
class C:
    def m(self):
        def helper():
            pass
        helper
#       ^
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
6 |         helper
            ^
Qualified Name: `main.C.m.<locals>.helper`
"#
        .trim(),
        report.trim(),
    );
}
