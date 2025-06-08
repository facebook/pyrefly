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

fn get_test_report(state: &State, handle: &Handle, position: TextSize) -> String {
    if let Some((_, _, Some(t))) = state.transaction().find_definition(handle, position) {
        format!("Docstring Result: `{}`", t.as_string())
    } else {
        "Docstring Result: None".to_owned()
    }
}

// TODO(kylei) same-module docstrings
#[test]
fn function_test() {
    let code = r#"
def f():
    """Test docstring"""
print(f())
#     ^
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
4 | print(f())
          ^
Docstring Result: None
"#
        .trim(),
        report.trim(),
    );
}

// TODO(kylei) same-module docstrings
#[test]
fn function_itself_test() {
    let code = r#"
def f():
#   ^
    """Test docstring"""
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
2 | def f():
        ^
Docstring Result: None
"#
        .trim(),
        report.trim(),
    );
}

// TODO(kylei): attribute docstrings
#[test]
fn method_test() {
    let code = r#"
class Foo:
    def f(self):
        """Test docstring"""
print(Foo().f())
#           ^
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
5 | print(Foo().f())
                ^
Docstring Result: None
"#
        .trim(),
        report.trim(),
    );
}

// TODO(kylei): attribute docstrings
#[test]
fn attribute_test() {
    let code = r#"
class Foo:
    def f():
        """Test docstring"""
print(Foo.f())
#         ^
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
5 | print(Foo.f())
              ^
Docstring Result: None
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn cross_module_function_test() {
    let lib = r#"
def f():
    """Test docstring"""
    return 1"#;
    let code = r#"
from lib import f
print(f())
#     ^
"#;
    let report =
        get_batched_lsp_operations_report(&[("main", code), ("lib", lib)], get_test_report);
    assert_eq!(
        r#"
# main.py
3 | print(f())
          ^
Docstring Result: `Test docstring`


# lib.py
"#
        .trim(),
        report.trim(),
    );
}

// TODO(kylei): attribute docstrings
#[test]
fn cross_module_method_test() {
    let lib = r#"
class Foo:
    def f(self):
        """Test docstring"""
        return 1"#;
    let code = r#"
from lib import Foo
print(Foo().f())
#           ^
"#;
    let report =
        get_batched_lsp_operations_report(&[("main", code), ("lib", lib)], get_test_report);
    assert_eq!(
        r#"
# main.py
3 | print(Foo().f())
                ^
Docstring Result: None


# lib.py
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn cross_module_class_test() {
    let lib = r#"
class Foo:
    """Test docstring"""
    x: int = 5"#;
    let code = r#"
from lib import Foo
Foo()
# ^
"#;
    let report =
        get_batched_lsp_operations_report(&[("main", code), ("lib", lib)], get_test_report);
    assert_eq!(
        r#"
# main.py
3 | Foo()
      ^
Docstring Result: `Test docstring`


# lib.py
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn module_test() {
    let lib = r#"
"""Test docstring"""
print("test")"#;
    let code = r#"
import lib
#      ^
"#;
    let report =
        get_batched_lsp_operations_report(&[("main", code), ("lib", lib)], get_test_report);
    assert_eq!(
        r#"
# main.py
2 | import lib
           ^
Docstring Result: `Test docstring`


# lib.py
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn module_binding_test() {
    let lib = r#"
"""Test docstring"""
print("test")"#;
    let code = r#"
import lib
print(lib)
#     ^
"#;
    let report =
        get_batched_lsp_operations_report(&[("main", code), ("lib", lib)], get_test_report);
    assert_eq!(
        r#"
# main.py
3 | print(lib)
          ^
Docstring Result: `Test docstring`


# lib.py
"#
        .trim(),
        report.trim(),
    );
}
