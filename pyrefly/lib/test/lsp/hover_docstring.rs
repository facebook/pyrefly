/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use itertools::Itertools as _;
use pretty_assertions::assert_eq;
use pyrefly_build::handle::Handle;
use pyrefly_python::docstring::Docstring;
use ruff_text_size::TextSize;

use crate::state::lsp::FindPreference;
use crate::state::state::State;
use crate::test::util::get_batched_lsp_operations_report;

fn test_report_factory(
    // File content for the file the docstring is found in
    file_content: &'static str,
) -> impl Fn(&State, &Handle, ruff_text_size::TextSize) -> std::string::String {
    move |state: &State, handle: &Handle, position: TextSize| -> String {
        let results = state
            .transaction()
            .find_definition(handle, position, FindPreference::default())
            .into_iter()
            .filter_map(|t| {
                let docstring_range = t.docstring_range?;
                Some(docstring_range)
            })
            .collect::<Vec<_>>();
        if !results.is_empty() {
            results
                .into_iter()
                .map(|d| {
                    let content = &file_content[d.start().to_usize()..d.end().to_usize()];
                    format!("Docstring Result: `{}`", Docstring::clean(content))
                })
                .join("\n")
        } else {
            "Docstring Result: None".to_owned()
        }
    }
}

#[test]
fn class_test() {
    let code = r#"
class F:
    """Test docstring"""
print(F)
#     ^
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], test_report_factory(code));
    assert_eq!(
        r#"
# main.py
4 | print(F)
          ^
Docstring Result: `Test docstring`
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn function_test() {
    let code = r#"
def f():
    """Test docstring"""
print(f())
#     ^
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], test_report_factory(code));
    assert_eq!(
        r#"
# main.py
4 | print(f())
          ^
Docstring Result: `Test docstring`
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn raw_quotes_test() {
    let code = r#"
def f():
#   ^
    r"""Test docstring"""
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], test_report_factory(code));
    assert_eq!(
        r#"
# main.py
2 | def f():
        ^
Docstring Result: `Test docstring`
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn class_itself_test() {
    let code = r#"
class Foo:
#     ^
    """Test docstring"""
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], test_report_factory(code));
    assert_eq!(
        r#"
# main.py
2 | class Foo:
          ^
Docstring Result: `Test docstring`
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn function_itself_test() {
    let code = r#"
def f():
#   ^
    """Test docstring"""
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], test_report_factory(code));
    assert_eq!(
        r#"
# main.py
2 | def f():
        ^
Docstring Result: `Test docstring`
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn attribute_itself_test() {
    let code = r#"
class Foo:
    def f():
#       ^
        """Test docstring"""
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], test_report_factory(code));
    assert_eq!(
        r#"
# main.py
3 |     def f():
            ^
Docstring Result: `Test docstring`
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn method_test() {
    let code = r#"
class Foo:
    def f(self):
        """Test docstring"""
print(Foo().f())
#           ^
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], test_report_factory(code));
    assert_eq!(
        r#"
# main.py
5 | print(Foo().f())
                ^
Docstring Result: `Test docstring`
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn attribute_test() {
    let code = r#"
class Foo:
    def f():
        """Test docstring"""
print(Foo.f())
#         ^
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], test_report_factory(code));
    assert_eq!(
        r#"
# main.py
5 | print(Foo.f())
              ^
Docstring Result: `Test docstring`
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn nested_class_test() {
    let code = r#"
class Foo:
    class Bar:
        """Test docstring"""
print(Foo.Bar)
#           ^
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], test_report_factory(code));
    assert_eq!(
        r#"
# main.py
5 | print(Foo.Bar)
                ^
Docstring Result: `Test docstring`
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
    let report = get_batched_lsp_operations_report(
        &[("main", code), ("lib", lib)],
        test_report_factory(lib),
    );
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
    let report = get_batched_lsp_operations_report(
        &[("main", code), ("lib", lib)],
        test_report_factory(lib),
    );
    assert_eq!(
        r#"
# main.py
3 | print(Foo().f())
                ^
Docstring Result: `Test docstring`


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
    let report = get_batched_lsp_operations_report(
        &[("main", code), ("lib", lib)],
        test_report_factory(lib),
    );
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
    let report = get_batched_lsp_operations_report(
        &[("main", code), ("lib", lib)],
        test_report_factory(lib),
    );
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
    let report = get_batched_lsp_operations_report(
        &[("main", code), ("lib", lib)],
        test_report_factory(lib),
    );
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

#[test]
fn test_removes_block_quote_symbols_at_start_of_line() {
    let code = r#"
def fun() -> None:
    """
    >>> d = {"col1": [1, 2], "col2": [3, 4]}
    """
    pass

f = fun()
#   ^
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], test_report_factory(code));
    assert_eq!(
        r#"
# main.py
8 | f = fun()
        ^
Docstring Result: `  
d = {"col1": [1, 2], "col2": [3, 4]}  
`
"#
        .trim(),
        report.trim(),
    );
}
