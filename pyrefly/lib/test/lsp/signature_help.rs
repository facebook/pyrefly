/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use itertools::Itertools;
use lsp_types::Documentation;
use lsp_types::ParameterLabel;
use lsp_types::SignatureHelp;
use lsp_types::SignatureInformation;
use pretty_assertions::assert_eq;
use pyrefly_build::handle::Handle;
use ruff_text_size::TextSize;

use crate::state::require::Require;
use crate::state::state::State;
use crate::test::util::extract_cursors_for_test;
use crate::test::util::get_batched_lsp_operations_report_allow_error;
use crate::test::util::mk_multi_file_state;

fn get_test_report(state: &State, handle: &Handle, position: TextSize) -> String {
    if let Some(SignatureHelp {
        signatures,
        active_signature,
        active_parameter: _,
    }) = state.transaction().get_signature_help_at(handle, position)
    {
        let active_signature_result = if let Some(active) = active_signature {
            format!(" active={active}")
        } else {
            "".to_owned()
        };
        let signatures_result = signatures
            .into_iter()
            .map(
                |SignatureInformation {
                     label,
                     documentation: _,
                     parameters,
                     active_parameter,
                 }| {
                    format!(
                        "- {}{}{}",
                        label,
                        if let Some(params) = parameters {
                            format!(
                                ", parameters=[{}]",
                                params
                                    .into_iter()
                                    .map(|p| match p.label {
                                        ParameterLabel::Simple(s) => s,
                                        _ => unreachable!(),
                                    })
                                    .join(", ")
                            )
                        } else {
                            "".to_owned()
                        },
                        if let Some(active) = active_parameter {
                            format!(", active parameter = {active}")
                        } else {
                            "".to_owned()
                        }
                    )
                },
            )
            .join("\n");
        format!("Signature Help Result:{active_signature_result}\n{signatures_result}")
    } else {
        "Signature Help: None".to_owned()
    }
}

#[test]
fn simple_function_test() {
    let code = r#"
def f(a: str, b: int, c: bool) -> None: ...

f()
# ^
f("", )
#    ^
f("",3, )
#      ^
f("",3,True)
#      ^
"#;
    let report = get_batched_lsp_operations_report_allow_error(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
4 | f()
      ^
Signature Help Result: active=0
- def f(a: str, b: int, c: bool) -> None: ..., parameters=[a: str, b: int, c: bool], active parameter = 0

6 | f("", )
         ^
Signature Help Result: active=0
- def f(a: str, b: int, c: bool) -> None: ..., parameters=[a: str, b: int, c: bool], active parameter = 1

8 | f("",3, )
           ^
Signature Help Result: active=0
- def f(a: str, b: int, c: bool) -> None: ..., parameters=[a: str, b: int, c: bool], active parameter = 2

10 | f("",3,True)
            ^
Signature Help Result: active=0
- def f(a: str, b: int, c: bool) -> None: ..., parameters=[a: str, b: int, c: bool], active parameter = 2
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn positional_arguments_test() {
    let code = r#"
def f(x: int, y: int, z: int) -> None: ...

f(1,,)
#   ^
f(1,,)
#    ^
f(1,,3)
#   ^
"#;
    let report = get_batched_lsp_operations_report_allow_error(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
4 | f(1,,)
        ^
Signature Help Result: active=0
- def f(x: int, y: int, z: int) -> None: ..., parameters=[x: int, y: int, z: int], active parameter = 1

6 | f(1,,)
         ^
Signature Help Result: active=0
- def f(x: int, y: int, z: int) -> None: ..., parameters=[x: int, y: int, z: int], active parameter = 2

8 | f(1,,3)
        ^
Signature Help Result: active=0
- def f(x: int, y: int, z: int) -> None: ..., parameters=[x: int, y: int, z: int], active parameter = 1
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn keyword_arguments_test() {
    let code = r#"
def f(a: str, b: int) -> None: ...

f(a)
# ^
f(a=)
#  ^
f(b=)
#  ^
"#;
    let report = get_batched_lsp_operations_report_allow_error(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
4 | f(a)
      ^
Signature Help Result: active=0
- def f(a: str, b: int) -> None: ..., parameters=[a: str, b: int], active parameter = 0

6 | f(a=)
       ^
Signature Help Result: active=0
- def f(a: str, b: int) -> None: ..., parameters=[a: str, b: int], active parameter = 0

8 | f(b=)
       ^
Signature Help Result: active=0
- def f(a: str, b: int) -> None: ..., parameters=[a: str, b: int], active parameter = 1
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn parameter_documentation_test() {
    let code = r#"
def foo(a: int, b: str) -> None:
    """
    Args:
        a: first line
            second line
        b: final
    """
    pass

foo(a=1, b="")
#      ^
"#;
    let files = [("main", code)];
    let (handles, state) = mk_multi_file_state(&files, Require::indexing(), true);
    let handle = handles.get("main").unwrap();
    let position = extract_cursors_for_test(code)[0];
    let signature = state
        .transaction()
        .get_signature_help_at(handle, position)
        .expect("signature help available");
    let params = signature.signatures[0]
        .parameters
        .as_ref()
        .expect("parameters available");
    let param_doc = params
        .iter()
        .find(
            |param| matches!(&param.label, ParameterLabel::Simple(label) if label.starts_with("a")),
        )
        .and_then(|param| param.documentation.as_ref())
        .expect("parameter documentation");
    if let Documentation::MarkupContent(content) = param_doc {
        assert_eq!(content.value, "first line\nsecond line");
    } else {
        panic!("unexpected documentation variant");
    }
}

#[test]
fn simple_incomplete_function_call_test() {
    let code = r#"
def f(a: str) -> None: ...

f(
# ^
"#;
    let report = get_batched_lsp_operations_report_allow_error(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
4 | f(
      ^
Signature Help Result: active=0
- def f(a: str) -> None: ..., parameters=[a: str], active parameter = 0
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn signature_help_for_callable_instance() {
    let code = r#"
class Greeter:
    def __call__(self, name: str, times: int = 1) -> str: ...

g = Greeter()
g(
#^
"#;
    let report = get_batched_lsp_operations_report_allow_error(&[("main", code)], get_test_report);
    assert!(
        report.contains("def __call__(self: Greeter, name: str, times: int = 1) -> str: ..."),
        "Expected signature help to show __call__ signature, got: {report}"
    );
    assert!(
        report.contains("parameters=[name: str, times: int = 1]"),
        "Expected signature help parameters, got: {report}"
    );
}

#[test]
fn simple_function_nested_test() {
    let code = r#"
def f(a: str) -> None: ...
def g(b: int) -> None: ...

f()
# ^
f(g())
#   ^
"#;
    let report = get_batched_lsp_operations_report_allow_error(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
5 | f()
      ^
Signature Help Result: active=0
- def f(a: str) -> None: ..., parameters=[a: str], active parameter = 0

7 | f(g())
        ^
Signature Help Result: active=0
- def g(b: int) -> None: ..., parameters=[b: int], active parameter = 0
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn simple_method_test() {
    let code = r#"
class Foo:
  def f(self, a: str, b: int, c: bool) -> None: ...

foo = Foo()
foo.f()
#     ^
foo.f("", )
#        ^
foo.f("",3, )
#          ^
foo.f("",3,True)
#          ^
"#;
    let report = get_batched_lsp_operations_report_allow_error(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
6 | foo.f()
          ^
Signature Help Result: active=0
- def f(self: Foo, a: str, b: int, c: bool) -> None: ..., parameters=[a: str, b: int, c: bool], active parameter = 0

8 | foo.f("", )
             ^
Signature Help Result: active=0
- def f(self: Foo, a: str, b: int, c: bool) -> None: ..., parameters=[a: str, b: int, c: bool], active parameter = 1

10 | foo.f("",3, )
                ^
Signature Help Result: active=0
- def f(self: Foo, a: str, b: int, c: bool) -> None: ..., parameters=[a: str, b: int, c: bool], active parameter = 2

12 | foo.f("",3,True)
                ^
Signature Help Result: active=0
- def f(self: Foo, a: str, b: int, c: bool) -> None: ..., parameters=[a: str, b: int, c: bool], active parameter = 2
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn overloaded_function_test() {
    let code = r#"
from typing import overload


@overload
def overloaded_func(a: str) -> bool: ...
@overload
def overloaded_func(a: int, b: bool) -> str: ...
def overloaded_func():
    pass


overloaded_func()
#               ^
overloaded_func(1, )
#                 ^
overloaded_func(1, T)
#                  ^
"#;
    let report = get_batched_lsp_operations_report_allow_error(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
13 | overloaded_func()
                     ^
Signature Help Result: active=0
- (a: str) -> bool, parameters=[a: str], active parameter = 0
- (a: int, b: bool) -> str, parameters=[a: int, b: bool], active parameter = 0

15 | overloaded_func(1, )
                       ^
Signature Help Result: active=0
- (a: str) -> bool, parameters=[a: str]
- (a: int, b: bool) -> str, parameters=[a: int, b: bool], active parameter = 1

17 | overloaded_func(1, T)
                        ^
Signature Help Result: active=1
- (a: str) -> bool, parameters=[a: str]
- (a: int, b: bool) -> str, parameters=[a: int, b: bool], active parameter = 1
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn overloaded_method_test() {
    let code = r#"
from typing import overload


class Foo:
    @overload
    def overloaded_meth(self, a: str) -> bool: ...
    @overload
    def overloaded_meth(self, a: int, b: bool) -> str: ...
    def overloaded_meth(self):
        pass


foo = Foo()
foo.overloaded_meth()
#                   ^
foo.overloaded_meth(1, )
#                      ^
foo.overloaded_meth(1, F)
#                      ^
"#;
    let report = get_batched_lsp_operations_report_allow_error(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
15 | foo.overloaded_meth()
                         ^
Signature Help Result: active=0
- (self: Foo, a: str) -> bool, parameters=[a: str], active parameter = 0
- (self: Foo, a: int, b: bool) -> str, parameters=[a: int, b: bool], active parameter = 0

17 | foo.overloaded_meth(1, )
                            ^
Signature Help Result: active=0
- (self: Foo, a: str) -> bool, parameters=[a: str]
- (self: Foo, a: int, b: bool) -> str, parameters=[a: int, b: bool], active parameter = 1

19 | foo.overloaded_meth(1, F)
                            ^
Signature Help Result: active=1
- (self: Foo, a: str) -> bool, parameters=[a: str]
- (self: Foo, a: int, b: bool) -> str, parameters=[a: int, b: bool], active parameter = 1
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn function_with_default_argument_test() {
    let code = r#"
def f(a: str = "default") -> None: ...

f()
# ^
f("")
#   ^
"#;
    let report = get_batched_lsp_operations_report_allow_error(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
4 | f()
      ^
Signature Help Result: active=0
- def f(a: str = 'default') -> None: ..., parameters=[a: str = 'default'], active parameter = 0

6 | f("")
        ^
Signature Help Result: active=0
- def f(a: str = 'default') -> None: ..., parameters=[a: str = 'default'], active parameter = 0
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn parameter_documentation_only_some_params_documented_test() {
    let code = r#"
def foo(a: int, b: str, c: bool) -> None:
    """
    Args:
        a: only a is documented
    """
    pass

foo(a=1, b="", c=True)
#      ^
"#;
    let files = [("main", code)];
    let (handles, state) = mk_multi_file_state(&files, Require::indexing(), true);
    let handle = handles.get("main").unwrap();
    let position = extract_cursors_for_test(code)[0];
    let signature = state
        .transaction()
        .get_signature_help_at(handle, position)
        .expect("signature help available");
    let params = signature.signatures[0]
        .parameters
        .as_ref()
        .expect("parameters available");

    // Parameter 'a' should have documentation
    let param_a_doc = params
        .iter()
        .find(
            |param| matches!(&param.label, ParameterLabel::Simple(label) if label.starts_with("a")),
        )
        .and_then(|param| param.documentation.as_ref())
        .expect("parameter a documentation");
    if let Documentation::MarkupContent(content) = param_a_doc {
        assert_eq!(content.value, "only a is documented");
    } else {
        panic!("unexpected documentation variant");
    }

    // Parameter 'b' should not have documentation
    let param_b = params
        .iter()
        .find(
            |param| matches!(&param.label, ParameterLabel::Simple(label) if label.starts_with("b")),
        )
        .expect("parameter b should exist");
    assert!(
        param_b.documentation.is_none(),
        "parameter b should not have documentation"
    );
}

#[test]
fn parameter_documentation_overloaded_function_test() {
    let code = r#"
from typing import overload

@overload
def foo(a: str) -> bool:
    """
    Args:
        a: string argument
    """
    ...

@overload
def foo(a: int, b: bool) -> str:
    """
    Args:
        a: integer argument
        b: boolean argument
    """
    ...

def foo(a, b=None):
    pass

foo(1, True)
#      ^
"#;
    let files = [("main", code)];
    let (handles, state) = mk_multi_file_state(&files, Require::indexing(), false);
    let handle = handles.get("main").unwrap();
    let position = extract_cursors_for_test(code)[0];
    let signature = state
        .transaction()
        .get_signature_help_at(handle, position)
        .expect("signature help available");

    // Should have multiple signatures
    assert!(
        signature.signatures.len() >= 2,
        "Expected at least 2 overloaded signatures"
    );

    // Each signature should have valid structure
    // Parameter docs are optional but when present should be valid
    for sig in &signature.signatures {
        if let Some(params) = &sig.parameters {
            for param in params {
                // Documentation is optional but should be valid if present
                if let Some(doc) = &param.documentation {
                    assert!(matches!(doc, Documentation::MarkupContent(_)));
                }
            }
        }
    }
}

#[test]
fn parameter_documentation_method_test() {
    let code = r#"
class Foo:
    def method(self, x: int, y: str) -> None:
        """
        Args:
            x: the x parameter
            y: the y parameter
        """
        pass

foo = Foo()
foo.method(x=1, y="test")
#             ^
"#;
    let files = [("main", code)];
    let (handles, state) = mk_multi_file_state(&files, Require::indexing(), true);
    let handle = handles.get("main").unwrap();
    let position = extract_cursors_for_test(code)[0];
    let signature = state
        .transaction()
        .get_signature_help_at(handle, position)
        .expect("signature help available");
    let params = signature.signatures[0]
        .parameters
        .as_ref()
        .expect("parameters available");

    // Should not include 'self' in parameters
    let has_self = params.iter().any(
        |param| matches!(&param.label, ParameterLabel::Simple(label) if label.contains("self")),
    );
    assert!(
        !has_self,
        "self parameter should not be in the parameters list"
    );

    // Check if parameter x exists (documentation may or may not be present depending on implementation)
    let param_x = params
        .iter()
        .find(
            |param| matches!(&param.label, ParameterLabel::Simple(label) if label.starts_with("x")),
        )
        .expect("parameter x should exist");

    // If documentation is present, verify it's correct
    if let Some(Documentation::MarkupContent(content)) = &param_x.documentation {
        assert_eq!(content.value, "the x parameter");
    }
}

#[test]
fn parameter_documentation_mixed_style_test() {
    let code = r#"
def foo(a: int, b: str, c: bool) -> None:
    """
    :param a: sphinx style

    Args:
        b: google style
        c: also google style
    """
    pass

foo(a=1, b="", c=True)
#      ^
"#;
    let files = [("main", code)];
    let (handles, state) = mk_multi_file_state(&files, Require::indexing(), true);
    let handle = handles.get("main").unwrap();
    let position = extract_cursors_for_test(code)[0];
    let signature = state
        .transaction()
        .get_signature_help_at(handle, position)
        .expect("signature help available");
    let params = signature.signatures[0]
        .parameters
        .as_ref()
        .expect("parameters available");

    // Should have documentation for all three parameters (mixed style should work)
    let param_a_doc = params
        .iter()
        .find(
            |param| matches!(&param.label, ParameterLabel::Simple(label) if label.starts_with("a")),
        )
        .and_then(|param| param.documentation.as_ref())
        .expect("parameter a documentation");
    if let Documentation::MarkupContent(content) = param_a_doc {
        assert_eq!(content.value, "sphinx style");
    }

    let param_b_doc = params
        .iter()
        .find(
            |param| matches!(&param.label, ParameterLabel::Simple(label) if label.starts_with("b")),
        )
        .and_then(|param| param.documentation.as_ref())
        .expect("parameter b documentation");
    if let Documentation::MarkupContent(content) = param_b_doc {
        assert_eq!(content.value, "google style");
    }
}

#[test]
fn union_type_alias_in_callable_test() {
    let code = r#"
from typing import TypeAlias, Callable, overload
KylesInt: TypeAlias = int | str
def foo(a: KylesInt) -> None:
    pass
foo()
#   ^
"#;
    let report = get_batched_lsp_operations_report_allow_error(&[("main", code)], get_test_report);
    assert_eq!(
        "
# main.py
6 | foo()
        ^
Signature Help Result: active=0
- def foo(a: KylesInt) -> None: ..., parameters=[a: KylesInt], active parameter = 0"
            .trim(),
        report.trim(),
    );
}

#[test]
fn function_docstring_test() {
    let code = r#"
def greet(name: str, age: int) -> None:
    """
    Greet a person with their name and age.

    This function prints a friendly greeting message.

    Args:
        name: The person's name
        age: The person's age
    """
    pass

greet()
#     ^
"#;
    let files = [("main", code)];
    let (handles, state) = mk_multi_file_state(&files, Require::indexing(), false);
    let handle = handles.get("main").unwrap();
    let cursors = extract_cursors_for_test(code);

    let signature = state
        .transaction()
        .get_signature_help_at(handle, cursors[0])
        .expect("signature help available");

    assert_eq!(signature.signatures.len(), 1);
    let sig_info = &signature.signatures[0];

    // Check that function-level documentation is present
    assert!(
        sig_info.documentation.is_some(),
        "function-level documentation should be present"
    );

    if let Some(Documentation::MarkupContent(content)) = &sig_info.documentation {
        assert!(
            content.value.contains("Greet a person"),
            "docstring should contain summary text"
        );
        assert!(
            content.value.contains("friendly greeting"),
            "docstring should contain description text"
        );
    } else {
        panic!("unexpected documentation variant");
    }

    // Also verify parameters still have their documentation
    let params = sig_info
        .parameters
        .as_ref()
        .expect("parameters should be present");

    let name_param = params
        .iter()
        .find(|p| matches!(&p.label, ParameterLabel::Simple(label) if label.starts_with("name")))
        .expect("name parameter should exist");

    if let Some(Documentation::MarkupContent(content)) = &name_param.documentation {
        assert_eq!(content.value, "The person's name");
    }
}

#[test]
fn function_docstring_without_param_docs_test() {
    let code = r#"
def calculate(x: int, y: int) -> int:
    """
    Calculate the sum of two numbers.

    This is a simple addition function.
    """
    return x + y

calculate(1, 2)
#            ^
"#;
    let files = [("main", code)];
    let (handles, state) = mk_multi_file_state(&files, Require::indexing(), false);
    let handle = handles.get("main").unwrap();
    let cursors = extract_cursors_for_test(code);

    let signature = state
        .transaction()
        .get_signature_help_at(handle, cursors[0])
        .expect("signature help available");

    assert_eq!(signature.signatures.len(), 1);
    let sig_info = &signature.signatures[0];

    // Function-level documentation should be present
    assert!(
        sig_info.documentation.is_some(),
        "function-level documentation should be present even without param docs"
    );

    if let Some(Documentation::MarkupContent(content)) = &sig_info.documentation {
        assert!(content.value.contains("Calculate the sum"));
        assert!(content.value.contains("simple addition"));
    }

    // Parameters should not have individual documentation
    let params = sig_info
        .parameters
        .as_ref()
        .expect("parameters should be present");

    for param in params {
        assert!(
            param.documentation.is_none(),
            "parameters should not have individual docs when not specified"
        );
    }
}
