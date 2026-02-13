/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use pretty_assertions::assert_eq;

use crate::state::lsp::AllOffPartial;
use crate::state::lsp::InlayHintConfig;
use crate::state::require::Require;
use crate::test::util::code_frame_of_source_at_position;
use crate::test::util::mk_multi_file_state_assert_no_errors;

fn generate_inlay_hint_report(code: &str, hint_config: InlayHintConfig) -> String {
    let files = [("main", code)];
    let (handles, state) = mk_multi_file_state_assert_no_errors(&files, Require::Exports);
    let mut report = String::new();
    for (name, code) in &files {
        report.push_str("# ");
        report.push_str(name);
        report.push_str(".py\n");
        let handle = handles.get(name).unwrap();
        for hint in state
            .transaction()
            .inlay_hints(handle, hint_config)
            .unwrap()
        {
            report.push_str(&code_frame_of_source_at_position(code, hint.position));
            report.push_str(" inlay-hint: `");
            report.push_str(&hint.label);
            report.push_str("`\n\n");
        }
        report.push('\n');
    }
    report
}

#[test]
fn basic_test() {
    let code = r#"from typing import Literal

def f(x: list[int], y: str, z: Literal[42]):
    return x

yyy = f([1, 2, 3], "test", 42)

def g() -> int:
    return 42

def h(*args):
    return args[0]

i = h()
"#;
    assert_eq!(
        r#"
# main.py
3 | def f(x: list[int], y: str, z: Literal[42]):
                                               ^ inlay-hint: ` -> list[int]`

6 | yyy = f([1, 2, 3], "test", 42)
       ^ inlay-hint: `: list[int]`
"#
        .trim(),
        generate_inlay_hint_report(code, Default::default()).trim()
    );
}

#[test]
fn test_constructor_inlay_hint() {
    let code = r#"
x = int()
y = list([1, 2, 3])
"#;
    // constructor calls for non-generic classes do not show inlay hints
    assert_eq!(
        r#"
# main.py
3 | y = list([1, 2, 3])
     ^ inlay-hint: `: list[int]`
"#
        .trim(),
        generate_inlay_hint_report(code, Default::default()).trim()
    );
}

#[test]
fn test_enum_literal_inlay_hint() {
    let code = r#"
from enum import Enum
import ssl
class X(Enum):
    A = 1
    B = 2

xa = X.A
xa2 = xa
imported = ssl.VerifyMode.CERT_NONE
"#;
    // enum literals do not show inlay hints
    assert_eq!(
        r#"
# main.py
9 | xa2 = xa
       ^ inlay-hint: `: Literal[X.A]`
"#
        .trim(),
        generate_inlay_hint_report(code, Default::default()).trim()
    );
}

#[test]
fn test_tuple_unpacking_inlay_hint() {
    let code = r#"
a = 1
b = 1

x, y = (a, b)
z = a
"#;
    // Individual hints for each unpacked variable
    assert_eq!(
        r#"
# main.py
5 | x, y = (a, b)
     ^ inlay-hint: `: Literal[1]`

5 | x, y = (a, b)
        ^ inlay-hint: `: Literal[1]`

6 | z = a
     ^ inlay-hint: `: Literal[1]`
"#
        .trim(),
        generate_inlay_hint_report(code, Default::default()).trim()
    );
}

#[test]
fn test_tuple_unpacking_from_function_call() {
    let code = r#"
def f() -> tuple[int, str]:
    return (1, "test")

x, y = f()
"#;
    // Individual hints for unpacked values from function calls
    assert_eq!(
        r#"
# main.py
5 | x, y = f()
     ^ inlay-hint: `: int`

5 | x, y = f()
        ^ inlay-hint: `: str`
"#
        .trim(),
        generate_inlay_hint_report(code, Default::default()).trim()
    );
}

#[test]
fn test_tuple_unpacking_no_hint_for_literals() {
    let code = r#"
x, y = (1, 2)
"#;
    // No hints when unpacking literal values
    assert_eq!(
        r#"
# main.py
"#
        .trim(),
        generate_inlay_hint_report(code, Default::default()).trim()
    );
}

#[test]
fn test_tuple_unpacking_with_prior_annotation() {
    let code = r#"
x: int
y: str
x, y = (1, "test")
"#;
    // No hints because variables already have annotations
    assert_eq!(
        r#"
# main.py
"#
        .trim(),
        generate_inlay_hint_report(code, Default::default()).trim()
    );
}

#[test]
fn test_nested_tuple_unpacking() {
    let code = r#"
def f() -> tuple[int, str]:
    return (1, "test")

(a, b), c = f(), 3
"#;
    // Individual hints for nested unpacked values from function call.
    // No hint for c because it's unpacked from a literal (3).
    assert_eq!(
        r#"
# main.py
5 | (a, b), c = f(), 3
      ^ inlay-hint: `: int`

5 | (a, b), c = f(), 3
         ^ inlay-hint: `: str`
"#
        .trim(),
        generate_inlay_hint_report(code, Default::default()).trim()
    );
}

#[test]
fn test_starred_unpacking_from_function() {
    let code = r#"
def get_list() -> list[int]:
    return [1, 2, 3, 4]

a, *b, c = get_list()
"#;
    // All variables get hints since we can't determine if elements are literals
    assert_eq!(
        r#"
# main.py
5 | a, *b, c = get_list()
     ^ inlay-hint: `: int`

5 | a, *b, c = get_list()
         ^ inlay-hint: `: list[int]`

5 | a, *b, c = get_list()
            ^ inlay-hint: `: int`
"#
        .trim(),
        generate_inlay_hint_report(code, Default::default()).trim()
    );
}

#[test]
fn test_starred_unpacking_from_literal() {
    let code = r#"
a, *b, c = [1, 2, 3, 4]
"#;
    // No hints for a and c (literals), but b gets hint since we can't extract slice elements
    assert_eq!(
        r#"
# main.py
2 | a, *b, c = [1, 2, 3, 4]
         ^ inlay-hint: `: list[int]`
"#
        .trim(),
        generate_inlay_hint_report(code, Default::default()).trim()
    );
}

#[test]
fn test_parameter_name_hints() {
    let code = r#"
def my_function(x: int, y: str, z: bool) -> None:
    pass

def another_func(name: str, value: int, flag: bool = False) -> str:
    return name

result = my_function(10, "hello", True)
output = another_func("test", 42, True)

class MyClass:
    def method(self, param1: int, param2: str) -> None:
        pass

obj = MyClass()
obj.method(5, "world")
"#;
    assert_eq!(
        r#"
# main.py
8 | result = my_function(10, "hello", True)
                         ^ inlay-hint: `x= `

8 | result = my_function(10, "hello", True)
                             ^ inlay-hint: `y= `

8 | result = my_function(10, "hello", True)
                                      ^ inlay-hint: `z= `

9 | output = another_func("test", 42, True)
                          ^ inlay-hint: `name= `

9 | output = another_func("test", 42, True)
                                  ^ inlay-hint: `value= `

9 | output = another_func("test", 42, True)
                                      ^ inlay-hint: `flag= `

16 | obj.method(5, "world")
                ^ inlay-hint: `param1= `

16 | obj.method(5, "world")
                   ^ inlay-hint: `param2= `
"#
        .trim(),
        generate_inlay_hint_report(
            code,
            InlayHintConfig {
                call_argument_names: AllOffPartial::All,
                variable_types: false,
                ..Default::default()
            }
        )
        .trim()
    );
}

#[test]
fn test_parameter_name_hints_with_variable_types() {
    let code = r#"
def my_function(x: int, y: str, z: bool) -> None:
    pass

def another_func(name: str, value: int, flag: bool = False) -> str:
    return name

result = my_function(10, "hello", True)
output = another_func("test", 42, True)

class MyClass:
    def method(self, param1: int, param2: str) -> None:
        pass

obj = MyClass()
obj.method(5, "world")
"#;
    assert_eq!(
        r#"
# main.py
8 | result = my_function(10, "hello", True)
          ^ inlay-hint: `: None`

9 | output = another_func("test", 42, True)
          ^ inlay-hint: `: str`

8 | result = my_function(10, "hello", True)
                         ^ inlay-hint: `x= `

8 | result = my_function(10, "hello", True)
                             ^ inlay-hint: `y= `

8 | result = my_function(10, "hello", True)
                                      ^ inlay-hint: `z= `

9 | output = another_func("test", 42, True)
                          ^ inlay-hint: `name= `

9 | output = another_func("test", 42, True)
                                  ^ inlay-hint: `value= `

9 | output = another_func("test", 42, True)
                                      ^ inlay-hint: `flag= `

16 | obj.method(5, "world")
                ^ inlay-hint: `param1= `

16 | obj.method(5, "world")
                   ^ inlay-hint: `param2= `
"#
        .trim(),
        generate_inlay_hint_report(
            code,
            InlayHintConfig {
                call_argument_names: AllOffPartial::All,
                variable_types: true,
                ..Default::default()
            }
        )
        .trim()
    );
}

#[test]
fn test_parameter_name_hints_with_varargs() {
    let code = r#"
def foo(s: str, *args: int, a: int, b: int, t: int) -> None:
    pass

foo("hello", 1, 2, 3, 5, a=1, b=2, t=4)
"#;
    assert_eq!(
        r#"
# main.py
5 | foo("hello", 1, 2, 3, 5, a=1, b=2, t=4)
        ^ inlay-hint: `s= `

5 | foo("hello", 1, 2, 3, 5, a=1, b=2, t=4)
                 ^ inlay-hint: `args= `
"#
        .trim(),
        generate_inlay_hint_report(
            code,
            InlayHintConfig {
                call_argument_names: AllOffPartial::All,
                variable_types: false,
                ..Default::default()
            }
        )
        .trim()
    );
}

/// todo(jvansch): Update test once parameter hints have locations.
#[test]
fn test_parameter_hints_do_not_have_locations() {
    let code = r#"
class MyType:
    pass

def my_function(x: MyType, y: str) -> None:
    pass

result = my_function(MyType(), "hello")
"#;

    let files = [("main", code)];
    let (handles, state) = mk_multi_file_state_assert_no_errors(&files, Require::Exports);
    let handle = handles.get("main").unwrap();

    let hints = state
        .transaction()
        .inlay_hints(
            handle,
            InlayHintConfig {
                call_argument_names: AllOffPartial::All,
                variable_types: false,
                ..Default::default()
            },
        )
        .unwrap();

    let x_hint = hints
        .iter()
        .find(|hint_data| hint_data.label_parts.iter().any(|(text, _)| text == "x= "));

    assert!(x_hint.is_some(), "Should have hint for parameter x");

    if let Some(hint_data) = x_hint {
        let x_part = hint_data.label_parts.iter().find(|(text, _)| text == "x= ");
        assert!(x_part.is_some());

        if let Some((text, location)) = x_part {
            assert_eq!(text, "x= ");
            assert!(
                location.is_none(),
                "Parameter hints should not have locations yet"
            );
        }
    }

    let y_hint = hints
        .iter()
        .find(|hint_data| hint_data.label_parts.iter().any(|(text, _)| text == "y= "));

    assert!(y_hint.is_some(), "Should have hint for parameter y");

    if let Some(hint_data) = y_hint {
        let y_part = hint_data.label_parts.iter().find(|(text, _)| text == "y= ");
        assert!(y_part.is_some());

        if let Some((_, location)) = y_part {
            assert!(
                location.is_none(),
                "Parameter hints should not have locations yet"
            );
        }
    }
}

#[test]
fn test_unpacked_variables_are_not_insertable() {
    let code = r#"
def get_tuple() -> tuple[int, str]:
    return (1, "hello")

# Regular variable assignment - should be insertable
result = get_tuple()

# Unpacked variables - should NOT be insertable
x, y = get_tuple()
"#;

    let files = [("main", code)];
    let (handles, state) = mk_multi_file_state_assert_no_errors(&files, Require::Exports);
    let handle = handles.get("main").unwrap();

    let hints = state
        .transaction()
        .inlay_hints(handle, Default::default())
        .unwrap();

    // Should have 3 hints: result, x, and y
    assert_eq!(hints.len(), 3, "Expected 3 hints");

    // First hint is for 'result' - should be insertable
    let result_hint = &hints[0];
    assert!(
        result_hint.insertable,
        "Regular variable 'result' should be insertable"
    );

    let x_hint = &hints[1];
    assert!(
        !x_hint.insertable,
        "Unpacked variable 'x' should NOT be insertable"
    );

    let y_hint = &hints[2];
    assert!(
        !y_hint.insertable,
        "Unpacked variable 'y' should NOT be insertable"
    );
}
