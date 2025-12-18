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
    let (handles, state) = mk_multi_file_state_assert_no_errors(&files, Require::indexing());
    let mut report = String::new();
    for (name, code) in &files {
        report.push_str("# ");
        report.push_str(name);
        report.push_str(".py\n");
        let handle = handles.get(name).unwrap();
        for (pos, label_parts) in state
            .transaction()
            .inlay_hints(handle, hint_config)
            .unwrap()
        {
            report.push_str(&code_frame_of_source_at_position(code, pos));
            report.push_str(" inlay-hint: `");
            // Concatenate label parts into a single string
            let hint: String = label_parts.iter().map(|(text, _)| text.as_str()).collect();
            report.push_str(&hint);
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
    let (handles, state) = mk_multi_file_state_assert_no_errors(&files, Require::indexing());
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
        .find(|(_, parts)| parts.iter().any(|(text, _)| text == "x= "));

    assert!(x_hint.is_some(), "Should have hint for parameter x");

    if let Some((_, parts)) = x_hint {
        let x_part = parts.iter().find(|(text, _)| text == "x= ");
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
        .find(|(_, parts)| parts.iter().any(|(text, _)| text == "y= "));

    assert!(y_hint.is_some(), "Should have hint for parameter y");

    if let Some((_, parts)) = y_hint {
        let y_part = parts.iter().find(|(text, _)| text == "y= ");
        assert!(y_part.is_some());

        if let Some((_, location)) = y_part {
            assert!(
                location.is_none(),
                "Parameter hints should not have locations yet"
            );
        }
    }
}
