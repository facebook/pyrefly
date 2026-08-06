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
use crate::test::util::mk_multi_file_state;
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
        for hint_data in state
            .transaction()
            .inlay_hints(handle, hint_config, Default::default())
            .unwrap()
        {
            let pos = hint_data.position;
            let label_parts = hint_data.label_parts;
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
fn pattern_capture_hint_not_insertable() {
    // A capture pattern cannot be annotated inline, so its inferred-type hint is
    // shown but must be marked non-insertable.
    let code = r#"
def f(xs: list[int]) -> None:
    match xs:
        case [head]:
            print(head)
"#;
    let files = [("main", code)];
    let (handles, state) = mk_multi_file_state_assert_no_errors(&files, Require::Exports);
    let handle = handles.get("main").unwrap();
    let hints = state
        .transaction()
        .inlay_hints(handle, InlayHintConfig::default(), Default::default())
        .unwrap();
    let head_hint = hints
        .iter()
        .find(|h| h.label_parts.iter().any(|(text, _)| text.contains("int")))
        .expect("expected an inferred-type hint for the `head` capture");
    assert!(
        head_hint.edits.is_none(),
        "capture inlay hints must not be insertable"
    );
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
fn test_new_type_inlay_hint() {
    let code = r#"from typing import NewType

N = NewType("N", int)
x = N
"#;
    assert_eq!(
        r#"
# main.py
4 | x = N
     ^ inlay-hint: `: (_x: int) -> N`
"#
        .trim(),
        generate_inlay_hint_report(code, Default::default()).trim()
    );

    let files = [("main", code)];
    let (handles, state) = mk_multi_file_state_assert_no_errors(&files, Require::Exports);
    let handle = handles.get("main").unwrap();
    let hints = state
        .transaction()
        .inlay_hints(handle, Default::default(), Default::default())
        .unwrap();
    assert_eq!(hints.len(), 1);
    assert!(hints[0].edits.is_none());
}

/// Test that we handle invalid `NewType`s gracefully when generating inlay hints.
#[test]
fn test_invalid_new_type_inlay_hint() {
    let code = r#"from typing import NewType

Bad = NewType("Bad", int | str)
x = Bad
"#;
    let files = [("main", code)];
    let (handles, state) = mk_multi_file_state(&files, Require::Exports, false);
    let handle = handles.get("main").unwrap();
    let hints = state
        .transaction()
        .inlay_hints(handle, Default::default(), Default::default())
        .unwrap();
    assert_eq!(hints.len(), 1);
}

#[test]
fn test_dunder_new_implicit_self_return_inlay_hint() {
    let code = r#"
class A:
    def __new__(cls, x: int | None = None):
        if x is None:
            return cls.__new__(cls, 5)
        return super().__new__(cls)
"#;
    assert_eq!(
        r#"
# main.py
3 |     def __new__(cls, x: int | None = None):
                                              ^ inlay-hint: ` -> Self`
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

fn parameter_name_hint_labels(code: &str, assert_zero_errors: bool) -> Vec<String> {
    let files = [("main", code)];
    let (handles, state) = mk_multi_file_state(&files, Require::Exports, assert_zero_errors);
    let handle = handles.get("main").unwrap();
    state
        .transaction()
        .inlay_hints(
            handle,
            InlayHintConfig {
                call_argument_names: AllOffPartial::All,
                variable_types: false,
                ..Default::default()
            },
            Default::default(),
        )
        .unwrap()
        .into_iter()
        .flat_map(|hint| hint.label_parts.into_iter().map(|(text, _)| text))
        .filter(|text| text.ends_with("= "))
        .collect()
}

#[test]
fn test_parameter_name_hint_after_keyword_argument() {
    let code = r#"
def test(a: str, b: str) -> None:
    pass

test(a="a", "b")
"#;
    assert_eq!(parameter_name_hint_labels(code, false), vec!["b= "]);
}

#[test]
fn test_parameter_name_hints_for_multiple_positional_arguments() {
    let code = r#"
def test(a: str, b: str) -> None:
    pass

test("a", "b")
"#;
    assert_eq!(parameter_name_hint_labels(code, true), vec!["a= ", "b= "]);
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

#[test]
fn test_parameter_name_hints_for_callable_object() {
    let code = r#"
class MarkDecorator:
    def __call__(self, *fixtures: str) -> None:
        pass

mark = MarkDecorator()
mark("database", "cache")
"#;
    assert_eq!(
        r#"
# main.py
7 | mark("database", "cache")
         ^ inlay-hint: `fixtures= `
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
            Default::default(),
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
        .inlay_hints(handle, Default::default(), Default::default())
        .unwrap();

    // Should have 3 hints: result, x, and y
    assert_eq!(hints.len(), 3, "Expected 3 hints");

    // First hint is for 'result' - should be insertable
    let result_hint = &hints[0];
    assert!(
        result_hint.edits.is_some(),
        "Regular variable 'result' should be insertable"
    );

    let x_hint = &hints[1];
    assert!(
        x_hint.edits.is_none(),
        "Unpacked variable 'x' should NOT be insertable"
    );

    let y_hint = &hints[2];
    assert!(
        y_hint.edits.is_none(),
        "Unpacked variable 'y' should NOT be insertable"
    );
}

#[test]
fn test_insertable_hint_combines_multiple_imports() {
    let files = [
        (
            "foo",
            r#"
class Foo:
    pass

def make_foo() -> Foo:
    return Foo()
"#,
        ),
        (
            "bar",
            r#"
class Bar:
    pass

def make_bar() -> Bar:
    return Bar()
"#,
        ),
        (
            "main",
            r#"
from foo import make_foo
from bar import make_bar

def choose(flag: bool):
    if flag:
        return make_foo()
    return make_bar()

value = choose(True)
"#,
        ),
    ];
    let (handles, state) = mk_multi_file_state_assert_no_errors(&files, Require::Exports);
    let hints = state
        .transaction()
        .inlay_hints(
            handles.get("main").unwrap(),
            Default::default(),
            Default::default(),
        )
        .unwrap();
    let hint = hints
        .iter()
        .find(|hint| {
            hint.edits
                .as_ref()
                .is_some_and(|edits| edits.annotation.starts_with(": "))
        })
        .expect("expected an insertable union hint");

    let edits = hint.edits.as_ref().unwrap();
    assert_eq!(edits.annotation, ": Bar | Foo");
    assert_eq!(edits.imports.len(), 1);
    assert_eq!(
        edits.imports[0].1,
        "from bar import Bar\nfrom foo import Foo\n"
    );
}

/// A module whose exports are referenced by the annotations under test.
const LIB: (&str, &str) = (
    "lib",
    r#"
class Value:
    class Inner:
        pass

class Other:
    pass

def make() -> Value:
    return Value()

def make_inner() -> Value.Inner:
    return Value.Inner()
"#,
);

/// The annotation text and combined import text of the single insertable hint
/// in `main`.
fn insertable_hint(files: &[(&'static str, &str)]) -> (String, String) {
    let (handles, state) = mk_multi_file_state_assert_no_errors(files, Require::Exports);
    let hints = state
        .transaction()
        .inlay_hints(
            handles.get("main").unwrap(),
            Default::default(),
            Default::default(),
        )
        .unwrap();
    let edits = hints
        .iter()
        .filter_map(|hint| hint.edits.as_ref())
        .collect::<Vec<_>>();
    assert_eq!(edits.len(), 1, "expected exactly one insertable hint");
    let imports = edits[0]
        .imports
        .iter()
        .map(|(_, text)| text.as_str())
        .collect::<String>();
    (edits[0].annotation.clone(), imports)
}

#[test]
fn test_insertable_hint_prefers_from_import() {
    let (annotation, imports) = insertable_hint(&[
        LIB,
        (
            "main",
            r#"
from lib import make

value = make()
"#,
        ),
    ]);
    assert_eq!(annotation, ": Value");
    assert_eq!(imports, "from lib import Value\n");
}

#[test]
fn test_insertable_hint_keeps_existing_module_import() {
    let (annotation, imports) = insertable_hint(&[
        LIB,
        (
            "main",
            r#"
import lib
from lib import make

value = make()
"#,
        ),
    ]);
    assert_eq!(annotation, ": lib.Value");
    assert_eq!(imports, "");
}

#[test]
fn test_insertable_hint_keeps_existing_module_alias() {
    let (annotation, imports) = insertable_hint(&[
        LIB,
        (
            "main",
            r#"
import lib as l
from lib import make

value = make()
"#,
        ),
    ]);
    assert_eq!(annotation, ": l.Value");
    assert_eq!(imports, "");
}

#[test]
fn test_insertable_hint_uses_existing_from_import() {
    let (annotation, imports) = insertable_hint(&[
        LIB,
        (
            "main",
            r#"
from lib import Value, make

value = make()
"#,
        ),
    ]);
    assert_eq!(annotation, ": Value");
    assert_eq!(imports, "");
}

#[test]
fn test_insertable_hint_uses_existing_from_import_alias() {
    let (annotation, imports) = insertable_hint(&[
        LIB,
        (
            "main",
            r#"
from lib import Value as V, make

value = make()
"#,
        ),
    ]);
    assert_eq!(annotation, ": V");
    assert_eq!(imports, "");
}

#[test]
fn test_insertable_hint_imports_head_of_nested_name() {
    let (annotation, imports) = insertable_hint(&[
        LIB,
        (
            "main",
            r#"
from lib import make_inner

value = make_inner()
"#,
        ),
    ]);
    assert_eq!(annotation, ": Value.Inner");
    assert_eq!(imports, "from lib import Value\n");
}

#[test]
fn test_insertable_hint_groups_names_from_one_module() {
    let (annotation, imports) = insertable_hint(&[
        LIB,
        (
            "pair",
            r#"
import lib

def make() -> lib.Value | lib.Other:
    return lib.Value()
"#,
        ),
        (
            "main",
            r#"
from pair import make

value = make()
"#,
        ),
    ]);
    assert_eq!(annotation, ": Other | Value");
    assert_eq!(imports, "from lib import Other, Value\n");
}

#[test]
fn test_insertable_hint_qualifies_name_used_by_two_modules() {
    // Importing `Value` from one module would leave the other reference
    // indistinguishable from it, so both stay qualified.
    let (annotation, imports) = insertable_hint(&[
        (
            "left",
            r#"
class Value:
    pass
"#,
        ),
        (
            "right",
            r#"
class Value:
    pass
"#,
        ),
        (
            "both",
            r#"
import left
import right

def make() -> left.Value | right.Value:
    return left.Value()
"#,
        ),
        (
            "main",
            r#"
from both import make

value = make()
"#,
        ),
    ]);
    assert_eq!(annotation, ": left.Value | right.Value");
    assert_eq!(imports, "import left\nimport right\n");
}

#[test]
fn test_insertable_hint_qualifies_name_bound_by_another_import() {
    let (annotation, imports) = insertable_hint(&[
        LIB,
        (
            "other",
            r#"
class Value:
    pass
"#,
        ),
        (
            "main",
            r#"
from lib import make
from other import Value

x: Value = Value()
value = make()
"#,
        ),
    ]);
    assert_eq!(annotation, ": lib.Value");
    assert_eq!(imports, "import lib\n");
}

#[test]
fn test_insertable_hint_qualifies_name_bound_at_module_scope() {
    let (annotation, imports) = insertable_hint(&[
        LIB,
        (
            "main",
            r#"
from lib import make

class Value:
    pass

value = make()
"#,
        ),
    ]);
    assert_eq!(annotation, ": lib.Value");
    assert_eq!(imports, "import lib\n");
}

#[test]
fn test_insertable_hint_qualifies_name_shadowing_builtin() {
    // `from shadow import filter` would change what `filter` means for the
    // whole file, not just inside the annotation.
    let (annotation, imports) = insertable_hint(&[
        (
            "shadow",
            r#"
class filter:
    pass

def make() -> filter:
    return filter()
"#,
        ),
        (
            "main",
            r#"
from shadow import make

value = make()
"#,
        ),
    ]);
    assert_eq!(annotation, ": shadow.filter");
    assert_eq!(imports, "import shadow\n");
}

#[test]
fn test_insertable_hint_orders_plain_imports_before_from_imports() {
    let (annotation, imports) = insertable_hint(&[
        LIB,
        (
            "left",
            r#"
class Value:
    pass
"#,
        ),
        (
            "right",
            r#"
class Value:
    pass
"#,
        ),
        (
            "mixed",
            r#"
import left
import lib
import right

def make() -> left.Value | right.Value | lib.Other:
    return left.Value()
"#,
        ),
        (
            "main",
            r#"
from mixed import make

value = make()
"#,
        ),
    ]);
    assert_eq!(annotation, ": Other | left.Value | right.Value");
    assert_eq!(
        imports,
        "import left\nimport right\nfrom lib import Other\n"
    );
}

#[test]
fn test_insertable_hint_renders_unknown_as_any() {
    let files = [
        (
            "producer",
            r#"
def make_values():
    return []
"#,
        ),
        (
            "main",
            r#"
from producer import make_values

values = make_values()
"#,
        ),
    ];
    let (handles, state) = mk_multi_file_state_assert_no_errors(&files, Require::Exports);
    let hints = state
        .transaction()
        .inlay_hints(
            handles.get("main").unwrap(),
            Default::default(),
            Default::default(),
        )
        .unwrap();
    let edits = hints
        .iter()
        .find_map(|hint| hint.edits.as_ref())
        .expect("expected an insertable hint");

    assert_eq!(edits.annotation, ": list[Any]");
    assert_eq!(edits.imports.len(), 1);
    assert_eq!(edits.imports[0].1, "from typing import Any\n");
}

#[test]
fn test_class_attribute_inlay_hint() {
    let code = r#"
def make_list() -> list[int]:
    return [1, 2, 3]

class MyClass:
    def __init__(self, x: int, y: str) -> None:
        self.x = x
        self.y = y
        self.data = make_list()
        self.name = "literal"
        self.count = 42
"#;
    // self.x and self.y are suppressed (self.x = x pattern, type visible at parameter).
    // self.data gets a hint (function call return).
    // self.name and self.count are suppressed (assigned from literals).
    assert_eq!(
        r#"
# main.py
9 |         self.data = make_list()
                     ^ inlay-hint: `: list[int]`
"#
        .trim(),
        generate_inlay_hint_report(code, Default::default()).trim()
    );
}

#[test]
fn test_class_attribute_inlay_hint_disabled() {
    let code = r#"
def make_list() -> list[int]:
    return [1, 2, 3]

class MyClass:
    def __init__(self) -> None:
        self.data = make_list()
"#;
    // No hints when variable_types is disabled
    assert_eq!(
        r#"
# main.py
"#
        .trim(),
        generate_inlay_hint_report(
            code,
            InlayHintConfig {
                variable_types: false,
                ..Default::default()
            }
        )
        .trim()
    );
}

#[test]
fn test_class_attribute_with_annotation() {
    let code = r#"
class MyClass:
    def __init__(self) -> None:
        self.x: int = 42
"#;
    // No hint because the attribute has an explicit annotation
    assert_eq!(
        r#"
# main.py
"#
        .trim(),
        generate_inlay_hint_report(code, Default::default()).trim()
    );
}

#[test]
fn test_class_attribute_constructor_suppressed() {
    let code = r#"
class Inner:
    pass

class Outer:
    def __init__(self) -> None:
        self.inner = Inner()
"#;
    // Constructor call matching the inferred class name should be suppressed
    assert_eq!(
        r#"
# main.py
"#
        .trim(),
        generate_inlay_hint_report(code, Default::default()).trim()
    );
}

#[test]
fn test_class_attribute_self_x_eq_x_suppressed() {
    let code = r#"
class MyClass:
    def __init__(self, x: int, y: str, data: list[int]) -> None:
        self.x = x
        self.y = y
        self.data = data
"#;
    // All attributes use the self.x = x pattern, so all hints are suppressed.
    assert_eq!(
        r#"
# main.py
"#
        .trim(),
        generate_inlay_hint_report(code, Default::default()).trim()
    );
}

#[test]
fn test_class_attribute_different_name_not_suppressed() {
    let code = r#"
class MyClass:
    def __init__(self, value: int) -> None:
        self.x = value
"#;
    // self.x = value is NOT the self.x = x pattern (names differ), so hint is shown.
    assert_eq!(
        r#"
# main.py
4 |         self.x = value
                  ^ inlay-hint: `: int`
"#
        .trim(),
        generate_inlay_hint_report(code, Default::default()).trim()
    );
}

#[test]
fn test_class_attribute_new_type_inlay_hint() {
    let code = r#"from typing import NewType

N = NewType("N", int)

class MyClass:
    def __init__(self) -> None:
        self.x = N
"#;
    // NewType attributes show the constructor signature, not the invalid `type[N]`.
    assert_eq!(
        r#"
# main.py
7 |         self.x = N
                  ^ inlay-hint: `: (_x: int) -> N`
"#
        .trim(),
        generate_inlay_hint_report(code, Default::default()).trim()
    );

    let files = [("main", code)];
    let (handles, state) = mk_multi_file_state_assert_no_errors(&files, Require::Exports);
    let handle = handles.get("main").unwrap();
    let hints = state
        .transaction()
        .inlay_hints(handle, Default::default(), Default::default())
        .unwrap();
    assert_eq!(hints.len(), 1);
    // NewType is a callable alias, so `type[N]` is not a valid annotation to insert.
    assert!(hints[0].edits.is_none());
}
