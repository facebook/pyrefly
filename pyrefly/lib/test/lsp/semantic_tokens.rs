/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use pretty_assertions::assert_eq;

use crate::state::semantic_tokens::SemanticTokensLegends;
use crate::test::util::mk_multi_file_state_assert_no_errors;

fn assert_full_semantic_tokens(files: &[(&'static str, &str)], expected: &str) {
    let (handles, state) = mk_multi_file_state_assert_no_errors(files);
    let mut report = String::new();
    for (name, code) in files {
        report.push_str("# ");
        report.push_str(name);
        report.push_str(".py\n");
        let handle = handles.get(name).unwrap();
        let tokens = state.transaction().semantic_tokens(handle, None).unwrap();

        let mut start_line: usize = 0;
        let mut start_col: usize = 0;
        let lsp_semantic_token_legends = SemanticTokensLegends::lsp_semantic_token_legends();
        let semantic_token_legends = SemanticTokensLegends::new();
        for token in tokens {
            start_col = match token.delta_line {
                0 => start_col + token.delta_start as usize,
                _ => token.delta_start as usize,
            };
            start_line += token.delta_line as usize;
            let line = code.lines().nth(start_line).unwrap();
            let end = start_col + token.length as usize;
            let text = if line.len() >= end {
                &line[start_col..end].to_owned()
            } else {
                &format!(
                    "{}... (continues {} characters)",
                    &line[start_col..],
                    end - line.len()
                )
            };
            report.push_str(&format!(
                "line: {}, column: {}, length: {}, text: {}\n",
                start_line, start_col, token.length, text
            ));
            report.push_str(&format!(
                "token-type: {}",
                lsp_semantic_token_legends.token_types[token.token_type as usize].as_str()
            ));
            let modifiers = semantic_token_legends.get_modifiers(token.token_modifiers_bitset);
            if !modifiers.is_empty() {
                report.push_str(", token-modifiers: [");
                for modifier in modifiers {
                    report.push_str(modifier.as_str());
                }
                report.push(']');
            }
            report.push_str("\n\n");
        }
        report.push('\n');
    }
    assert_eq!(expected.trim(), report.trim(), "actual:\n{}", report.trim());
}

#[test]
fn module_name_test() {
    let code = r#"
from json import decoder
"#;
    assert_full_semantic_tokens(
        &[("main", code)],
        r#"
# main.py
line: 1, column: 17, length: 7, text: decoder
token-type: namespace
"#,
    );
}

#[test]
fn variable_and_constant_test() {
    let code = r#"
foo = 3
ALL_CAPS = "foo"
str(foo) + ALL_CAPS
"#;
    assert_full_semantic_tokens(
        &[("main", code)],
        r#"
# main.py
line: 1, column: 0, length: 3, text: foo
token-type: variable

line: 2, column: 0, length: 8, text: ALL_CAPS
token-type: variable, token-modifiers: [readonly]

line: 3, column: 0, length: 3, text: str
token-type: class, token-modifiers: [defaultLibrary]

line: 3, column: 4, length: 3, text: foo
token-type: variable

line: 3, column: 11, length: 8, text: ALL_CAPS
token-type: variable, token-modifiers: [readonly]
"#,
    );
}

#[test]
fn function_test() {
    let code = r#"
def foo(v: int) -> int: ...
bar = 3

foo
foo(foo(3))
foo(bar)
"#;
    assert_full_semantic_tokens(
        &[("main", code)],
        r#"
# main.py
line: 1, column: 4, length: 3, text: foo
token-type: function

line: 1, column: 8, length: 1, text: v
token-type: parameter

line: 1, column: 11, length: 3, text: int
token-type: class, token-modifiers: [defaultLibrary]

line: 1, column: 19, length: 3, text: int
token-type: class, token-modifiers: [defaultLibrary]

line: 2, column: 0, length: 3, text: bar
token-type: variable

line: 4, column: 0, length: 3, text: foo
token-type: function

line: 5, column: 0, length: 3, text: foo
token-type: function

line: 5, column: 4, length: 3, text: foo
token-type: function

line: 6, column: 0, length: 3, text: foo
token-type: function

line: 6, column: 4, length: 3, text: bar
token-type: variable"#,
    );
}

#[test]
fn method_and_property_test() {
    let code = r#"
class Test:
    def foo(self) -> int: ...
    def bar(self, x: int) -> int: ...
    x: int
Test.foo
Test().foo()
Test().x
Test().bar(Test().x)
"#;
    assert_full_semantic_tokens(
        &[("main", code)],
        r#"
# main.py
line: 1, column: 6, length: 4, text: Test
token-type: class

line: 2, column: 8, length: 3, text: foo
token-type: function

line: 2, column: 12, length: 4, text: self
token-type: parameter

line: 2, column: 21, length: 3, text: int
token-type: class, token-modifiers: [defaultLibrary]

line: 3, column: 8, length: 3, text: bar
token-type: function

line: 3, column: 12, length: 4, text: self
token-type: parameter

line: 3, column: 18, length: 1, text: x
token-type: parameter

line: 3, column: 21, length: 3, text: int
token-type: class, token-modifiers: [defaultLibrary]

line: 3, column: 29, length: 3, text: int
token-type: class, token-modifiers: [defaultLibrary]

line: 4, column: 7, length: 3, text: int
token-type: class, token-modifiers: [defaultLibrary]

line: 5, column: 0, length: 4, text: Test
token-type: class

line: 5, column: 5, length: 3, text: foo
token-type: property

line: 6, column: 0, length: 4, text: Test
token-type: class

line: 6, column: 7, length: 3, text: foo
token-type: method

line: 7, column: 0, length: 4, text: Test
token-type: class

line: 7, column: 7, length: 1, text: x
token-type: property

line: 8, column: 0, length: 4, text: Test
token-type: class

line: 8, column: 7, length: 3, text: bar
token-type: method

line: 8, column: 11, length: 4, text: Test
token-type: class

line: 8, column: 18, length: 1, text: x
token-type: property
"#,
    );
}

#[test]
fn type_alias_test() {
    let code = r#"
type A = int
def foo(v: A) -> int: 
  return 3

type A2 = A
"#;
    assert_full_semantic_tokens(
        &[("main", code)],
        r#"
# main.py
line: 1, column: 5, length: 1, text: A
token-type: interface

line: 1, column: 9, length: 3, text: int
token-type: class, token-modifiers: [defaultLibrary]

line: 2, column: 4, length: 3, text: foo
token-type: function

line: 2, column: 8, length: 1, text: v
token-type: parameter

line: 2, column: 11, length: 1, text: A
token-type: interface

line: 2, column: 17, length: 3, text: int
token-type: class, token-modifiers: [defaultLibrary]

line: 5, column: 5, length: 2, text: A2
token-type: interface

line: 5, column: 10, length: 1, text: A
token-type: interface
"#,
    );
}

#[test]
fn type_param_test() {
    let code = r#"
def foo[T](v: T) -> T: 
  return v
"#;
    assert_full_semantic_tokens(
        &[("main", code)],
        r#"
# main.py
line: 1, column: 4, length: 3, text: foo
token-type: function

line: 1, column: 8, length: 1, text: T
token-type: typeParameter

line: 1, column: 11, length: 1, text: v
token-type: parameter

line: 1, column: 14, length: 1, text: T
token-type: typeParameter

line: 1, column: 20, length: 1, text: T
token-type: typeParameter

line: 2, column: 9, length: 1, text: v
token-type: parameter
"#,
    );
}

#[test]
fn control_flow_merge_test_try() {
    let code = r#"
import typing
try:
    print("hello")
except:
    print("world")
"#;
    assert_full_semantic_tokens(
        &[("main", code)],
        r#"
# main.py
line: 1, column: 7, length: 6, text: typing
token-type: namespace, token-modifiers: [defaultLibrary]

line: 3, column: 4, length: 5, text: print
token-type: function, token-modifiers: [defaultLibrary]

line: 5, column: 4, length: 5, text: print
token-type: function, token-modifiers: [defaultLibrary]
"#,
    );
}

#[test]
fn control_flow_merge_test_def() {
    let code = r#"
b: bool = True

if b:
    def f() -> int:
        return 1
else:
    def f() -> bool:
        return False

f()
"#;
    assert_full_semantic_tokens(
        &[("main", code)],
        r#"
# main.py
line: 1, column: 0, length: 1, text: b
token-type: variable

line: 1, column: 3, length: 4, text: bool
token-type: class, token-modifiers: [defaultLibrary]

line: 3, column: 3, length: 1, text: b
token-type: variable

line: 4, column: 8, length: 1, text: f
token-type: function

line: 4, column: 15, length: 3, text: int
token-type: class, token-modifiers: [defaultLibrary]

line: 7, column: 8, length: 1, text: f
token-type: function

line: 7, column: 15, length: 4, text: bool
token-type: class, token-modifiers: [defaultLibrary]

line: 10, column: 0, length: 1, text: f
token-type: function
"#,
    );
}

#[test]
fn kwargs() {
    let code = r#"
class Foo:
    def foo(self, a: int): ...
def foo(a: int): ...
foo(a=1)
Foo().foo(a=1)
"#;
    assert_full_semantic_tokens(
        &[("main", code)],
        r#"
# main.py
line: 1, column: 6, length: 3, text: Foo
token-type: class

line: 2, column: 8, length: 3, text: foo
token-type: function

line: 2, column: 12, length: 4, text: self
token-type: parameter

line: 2, column: 18, length: 1, text: a
token-type: parameter

line: 2, column: 21, length: 3, text: int
token-type: class, token-modifiers: [defaultLibrary]

line: 3, column: 4, length: 3, text: foo
token-type: function

line: 3, column: 8, length: 1, text: a
token-type: parameter

line: 3, column: 11, length: 3, text: int
token-type: class, token-modifiers: [defaultLibrary]

line: 4, column: 0, length: 3, text: foo
token-type: function

line: 4, column: 4, length: 1, text: a
token-type: parameter

line: 5, column: 0, length: 3, text: Foo
token-type: class

line: 5, column: 6, length: 3, text: foo
token-type: method

line: 5, column: 10, length: 1, text: a
token-type: parameter"#,
    );
}
