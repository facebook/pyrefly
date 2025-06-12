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
                "token-type: {}\n\n",
                SemanticTokensLegends::lsp_semantic_token_legends().token_types
                    [token.token_type as usize]
                    .as_str()
            ));
        }
        report.push('\n');
    }
    assert_eq!(expected.trim(), report.trim());
}

#[test]
fn variable_and_constant_test() {
    // TODO: treat all caps as constants
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
token-type: variable

line: 3, column: 4, length: 3, text: foo
token-type: variable

line: 3, column: 11, length: 8, text: ALL_CAPS
token-type: variable
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
    x: int
Test.foo
Test().foo()
Test().x
"#;
    assert_full_semantic_tokens(
        &[("main", code)],
        r#"
# main.py
line: 4, column: 0, length: 4, text: Test
token-type: class

line: 4, column: 5, length: 3, text: foo
token-type: property

line: 5, column: 0, length: 4, text: Test
token-type: class

line: 5, column: 7, length: 3, text: foo
token-type: method

line: 6, column: 0, length: 4, text: Test
token-type: class

line: 6, column: 7, length: 1, text: x
token-type: property"#,
    );
}

#[test]
fn type_param_test() {
    let code = r#"
type T = int
def foo(v: T) -> int: 
  return 3

type T2 = T
"#;
    assert_full_semantic_tokens(
        &[("main", code)],
        r#"
# main.py
line: 2, column: 11, length: 1, text: T
token-type: interface

line: 5, column: 10, length: 1, text: T
token-type: interface
"#,
    );
}

#[test]
fn try_test() {
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
"#,
    );
}
