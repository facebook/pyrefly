/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use pretty_assertions::assert_eq;

use crate::state::require::Require;
use crate::state::semantic_tokens::SemanticTokensLegends;
use crate::test::util::mk_multi_file_state_assert_no_errors;

fn utf16_to_byte_index(line: &str, utf16_offset: usize) -> usize {
    if utf16_offset == 0 {
        return 0;
    }
    let mut seen = 0;
    for (byte_idx, ch) in line.char_indices() {
        if seen == utf16_offset {
            return byte_idx;
        }
        seen += ch.len_utf16();
        if seen == utf16_offset {
            return byte_idx + ch.len_utf8();
        }
    }
    line.len()
}

fn assert_full_semantic_tokens(files: &[(&'static str, &str)], expected: &str) {
    let (handles, state) = mk_multi_file_state_assert_no_errors(files, Require::Exports);
    let mut report = String::new();
    for (name, code) in files {
        report.push_str("# ");
        report.push_str(name);
        report.push_str(".py\n");
        let handle = handles.get(name).unwrap();
        let tokens = state
            .transaction()
            .semantic_tokens(handle, None, None)
            .unwrap();

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
            let token_length = token.length as usize;
            let end_utf16 = start_col + token_length;
            let line_utf16_len = line.encode_utf16().count();
            let text = if end_utf16 <= line_utf16_len {
                let start_byte = utf16_to_byte_index(line, start_col);
                let end_byte = utf16_to_byte_index(line, end_utf16);
                line[start_byte..end_byte].to_owned()
            } else {
                let start_byte = utf16_to_byte_index(line, start_col.min(line_utf16_len));
                format!(
                    "{}... (continues {} characters)",
                    &line[start_byte..],
                    end_utf16 - line_utf16_len
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
line: 1, column: 5, length: 4, text: json
token-type: namespace

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
fn type_aware_variables_test() {
    let code = r#"
from typing import Literal

class Foo:
  def bar(self):
    return 3
Bar = Foo

def fun():
  return 3

fun2 = fun
meth = Bar().bar
"#;
    assert_full_semantic_tokens(
        &[("main", code)],
        r#"
# main.py
line: 1, column: 5, length: 6, text: typing
token-type: namespace

line: 1, column: 19, length: 7, text: Literal
token-type: class, token-modifiers: [defaultLibrary]

line: 3, column: 6, length: 3, text: Foo
token-type: class

line: 4, column: 6, length: 3, text: bar
token-type: method

line: 4, column: 10, length: 4, text: self
token-type: parameter

line: 6, column: 0, length: 3, text: Bar
token-type: class

line: 6, column: 6, length: 3, text: Foo
token-type: class

line: 8, column: 4, length: 3, text: fun
token-type: function

line: 11, column: 0, length: 4, text: fun2
token-type: function

line: 11, column: 7, length: 3, text: fun
token-type: function

line: 12, column: 0, length: 4, text: meth
token-type: method

line: 12, column: 7, length: 3, text: Bar
token-type: class

line: 12, column: 13, length: 3, text: bar
token-type: method
"#,
    );
}

#[test]
fn deprecated_token_for_disabled_branch() {
    let code = r#"
import sys

if sys.version_info < (3, 9):
    class Legacy:
        value = 1
else:
    class Modern:
        value = 2
"#;

    assert_full_semantic_tokens(
        &[("main", code)],
        r#"
# main.py
line: 1, column: 7, length: 3, text: sys
token-type: namespace

line: 3, column: 3, length: 3, text: sys
token-type: namespace

line: 3, column: 7, length: 12, text: version_info
token-type: property

line: 4, column: 10, length: 6, text: Legacy
token-type: class

line: 5, column: 8, length: 5, text: value
token-type: variable

line: 7, column: 10, length: 6, text: Modern
token-type: class

line: 8, column: 8, length: 5, text: value
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
token-type: method

line: 2, column: 12, length: 4, text: self
token-type: parameter

line: 2, column: 21, length: 3, text: int
token-type: class, token-modifiers: [defaultLibrary]

line: 3, column: 8, length: 3, text: bar
token-type: method

line: 3, column: 12, length: 4, text: self
token-type: parameter

line: 3, column: 18, length: 1, text: x
token-type: parameter

line: 3, column: 21, length: 3, text: int
token-type: class, token-modifiers: [defaultLibrary]

line: 3, column: 29, length: 3, text: int
token-type: class, token-modifiers: [defaultLibrary]

line: 4, column: 4, length: 1, text: x
token-type: variable

line: 4, column: 7, length: 3, text: int
token-type: class, token-modifiers: [defaultLibrary]

line: 5, column: 0, length: 4, text: Test
token-type: class

line: 5, column: 5, length: 3, text: foo
token-type: function

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
fn for_name_test() {
    let code = r#"
for x in range(1):
    [(y, z) for y, z in {x: 0}.items()]
"#;
    assert_full_semantic_tokens(
        &[("main", code)],
        r#"
# main.py
line: 1, column: 4, length: 1, text: x
token-type: variable

line: 1, column: 9, length: 5, text: range
token-type: class, token-modifiers: [defaultLibrary]

line: 2, column: 6, length: 1, text: y
token-type: variable

line: 2, column: 9, length: 1, text: z
token-type: variable

line: 2, column: 16, length: 1, text: y
token-type: variable

line: 2, column: 19, length: 1, text: z
token-type: variable

line: 2, column: 25, length: 1, text: x
token-type: variable

line: 2, column: 31, length: 5, text: items
token-type: method
"#,
    );
}

#[test]
fn with_name_test() {
    let code = r#"
with open("foo.txt") as f1, open("bar.txt") as f2:
    pass
"#;
    assert_full_semantic_tokens(
        &[("main", code)],
        r#"
# main.py
line: 1, column: 5, length: 4, text: open
token-type: function, token-modifiers: [defaultLibrary]

line: 1, column: 24, length: 2, text: f1
token-type: variable

line: 1, column: 28, length: 4, text: open
token-type: function, token-modifiers: [defaultLibrary]

line: 1, column: 47, length: 2, text: f2
token-type: variable
"#,
    );
}

#[test]
fn exception_handler_name_test() {
    let code = r#"
try:
    pass
except Exception as e:
    pass
"#;
    assert_full_semantic_tokens(
        &[("main", code)],
        r#"
# main.py
line: 3, column: 7, length: 9, text: Exception
token-type: class, token-modifiers: [defaultLibrary]

line: 3, column: 20, length: 1, text: e
token-type: variable
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
token-type: method

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

#[test]
fn assignment_first_line() {
    let code = r#"foo = 3"#;
    assert_full_semantic_tokens(
        &[("main", code)],
        r#"
# main.py
line: 0, column: 0, length: 3, text: foo
token-type: variable
"#,
    );
}

// todo(kylei): should be 3 semantic tokens (including after reassignment) #1033
#[test]
fn reassignment() {
    let code = r#"
foo = 3
foo += 1
foo"#;
    assert_full_semantic_tokens(
        &[("main", code)],
        r#"
# main.py
line: 1, column: 0, length: 3, text: foo
token-type: variable

line: 2, column: 0, length: 3, text: foo
token-type: variable"#,
    );
}

#[test]
fn unassigned_attribute() {
    let code = r#"
class Test:
    x: int
    x: int = 5"#;
    assert_full_semantic_tokens(
        &[("main", code)],
        r#"
# main.py
line: 1, column: 6, length: 4, text: Test
token-type: class

line: 2, column: 4, length: 1, text: x
token-type: variable

line: 2, column: 7, length: 3, text: int
token-type: class, token-modifiers: [defaultLibrary]

line: 3, column: 4, length: 1, text: x
token-type: variable

line: 3, column: 7, length: 3, text: int
token-type: class, token-modifiers: [defaultLibrary]"#,
    );
}

#[test]
fn nested_class() {
    let code = r#"
from typing import SupportsFloat
class Test:
    class nested: pass
Test.nested"#;
    assert_full_semantic_tokens(
        &[("main", code)],
        r#"
    # main.py
line: 1, column: 5, length: 6, text: typing
token-type: namespace

line: 1, column: 19, length: 13, text: SupportsFloat
token-type: class, token-modifiers: [defaultLibrary]

line: 2, column: 6, length: 4, text: Test
token-type: class

line: 3, column: 10, length: 6, text: nested
token-type: class

line: 4, column: 0, length: 4, text: Test
token-type: class

line: 4, column: 5, length: 6, text: nested
token-type: class"#,
    );
}

#[test]
fn module_dot_access() {
    let code = r#"
import typing
from typing import SupportsFloat
typing.SupportsFloat
SupportsFloat"#;
    assert_full_semantic_tokens(
        &[("main", code)],
        r#"
    # main.py
line: 1, column: 7, length: 6, text: typing
token-type: namespace, token-modifiers: [defaultLibrary]

line: 2, column: 5, length: 6, text: typing
token-type: namespace

line: 2, column: 19, length: 13, text: SupportsFloat
token-type: class, token-modifiers: [defaultLibrary]

line: 3, column: 0, length: 6, text: typing
token-type: namespace, token-modifiers: [defaultLibrary]

line: 3, column: 7, length: 13, text: SupportsFloat
token-type: class

line: 4, column: 0, length: 13, text: SupportsFloat
token-type: class, token-modifiers: [defaultLibrary]"#,
    );
}

#[test]
fn module_overloaded_function_dot_access() {
    let lib = r#"
from typing import overload

@overload
def foo(a: str) -> None: ...
@overload
def foo(a: int) -> None: ...
def foo(a: str | int) -> None: ...
"#;
    let code = r#"
import lib
lib.foo
"#;
    assert_full_semantic_tokens(
        &[("main", code), ("lib", lib)],
        r#"
# main.py
line: 1, column: 7, length: 3, text: lib
token-type: namespace

line: 2, column: 0, length: 3, text: lib
token-type: namespace

line: 2, column: 4, length: 3, text: foo
token-type: function


# lib.py
line: 1, column: 5, length: 6, text: typing
token-type: namespace

line: 1, column: 19, length: 8, text: overload
token-type: function, token-modifiers: [defaultLibrary]

line: 3, column: 1, length: 8, text: overload
token-type: function, token-modifiers: [defaultLibrary]

line: 4, column: 4, length: 3, text: foo
token-type: function

line: 4, column: 8, length: 1, text: a
token-type: parameter

line: 4, column: 11, length: 3, text: str
token-type: class, token-modifiers: [defaultLibrary]

line: 5, column: 1, length: 8, text: overload
token-type: function, token-modifiers: [defaultLibrary]

line: 6, column: 4, length: 3, text: foo
token-type: function

line: 6, column: 8, length: 1, text: a
token-type: parameter

line: 6, column: 11, length: 3, text: int
token-type: class, token-modifiers: [defaultLibrary]

line: 7, column: 4, length: 3, text: foo
token-type: function

line: 7, column: 8, length: 1, text: a
token-type: parameter

line: 7, column: 11, length: 3, text: str
token-type: class, token-modifiers: [defaultLibrary]

line: 7, column: 17, length: 3, text: int
token-type: class, token-modifiers: [defaultLibrary]
"#,
    );
}

#[test]
fn highlights_multibyte_identifiers() {
    let code = r#"
名字 = "值"
print(名字)
"#;

    // NOTE: The identifier "名字" is 2 characters long
    assert_full_semantic_tokens(
        &[("main", code)],
        r#"
# main.py
line: 1, column: 0, length: 2, text: 名字
token-type: variable, token-modifiers: [readonly]

line: 2, column: 0, length: 5, text: print
token-type: function, token-modifiers: [defaultLibrary]

line: 2, column: 6, length: 2, text: 名字
token-type: variable, token-modifiers: [readonly]
"#,
    );
}

#[test]
fn import_with_same_name_alias_test() {
    let lib = r#"
def func():
    pass
"#;
    let code = r#"
from lib import func as func
"#;
    assert_full_semantic_tokens(
        &[("main", code), ("lib", lib)],
        r#"
# main.py
line: 1, column: 5, length: 3, text: lib
token-type: namespace

line: 1, column: 16, length: 4, text: func
token-type: function

line: 1, column: 24, length: 4, text: func
token-type: function


# lib.py
line: 1, column: 4, length: 4, text: func
token-type: function
"#,
    );
}

#[test]
fn import_with_renamed_alias_test() {
    let foo = r#"
def bar():
    pass
"#;
    let code = r#"
from foo import bar as baz
"#;
    assert_full_semantic_tokens(
        &[("main", code), ("foo", foo)],
        r#"
# main.py
line: 1, column: 5, length: 3, text: foo
token-type: namespace

line: 1, column: 16, length: 3, text: bar
token-type: function

line: 1, column: 23, length: 3, text: baz
token-type: function


# foo.py
line: 1, column: 4, length: 3, text: bar
token-type: function
"#,
    );
}

#[test]
fn submodule_attribute_namespace() {
    let mymod_init = r#"# mymod/__init__.py
def version() -> str: ...
"#;
    let mymod_submod_init = r#"# mymod/submod/__init__.py
class Foo: ...
"#;
    let code = r#"
import mymod.submod
mymod.submod.Foo
"#;
    assert_full_semantic_tokens(
        &[
            ("main", code),
            ("mymod", mymod_init),
            ("mymod.submod", mymod_submod_init),
        ],
        r#"
# main.py
line: 1, column: 7, length: 12, text: mymod.submod
token-type: namespace

line: 2, column: 0, length: 5, text: mymod
token-type: namespace

line: 2, column: 6, length: 6, text: submod
token-type: namespace

line: 2, column: 13, length: 3, text: Foo
token-type: class


# mymod.py
line: 1, column: 4, length: 7, text: version
token-type: function

line: 1, column: 17, length: 3, text: str
token-type: class, token-modifiers: [defaultLibrary]


# mymod.submod.py
line: 1, column: 6, length: 3, text: Foo
token-type: class
"#,
    );
}

#[test]
fn submodule_attribute_namespace_when_intermediate_module_missing() {
    // Only mymod.submod exists, not mymod itself
    let mymod_submod_init = r#"# mymod/submod/__init__.py
class Foo: ...
"#;
    let code = r#"
import mymod.submod
mymod.submod.Foo
"#;
    assert_full_semantic_tokens(
        &[("main", code), ("mymod.submod", mymod_submod_init)],
        r#"
# main.py
line: 1, column: 7, length: 12, text: mymod.submod
token-type: namespace

line: 2, column: 6, length: 6, text: submod
token-type: namespace

line: 2, column: 13, length: 3, text: Foo
token-type: class


# mymod.submod.py
line: 1, column: 6, length: 3, text: Foo
token-type: class
"#,
    );
}

#[test]
fn try_nested_statements_test() {
    let code = r#"
try:
    class TryClass:
        def method(self): ...
except Exception as e:
    class ExceptClass: ...
else:
    def else_func(): ...
finally:
    def finally_func(): ...
"#;
    assert_full_semantic_tokens(
        &[("main", code)],
        r#"
# main.py
line: 2, column: 10, length: 8, text: TryClass
token-type: class

line: 3, column: 12, length: 6, text: method
token-type: method

line: 3, column: 19, length: 4, text: self
token-type: parameter

line: 4, column: 7, length: 9, text: Exception
token-type: class, token-modifiers: [defaultLibrary]

line: 4, column: 20, length: 1, text: e
token-type: variable

line: 5, column: 10, length: 11, text: ExceptClass
token-type: class

line: 7, column: 8, length: 9, text: else_func
token-type: function

line: 9, column: 8, length: 12, text: finally_func
token-type: function
"#,
    );
}

#[test]
fn with_nested_statements_test() {
    let code = r#"
with open("foo.txt") as f:
    class WithClass:
        x: int
    def with_func(): ...
"#;
    assert_full_semantic_tokens(
        &[("main", code)],
        r#"
# main.py
line: 1, column: 5, length: 4, text: open
token-type: function, token-modifiers: [defaultLibrary]

line: 1, column: 24, length: 1, text: f
token-type: variable

line: 2, column: 10, length: 9, text: WithClass
token-type: class

line: 3, column: 8, length: 1, text: x
token-type: variable

line: 3, column: 11, length: 3, text: int
token-type: class, token-modifiers: [defaultLibrary]

line: 4, column: 8, length: 9, text: with_func
token-type: function
"#,
    );
}
