/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use itertools::Itertools;
use pretty_assertions::assert_eq;
use pyrefly_build::handle::Handle;
use ruff_text_size::TextSize;

use crate::state::state::State;
use crate::test::util::code_frame_of_source_at_range;
use crate::test::util::get_batched_lsp_operations_report;

fn get_test_report(state: &State, handle: &Handle, position: TextSize) -> String {
    let transaction = state.transaction();
    let ranges = transaction.find_local_references(handle, position);
    let module_info = transaction.get_module_info(handle).unwrap();
    format!(
        "References:\n{}",
        ranges
            .into_iter()
            .map(|range| code_frame_of_source_at_range(module_info.contents(), range))
            .join("\n")
    )
}

#[test]
fn no_references_test() {
    let code = r#"
1234567
# ^

"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
2 | 1234567
      ^
References:
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn basic_variables_test() {
    let code = r#"
foo = 3
# ^
foo + 4 + foo
#          ^

def ff():
  foo = 4
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
2 | foo = 3
      ^
References:
2 | foo = 3
    ^^^
4 | foo + 4 + foo
    ^^^
4 | foo + 4 + foo
              ^^^

4 | foo + 4 + foo
               ^
References:
2 | foo = 3
    ^^^
4 | foo + 4 + foo
    ^^^
4 | foo + 4 + foo
              ^^^
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn basic_attribute_test() {
    let code = r#"
class MyClass:
  x = 5
MyClass().x
#         ^
c = MyClass()
c.x
# ^

class AnotherClass:
  x = 5
AnotherClass().x
#              ^

class ExtendsBase(MyClass):
  pass

ExtendsBase().x
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
4 | MyClass().x
              ^
References:
3 |   x = 5
      ^
4 | MyClass().x
              ^
7 | c.x
      ^
18 | ExtendsBase().x
                   ^

7 | c.x
      ^
References:
3 |   x = 5
      ^
4 | MyClass().x
              ^
7 | c.x
      ^
18 | ExtendsBase().x
                   ^

12 | AnotherClass().x
                    ^
References:
11 |   x = 5
       ^
12 | AnotherClass().x
                    ^
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn attribute_reference_triggered_from_attribute_definition_test() {
    let code = r#"
class MyClass:
    attribute = 5
#    ^

    def method(self) -> None:
        self.attribute


obj = MyClass()
print(obj.attribute)
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
3 |     attribute = 5
         ^
References:
3 |     attribute = 5
        ^^^^^^^^^
7 |         self.attribute
                 ^^^^^^^^^
11 | print(obj.attribute)
               ^^^^^^^^^
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn synthetic_reference_regression_test() {
    let code = r#"
foo = 3
# ^
try:
  pass
except Exception:
  foo
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
2 | foo = 3
      ^
References:
2 | foo = 3
    ^^^
7 |   foo
      ^^^
"#
        .trim(),
        report.trim(),
    );
}

// TODO: references on constructors
#[test]
fn dunder_init() {
    let code = r#"
class Foo:
    def __init__(self): ...
    #   ^
    
Foo()
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
3 |     def __init__(self): ...
            ^
References:
3 |     def __init__(self): ...
            ^^^^^^^^
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn narrowing() {
    let code = r#"
xyz = "test"
# ^
if isinstance(xyz, int):
    print(xyz)
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
2 | xyz = "test"
      ^
References:
2 | xyz = "test"
    ^^^
4 | if isinstance(xyz, int):
                  ^^^
5 |     print(xyz)
              ^^^
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn method() {
    let code = r#"
class C:
    def f(self) -> str:
    #   ^
        ...

    def g(self) -> str:
        return self.f()
        #           ^
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
3 |     def f(self) -> str:
            ^
References:
3 |     def f(self) -> str:
            ^
8 |         return self.f()
                        ^

8 |         return self.f()
                        ^
References:
3 |     def f(self) -> str:
            ^
8 |         return self.f()
                        ^
"#
        .trim(),
        report.trim(),
    );
}

// todo(kylei): reference on parent method should find child method
#[test]
fn child_method() {
    let code = r#"
class C:
    def f(self) -> str: ...
    #   ^

class Concrete(C):
    def f(self) -> str:
        return "test"
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
3 |     def f(self) -> str: ...
            ^
References:
3 |     def f(self) -> str: ...
            ^
"#
        .trim(),
        report.trim(),
    );
}

// todo(kylei): reference on child method should find parent method
#[test]
fn parent_method() {
    let code = r#"
class C:
    def f(self) -> str: ...
    
class Concrete(C):
    def f(self) -> str:
    #   ^
        return "test"
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
6 |     def f(self) -> str:
            ^
References:
6 |     def f(self) -> str:
            ^
"#
        .trim(),
        report.trim(),
    );
}

// todo(kylei): reference on child method should find parent + other child methods
#[test]
fn sibling_method() {
    let code = r#"
class C:
    def f(self) -> str: ...
    
class Concrete(C):
    def f(self) -> str:
    #   ^
        return "test"

class Concrete2(C):
    def f(self) -> str:
        return "test"

"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
6 |     def f(self) -> str:
            ^
References:
6 |     def f(self) -> str:
            ^
"#
        .trim(),
        report.trim(),
    );
}

// todo(kylei): references on renames should find lhs of assignment
#[test]
fn reassigned_local() {
    let code = r#"
xy = 5
#^
xy = xy + 1
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_test_report);
    assert_eq!(
        r#"
# main.py
2 | xy = 5
     ^
References:
2 | xy = 5
    ^^
4 | xy = xy + 1
         ^^
"#
        .trim(),
        report.trim(),
    );
}
