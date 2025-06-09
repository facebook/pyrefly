/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use itertools::Itertools;
use pretty_assertions::assert_eq;
use ruff_text_size::TextSize;

use crate::state::handle::Handle;
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
