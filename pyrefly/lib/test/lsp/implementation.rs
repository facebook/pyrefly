/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use itertools::Itertools as _;
use pretty_assertions::assert_eq;
use pyrefly_build::handle::Handle;
use pyrefly_python::module::TextRangeWithModule;
use ruff_text_size::TextSize;

use crate::state::lsp::FindPreference;
use crate::state::state::State;
use crate::test::util::code_frame_of_source_at_range;
use crate::test::util::get_batched_lsp_operations_report;

fn get_implementations_report(state: &State, handle: &Handle, position: TextSize) -> String {
    // Use the new cancellable transaction infrastructure
    let mut transaction = state.cancellable_transaction();

    // First find the definition at the position
    let Some(def_item) = transaction
        .as_ref()
        .find_definition(handle, position, FindPreference::default())
        .into_iter()
        .next()
    else {
        return "Implementation Result: None".to_owned();
    };

    let definition = TextRangeWithModule::new(def_item.module.clone(), def_item.definition_range);

    let impls = match transaction
        .find_global_implementations_from_definition(handle.sys_info(), definition)
    {
        Ok(impls) => impls,
        Err(_) => return "Implementation Result: Cancelled".to_owned(),
    };

    if !impls.is_empty() {
        impls
            .into_iter()
            .map(
                |TextRangeWithModule {
                     module: module_info,
                     range,
                 }| {
                    format!(
                        "Implementation Result:\n{}",
                        code_frame_of_source_at_range(module_info.contents(), range)
                    )
                },
            )
            .join("\n")
    } else {
        "Implementation Result: None".to_owned()
    }
}

#[test]
fn go_to_implementations_test() {
    let code = r#"
class Parent:
    def foo(self) -> int:
#       ^
        return 1

class Child1(Parent):
    def foo(self) -> int:
        return 2

class Child2(Parent):
    def foo(self) -> int:
        return 3

class GrandChild(Child1):
    def foo(self) -> int:
        return 4
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_implementations_report);
    assert_eq!(
        r#"
# main.py
3 |     def foo(self) -> int:
            ^
Implementation Result:
8 |     def foo(self) -> int:
            ^^^
Implementation Result:
12 |     def foo(self) -> int:
             ^^^
Implementation Result:
16 |     def foo(self) -> int:
             ^^^
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn go_to_implementations_on_call_test() {
    let code = r#"
class Parent:
    def foo(self) -> int:
        return 1

class Child1(Parent):
    def foo(self) -> int:
        return 2

class Child2(Parent):
    def foo(self) -> int:
        return 3

def test(p: Parent) -> None:
    p.foo()
#     ^
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_implementations_report);
    assert_eq!(
        r#"
# main.py
15 |     p.foo()
           ^
Implementation Result:
7 |     def foo(self) -> int:
            ^^^
Implementation Result:
11 |     def foo(self) -> int:
             ^^^
"#
        .trim(),
        report.trim(),
    );
}
