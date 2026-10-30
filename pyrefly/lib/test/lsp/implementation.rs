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
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;
use vec1::Vec1;

use crate::state::lsp::FindPreference;
use crate::state::state::State;
use crate::test::util::TestEnv;
use crate::test::util::code_frame_of_source_at_range;
use crate::test::util::get_batched_lsp_operations_report;

fn get_implementations_report(state: &State, handle: &Handle, position: TextSize) -> String {
    // Use the new cancellable transaction infrastructure
    let mut transaction = state.cancellable_transaction();

    // First find the definition at the position
    let Some(def_item) = transaction
        .as_ref()
        .find_definition(handle, position, FindPreference::default())
        .map(Vec1::into_vec)
        .unwrap_or_default()
        .into_iter()
        .next()
    else {
        return "Implementation Result: None".to_owned();
    };

    let definition = TextRangeWithModule::new(def_item.module.clone(), def_item.definition_range);

    let impls = match transaction
        .find_global_implementations_from_definition(*handle.sys_info(), definition)
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

#[test]
fn override_code_lens_batches_methods_across_modules() {
    let base = "class Base:\n    def f(self): pass\n    def g(self): pass\n";
    let child = "from base import Base\nclass Child(Base):\n    def f(self): pass\n";
    let mut env = TestEnv::new();
    env.add("base", base);
    env.add("child", child);
    let (state, handle) = env.to_state();
    let ranges = ["f(self)", "g(self)"].map(|name| {
        let start = TextSize::new(base.find(name).unwrap() as u32);
        TextRange::new(start, start + TextSize::new(1))
    });
    let targets = state
        .cancellable_transaction()
        .override_code_lens_targets(&handle("base"), &ranges)
        .unwrap();
    assert_eq!(targets.len(), 2);
    assert!(targets.iter().all(|target| target.parents.is_empty()));
    assert_eq!(targets[0].implementations.len(), 1);
    let implementation = &targets[0].implementations[0];
    assert_eq!(implementation.module.name().as_str(), "child");
    assert_eq!(implementation.module.code_at(implementation.range), "f");
    assert!(targets[1].implementations.is_empty());
}

#[test]
fn override_code_lens_batches_methods_without_overrides() {
    let mut code = "class C:\n".to_owned();
    for i in 0..200 {
        code.push_str(&format!("    def method_{i}(self): pass\n"));
    }
    let ranges = (0..200)
        .map(|i| {
            let name = format!("method_{i}");
            let start = TextSize::new(code.find(&format!("{name}(self)")).unwrap() as u32);
            TextRange::new(start, start + TextSize::new(name.len() as u32))
        })
        .collect::<Vec<_>>();
    let mut env = TestEnv::new();
    env.add("main", &code);
    let (state, handle) = env.to_state();
    let mut transaction = state.cancellable_transaction();
    let targets = transaction
        .override_code_lens_targets(&handle("main"), &ranges)
        .unwrap();
    assert_eq!(targets.len(), 200);
    assert!(
        targets
            .iter()
            .all(|target| target.parents.is_empty() && target.implementations.is_empty())
    );
    transaction.get_cancellation_handle().cancel();
    assert!(
        transaction
            .override_code_lens_targets(&handle("main"), &ranges)
            .is_err()
    );
}
