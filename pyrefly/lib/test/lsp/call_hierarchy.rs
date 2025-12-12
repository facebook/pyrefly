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

fn format_call_site(
    label: &str,
    name: &str,
    source: &str,
    call_range: ruff_text_size::TextRange,
) -> String {
    format!(
        "{}: {}\nCall site:\n{}",
        label,
        name,
        code_frame_of_source_at_range(source, call_range),
    )
}

fn get_callers_report(state: &State, handle: &Handle, position: TextSize) -> String {
    let mut transaction = state.cancellable_transaction();

    let Some(def_item) = transaction
        .as_ref()
        .find_definition(handle, position, FindPreference::default())
        .into_iter()
        .next()
    else {
        return "Callers Result: None".to_owned();
    };

    let definition = TextRangeWithModule::new(def_item.module.clone(), def_item.definition_range);

    let callers = match transaction.find_global_incoming_calls_from_function_definition(
        handle.sys_info(),
        def_item.metadata.clone(),
        &definition,
    ) {
        Ok(callers) => callers,
        Err(_) => return "Callers Result: Cancelled".to_owned(),
    };

    if !callers.is_empty() {
        callers
            .into_iter()
            .flat_map(|(module_info, calls)| {
                calls
                    .into_iter()
                    .map(move |(call_range, caller_name, _caller_range)| {
                        format_call_site("Caller", &caller_name, module_info.contents(), call_range)
                    })
            })
            .join("\n")
    } else {
        "Callers Result: None".to_owned()
    }
}

fn get_callees_report(state: &State, handle: &Handle, position: TextSize) -> String {
    let mut transaction = state.cancellable_transaction();

    let callees =
        match transaction.find_global_outgoing_calls_from_function_definition(handle, position) {
            Ok(callees) => callees,
            Err(_) => return "Callees Result: Cancelled".to_owned(),
        };

    if !callees.is_empty() {
        let Some(module_info) = transaction.as_ref().get_module_info(handle) else {
            return "Callees Result: None".to_owned();
        };

        callees
            .into_iter()
            .flat_map(|(callee_module, calls)| {
                let source = module_info.contents().to_owned();
                calls
                    .into_iter()
                    .map(move |(call_range, callee_def_range)| {
                        // Construct the qualified name from the module and definition
                        let target_name = callee_module.code_at(callee_def_range);
                        let callee_name = format!("{}.{}", callee_module.name(), target_name);
                        format_call_site("Callee", &callee_name, &source, call_range)
                    })
            })
            .join("\n")
    } else {
        "Callees Result: None".to_owned()
    }
}

#[test]
fn find_callers_simple_test() {
    let code = r#"
def greet(name: str) -> str:
#   ^
    return f"Hello, {name}!"

def test_greet():
    result = greet("World")
    print(result)

def another_caller():
    msg = greet("Alice")
    return msg
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_callers_report);
    assert!(report.contains("# main.py"));
    assert!(report.contains("Caller: main.test_greet"));
    assert!(report.contains("Caller: main.another_caller"));
    assert!(report.contains("greet(\"World\")"));
    assert!(report.contains("greet(\"Alice\")"));
}

#[test]
fn find_callers_no_calls_test() {
    let code = r#"
def unused_function():
#   ^
    return "I'm never called"

def some_other_function():
    return "I do my own thing"
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_callers_report);
    assert_eq!(
        r#"
# main.py
2 | def unused_function():
        ^
Callers Result: None
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn find_callers_nested_calls_test() {
    let code = r#"
def helper() -> int:
#   ^
    return 42

def level1():
    return helper()

def level2():
    value = helper()
    return value * 2

def level3():
    a = helper()
    b = helper()
    return a + b
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_callers_report);
    assert!(report.contains("# main.py"));
    assert!(report.contains("Caller: main.level1"));
    assert!(report.contains("Caller: main.level2"));
    assert!(report.contains("Caller: main.level3"));
    // level3 should have two call sites to helper()
    let level3_count = report.matches("Caller: main.level3").count();
    assert_eq!(level3_count, 2, "level3 should have two calls to helper()");
}

#[test]
fn find_callers_nested_in_other_statements_works_at_definition_test() {
    let code = r#"
def helper() -> int:
#   ^
    return 42

def level1():
    if True:
        helper()
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_callers_report);
    assert!(report.contains("# main.py"));
    assert!(report.contains("Caller: main.level1"));
}

#[test]
fn find_callers_nested_in_other_statements_works_at_call_site_definition_test() {
    let code = r#"
def helper() -> int:
    return 42

def level1():
    if True:
        helper()
#       ^
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_callers_report);
    assert!(report.contains("# main.py"));
    assert!(report.contains("Caller: main.level1"));
}

#[test]
fn find_callers_method_at_definition_test() {
    let code = r#"
class A:
    def greet(self):
#       ^
        print("hello world")

def call_class():
    a = A()
    a.greet()
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_callers_report);
    assert!(report.contains("# main.py"));
    assert!(report.contains("Caller: main.call_class"));
}

#[test]
fn find_callers_method_at_call_site_test() {
    let code = r#"
class A:
    def greet(self):
        print("hello world")

def call_class():
    a = A()
    a.greet()
#     ^
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_callers_report);
    assert!(report.contains("# main.py"));
    assert!(report.contains("Caller: main.call_class"));
}

#[test]
fn find_callees_simple_test() {
    let code = r#"
def greet(name: str) -> str:
    return f"Hello, {name}!"

def print_message(msg: str):
    print(msg)

def test_greet():
#   ^
    result = greet("World")
    print_message(result)
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_callees_report);
    assert!(report.contains("# main.py"));
    assert!(report.contains("Callee: main.greet"));
    assert!(report.contains("Callee: main.print_message"));
    assert!(report.contains("greet(\"World\")"));
    assert!(report.contains("print_message(result)"));
}

#[test]
fn find_callees_no_calls_test() {
    let code = r#"
def standalone_function():
#   ^
    x = 42
    return x * 2
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_callees_report);
    assert_eq!(
        r#"
# main.py
2 | def standalone_function():
        ^
Callees Result: None
"#
        .trim(),
        report.trim(),
    );
}

#[test]
fn find_callees_multiple_calls_test() {
    let code = r#"
def add(a: int, b: int) -> int:
    return a + b

def multiply(a: int, b: int) -> int:
    return a * b

def subtract(a: int, b: int) -> int:
    return a - b

def calculate():
#   ^
    x = add(5, 3)
    y = multiply(x, 2)
    z = subtract(y, 1)
    final = add(z, 10)
    return final
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_callees_report);
    assert!(report.contains("# main.py"));
    assert!(report.contains("Callee: main.add"));
    assert!(report.contains("Callee: main.multiply"));
    assert!(report.contains("Callee: main.subtract"));
    // add() should be called twice
    let add_count = report.matches("Callee: main.add").count();
    assert_eq!(add_count, 2, "add() should be called twice in calculate()");
}

#[test]
fn find_callees_nested_in_other_statements_works_at_definition_test() {
    let code = r#"
def level2():
    print("Hello")

def level1():
#   ^
    if True:
        level2()
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_callees_report);
    assert!(report.contains("# main.py"));
    assert!(report.contains("Callee: main.level2"));
}

#[test]
fn find_callees_nested_in_other_statements_works_at_call_site_definition_test() {
    let code = r#"
def level2():
    print("Hello")

def level1():
    if True:
        level2()

def caller():
    level1()
#   ^
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_callees_report);
    assert!(report.contains("# main.py"));
    assert!(report.contains("Callee: main.level2"));
}

#[test]
fn find_callees_method_at_definition_test() {
    let code = r#"
def greet():
    print("hello world")

class A:
    def other_function(self):
        print("Hello")
    def call_greet(self):
#       ^
        greet()
        self.other_function()
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_callees_report);
    assert!(report.contains("# main.py"));
    assert!(report.contains("Callee: main.greet"));
    assert!(report.contains("Callee: main.other_function"));
}

#[test]
fn find_callees_method_at_call_site_test() {
    let code = r#"
class A:
    def call_greet(self):
        self.other_function()

    def other_function(self):
        print("hello")

a = A()
a.call_greet()
#  ^
"#;
    let report = get_batched_lsp_operations_report(&[("main", code)], get_callees_report);
    assert!(report.contains("# main.py"));
    assert!(report.contains("Callee: main.other_function"));
}
