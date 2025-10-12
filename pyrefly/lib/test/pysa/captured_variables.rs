/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;
use std::collections::HashSet;

use pretty_assertions::assert_eq;
use ruff_python_ast::name::Name;

use crate::report::pysa::captured_variable::CapturedVariable;
use crate::report::pysa::captured_variable::ModuleCapturedVariables;
use crate::report::pysa::captured_variable::export_captured_variables;
use crate::report::pysa::context::ModuleContext;
use crate::report::pysa::module::ModuleIds;
use crate::test::pysa::utils::create_state;
use crate::test::pysa::utils::get_handle_for_module_name;

fn create_captured_variable(name: &str) -> CapturedVariable {
    CapturedVariable {
        name: Name::from(name),
    }
}

fn captured_variables_from_actual(
    captures: ModuleCapturedVariables,
) -> HashMap<Name, HashSet<CapturedVariable>> {
    captures
        .into_iter()
        .map(|(function, captures)| (function.function_name, captures))
        .collect::<HashMap<_, _>>()
}

fn captured_variables_from_expected(
    captures: HashMap<Name, Vec<CapturedVariable>>,
) -> HashMap<Name, HashSet<CapturedVariable>> {
    captures
        .into_iter()
        .map(|(function, captures)| (function, captures.into_iter().collect::<HashSet<_>>()))
        .collect::<HashMap<_, _>>()
}

fn test_exported_captured_variables(
    module_name: &str,
    python_code: &str,
    expected_captures: HashMap<Name, Vec<CapturedVariable>>,
) {
    let state = create_state(module_name, python_code);
    let transaction = state.transaction();
    let handles = transaction.handles();
    let module_ids = ModuleIds::new(&handles);

    let test_module_handle = get_handle_for_module_name(module_name, &transaction);

    let context = ModuleContext::create(test_module_handle, &transaction, &module_ids).unwrap();

    let expected_captures = captured_variables_from_expected(expected_captures);
    let actual_captures = captured_variables_from_actual(export_captured_variables(&context));

    assert_eq!(expected_captures, actual_captures);
}

#[macro_export]
macro_rules! exported_captured_variables_testcase {
    ($name:ident, $code:literal, $expected:expr,) => {
        #[test]
        fn $name() {
            $crate::test::pysa::captured_variables::test_exported_captured_variables(
                "test", $code, $expected,
            );
        }
    };
}

exported_captured_variables_testcase!(
    test_export_simple_captured_variable,
    r#"
def foo():
    x = 0
    def inner():
        print(x)
"#,
    HashMap::from([("inner".into(), vec![create_captured_variable("x")]),]),
);

exported_captured_variables_testcase!(
    test_export_capture_parameter,
    r#"
def foo(x):
    def inner():
        print(x)
"#,
    HashMap::from([("inner".into(), vec![create_captured_variable("x")]),]),
);

exported_captured_variables_testcase!(
    test_export_capture_list_append,
    r#"
def foo():
    x = []
    def inner():
        x.append(1)
"#,
    HashMap::from([("inner".into(), vec![create_captured_variable("x")]),]),
);

exported_captured_variables_testcase!(
    test_export_capture_nested,
    r#"
def foo():
    x = []
    def inner():
        def nested():
            print(x)
"#,
    HashMap::from([("nested".into(), vec![create_captured_variable("x")]),]),
);

exported_captured_variables_testcase!(
    test_export_capture_shadowing,
    r#"
def foo():
    x = []
    def inner(x):
        print(x)
"#,
    HashMap::new(),
);

exported_captured_variables_testcase!(
    test_export_capture_shadowing_parameter,
    r#"
def foo(x):
    def inner(x):
        print(x)
"#,
    HashMap::new(),
);

exported_captured_variables_testcase!(
    test_export_capture_unpack,
    r#"
def foo():
    x, y = [], []
    def inner():
        x.append(1)
        y.append(2)
"#,
    HashMap::from([(
        "inner".into(),
        vec![create_captured_variable("x"), create_captured_variable("y")]
    ),]),
);

exported_captured_variables_testcase!(
    test_export_capture_usage_before_declaration,
    r#"
def foo():
    def inner():
        x.append(1)
    
    x = []
"#,
    HashMap::from([("inner".into(), vec![create_captured_variable("x")]),]),
);

exported_captured_variables_testcase!(
    test_export_capture_nonlocal,
    r#"
def foo():
    x = 1

    def inner():
        nonlocal x
        x = 2
"#,
    HashMap::from([("inner".into(), vec![create_captured_variable("x")]),]),
);

exported_captured_variables_testcase!(
    test_export_capture_nested_nonlocal,
    r#"
def foo():
    x = 1

    def inner():
        def nested():
            nonlocal x
            x = 2
"#,
    HashMap::from([("nested".into(), vec![create_captured_variable("x")]),]),
);

exported_captured_variables_testcase!(
    test_export_capture_parameter_reassigned,
    r#"
def foo(x):
    x = 0

    def inner():
        print(x)
"#,
    HashMap::from([("inner".into(), vec![create_captured_variable("x")]),]),
);

exported_captured_variables_testcase!(
    test_export_captured_variable_with_updates,
    r#"
def foo():
    x = []
    x.append(1)
    print(x)

    def inner():
        print(x)
"#,
    HashMap::from([("inner".into(), vec![create_captured_variable("x")]),]),
);

exported_captured_variables_testcase!(
    test_export_capture_narrowed,
    r#"
def int_or_str(cond) -> int | str:
    return 1 if cond else "1"

def foo(cond):
    x = int_or_str(cond)
    if isinstance(x, str):
        return
    def inner():
        print(x)
"#,
    HashMap::from([("inner".into(), vec![create_captured_variable("x")]),]),
);

exported_captured_variables_testcase!(
    test_export_capture_conditional_definition,
    r#"
def foo(cond):
    if cond:
        x = 1
    else:
        x = 2
    def inner():
        print(x)
"#,
    HashMap::from([("inner".into(), vec![create_captured_variable("x")]),]),
);
