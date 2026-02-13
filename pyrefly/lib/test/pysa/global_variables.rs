/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;

use pretty_assertions::assert_eq;
use ruff_python_ast::name::Name;

use crate::report::pysa::context::ModuleContext;
use crate::report::pysa::global_variable::GlobalVariable;
use crate::report::pysa::global_variable::collect_global_variables;
use crate::report::pysa::global_variable::export_global_variables;
use crate::report::pysa::location::PysaLocation;
use crate::report::pysa::module::ModuleIds;
use crate::report::pysa::types::PysaType;
use crate::test::pysa::utils::create_location;
use crate::test::pysa::utils::create_state;
use crate::test::pysa::utils::get_class;
use crate::test::pysa::utils::get_handle_for_module_name;

fn create_global_variable(type_: Option<PysaType>, location: PysaLocation) -> GlobalVariable {
    GlobalVariable { type_, location }
}

fn test_exported_global_variables(
    module_name: &str,
    python_code: &str,
    create_expected_globals: &dyn Fn(&ModuleContext) -> HashMap<Name, GlobalVariable>,
) {
    let state = create_state(module_name, python_code);
    let transaction = state.transaction();
    let handles = transaction.handles();
    let module_ids = ModuleIds::new(&handles);

    let test_module_handle = get_handle_for_module_name(module_name, &transaction);

    let context = ModuleContext::create(test_module_handle, &transaction, &module_ids).unwrap();

    let expected_globals = create_expected_globals(&context);

    let whole_program_global_variables =
        collect_global_variables(&handles, &transaction, &module_ids);
    let actual_globals = export_global_variables(&whole_program_global_variables, &context);

    assert_eq!(expected_globals, actual_globals);
}

#[macro_export]
macro_rules! exported_global_variables_testcase {
    ($name:ident, $code:literal, $expected:expr,) => {
        #[test]
        fn $name() {
            $crate::test::pysa::global_variables::test_exported_global_variables(
                "test", $code, $expected,
            );
        }
    };
}

exported_global_variables_testcase!(
    test_export_simple_global_variable,
    r#"
x: int = 42
"#,
    &|context: &ModuleContext| {
        HashMap::from([(
            "x".into(),
            create_global_variable(
                Some(PysaType::from_class_type(context.stdlib.int(), context)),
                create_location(2, 1, 2, 2),
            ),
        )])
    },
);

exported_global_variables_testcase!(
    test_export_untyped_global_variable,
    r#"
y = "hello"
"#,
    &|context: &ModuleContext| {
        HashMap::from([(
            "y".into(),
            create_global_variable(
                Some(PysaType::from_class_type(context.stdlib.str(), context)),
                create_location(2, 1, 2, 2),
            ),
        )])
    },
);

exported_global_variables_testcase!(
    test_export_none_global_variable,
    r#"
z = None
"#,
    &|_: &ModuleContext| {
        HashMap::from([(
            "z".into(),
            create_global_variable(Some(PysaType::none()), create_location(2, 1, 2, 2)),
        )])
    },
);

exported_global_variables_testcase!(
    test_export_multiple_global_variables,
    r#"
a: int = 1
b: str = "test"
c = 3.14
"#,
    &|context: &ModuleContext| {
        HashMap::from([
            (
                "a".into(),
                create_global_variable(
                    Some(PysaType::from_class_type(context.stdlib.int(), context)),
                    create_location(2, 1, 2, 2),
                ),
            ),
            (
                "b".into(),
                create_global_variable(
                    Some(PysaType::from_class_type(context.stdlib.str(), context)),
                    create_location(3, 1, 3, 2),
                ),
            ),
            (
                "c".into(),
                create_global_variable(
                    Some(PysaType::from_class_type(context.stdlib.float(), context)),
                    create_location(4, 1, 4, 2),
                ),
            ),
        ])
    },
);

exported_global_variables_testcase!(
    test_export_global_variables_with_tuple_unpacking,
    r#"
x, y = 1, "hello"
"#,
    &|context: &ModuleContext| {
        HashMap::from([
            (
                "x".into(),
                create_global_variable(
                    Some(PysaType::from_class_type(context.stdlib.int(), context)),
                    create_location(2, 1, 2, 2),
                ),
            ),
            (
                "y".into(),
                create_global_variable(
                    Some(PysaType::from_class_type(context.stdlib.str(), context)),
                    create_location(2, 4, 2, 5),
                ),
            ),
        ])
    },
);

exported_global_variables_testcase!(
    test_ignore_local_variables_in_functions,
    r#"
global_var = 42

def my_function():
    local_var = "hello"
    return local_var
"#,
    &|context: &ModuleContext| {
        HashMap::from([(
            "global_var".into(),
            create_global_variable(
                Some(PysaType::from_class_type(context.stdlib.int(), context)),
                create_location(2, 1, 2, 11),
            ),
        )])
    },
);

exported_global_variables_testcase!(
    test_ignore_class_attributes,
    r#"
global_var = 100

class MyClass:
    class_attr = "not a global"
"#,
    &|context: &ModuleContext| {
        HashMap::from([(
            "global_var".into(),
            create_global_variable(
                Some(PysaType::from_class_type(context.stdlib.int(), context)),
                create_location(2, 1, 2, 11),
            ),
        )])
    },
);

exported_global_variables_testcase!(
    test_export_augmented_assignment,
    r#"
counter = 0
counter += 1
"#,
    &|context: &ModuleContext| {
        HashMap::from([(
            "counter".into(),
            create_global_variable(
                Some(PysaType::from_class_type(context.stdlib.int(), context)),
                create_location(2, 1, 2, 8),
            ),
        )])
    },
);

exported_global_variables_testcase!(
    test_type_variables_not_exported,
    r#"
import typing
T = typing.TypeVar("T")
"#,
    &|_: &ModuleContext| { HashMap::new() },
);

exported_global_variables_testcase!(
    test_callables_not_exported,
    r#"
class A:
    def __init__(self) -> None:
        pass
    def foo(self) -> None:
        pass
a = A()
foo = a.foo
"#,
    &|context: &ModuleContext| {
        HashMap::from([(
            "a".into(),
            create_global_variable(
                Some(PysaType::from_class(
                    &get_class("test", "A", context),
                    context,
                )),
                create_location(7, 1, 7, 2),
            ),
        )])
    },
);
