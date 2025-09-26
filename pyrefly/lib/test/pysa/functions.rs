/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use pretty_assertions::assert_eq;
use pyrefly_types::class::ClassType;
use pyrefly_types::types::Type;

use crate::report::pysa::FunctionBaseDefinition;
use crate::report::pysa::FunctionDefinition;
use crate::report::pysa::FunctionParameter;
use crate::report::pysa::FunctionParameters;
use crate::report::pysa::FunctionSignature;
use crate::report::pysa::ModuleContext;
use crate::report::pysa::ModuleIds;
use crate::report::pysa::PysaType;
use crate::report::pysa::ScopeParent;
use crate::report::pysa::WholeProgramReversedOverrideGraph;
use crate::report::pysa::add_undecorated_signatures;
use crate::report::pysa::export_all_functions;
use crate::test::pysa::utils::create_location;
use crate::test::pysa::utils::create_state;
use crate::test::pysa::utils::get_class;
use crate::test::pysa::utils::get_class_ref;
use crate::test::pysa::utils::get_handle_for_module_name;

fn create_function_definition(
    name: &str,
    parent: ScopeParent,
    undecorated_signatures: Vec<FunctionSignature>,
) -> FunctionDefinition {
    FunctionDefinition {
        base: FunctionBaseDefinition {
            name: name.to_owned(),
            parent,
            is_overload: false,
            is_staticmethod: false,
            is_classmethod: false,
            is_property_getter: false,
            is_property_setter: false,
            is_stub: false,
            defining_class: None,
            overridden_base_method: None,
        },
        undecorated_signatures,
    }
}

fn create_simple_signature(
    parameters: Vec<FunctionParameter>,
    return_annotation: PysaType,
) -> FunctionSignature {
    FunctionSignature {
        parameters: FunctionParameters::List(parameters),
        return_annotation,
    }
}

fn test_exported_functions(
    module_name: &str,
    python_code: &str,
    create_expected_function_definitions: &dyn Fn(&ModuleContext) -> Vec<FunctionDefinition>,
) {
    let state = create_state(module_name, python_code);
    let transaction = state.transaction();
    let handles = transaction.handles();
    let module_ids = ModuleIds::new(&handles);

    let test_module_handle = get_handle_for_module_name(module_name, &transaction);

    let context = ModuleContext::create(&test_module_handle, &transaction, &module_ids).unwrap();

    let expected_function_definitions = create_expected_function_definitions(&context);

    let reversed_override_graph = WholeProgramReversedOverrideGraph::new();
    let actual_function_definitions = add_undecorated_signatures(
        &export_all_functions(&reversed_override_graph, &context),
        &context,
    );

    // Sort definitions by function Id.
    let mut actual_function_definitions = actual_function_definitions
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect::<Vec<_>>();
    actual_function_definitions.sort_by_key(|(function_id, _)| function_id.clone());
    let actual_function_definitions = actual_function_definitions
        .into_iter()
        .map(|(_, function_definition)| function_definition)
        .collect::<Vec<_>>();

    assert_eq!(expected_function_definitions, actual_function_definitions);
}

#[macro_export]
macro_rules! exported_functions_testcase {
    ($name:ident, $code:literal, $expected:expr,) => {
        #[test]
        fn $name() {
            $crate::test::pysa::functions::test_exported_functions("test", $code, $expected);
        }
    };
}

#[macro_export]
macro_rules! exported_function_testcase {
    ($name:ident, $code:literal, $expected:expr,) => {
        #[test]
        fn $name() {
            let expected_closure = $expected;
            $crate::test::pysa::functions::test_exported_functions(
                "test",
                $code,
                &|context: &ModuleContext| vec![expected_closure(context)],
            );
        }
    };
}

exported_function_testcase!(
    test_export_simple_function,
    r#"
def foo(x: int) -> str:
    return ""
"#,
    &|context: &ModuleContext| {
        create_function_definition(
            "foo",
            ScopeParent::TopLevel,
            /* overloads */
            vec![create_simple_signature(
                vec![FunctionParameter::Pos {
                    name: "x".to_owned(),
                    annotation: PysaType::from_class_type(context.stdlib.int(), context),
                    required: true,
                }],
                PysaType::from_class_type(context.stdlib.str(), context),
            )],
        )
    },
);

exported_function_testcase!(
    test_export_function_with_various_parameters,
    r#"
def complex_function(pos_arg: int, /, pos_or_kw: str, *args: float, kw_only: bool, **kwargs: int) -> None:
    pass
"#,
    &|context: &ModuleContext| {
        create_function_definition(
            "complex_function",
            ScopeParent::TopLevel,
            /* overloads */
            vec![create_simple_signature(
                vec![
                    FunctionParameter::PosOnly {
                        name: Some("pos_arg".to_owned()),
                        annotation: PysaType::from_class_type(context.stdlib.int(), context),
                        required: true,
                    },
                    FunctionParameter::Pos {
                        name: "pos_or_kw".to_owned(),
                        annotation: PysaType::from_class_type(context.stdlib.str(), context),
                        required: true,
                    },
                    FunctionParameter::VarArg {
                        name: Some("args".to_owned()),
                        annotation: PysaType::from_class_type(context.stdlib.float(), context),
                    },
                    FunctionParameter::KwOnly {
                        name: "kw_only".to_owned(),
                        annotation: PysaType::from_class_type(context.stdlib.bool(), context),
                        required: true,
                    },
                    FunctionParameter::Kwargs {
                        name: Some("kwargs".to_owned()),
                        annotation: PysaType::from_class_type(context.stdlib.int(), context),
                    },
                ],
                PysaType::none(),
            )],
        )
    },
);

exported_function_testcase!(
    test_export_method,
    r#"
class MyClass:
    def method(self):
        pass
"#,
    &|context: &ModuleContext| {
        create_function_definition(
            "method",
            ScopeParent::Class {
                location: create_location(2, 7, 2, 14),
            },
            /* overloads */
            vec![create_simple_signature(
                vec![FunctionParameter::Pos {
                    name: "self".to_owned(),
                    annotation: PysaType::from_class(
                        &get_class("test", "MyClass", context),
                        context,
                    ),
                    required: true,
                }],
                PysaType::none(),
            )],
        )
        .with_defining_class(get_class_ref("test", "MyClass", context))
    },
);

exported_function_testcase!(
    test_export_staticmethod,
    r#"
class MyClass:
    @staticmethod
    def static_method(x: int) -> str:
        return ""
"#,
    &|context: &ModuleContext| {
        create_function_definition(
            "static_method",
            ScopeParent::Class {
                location: create_location(2, 7, 2, 14),
            },
            /* overloads */
            vec![create_simple_signature(
                vec![FunctionParameter::Pos {
                    name: "x".to_owned(),
                    annotation: PysaType::from_class_type(context.stdlib.int(), context),
                    required: true,
                }],
                PysaType::from_class_type(context.stdlib.str(), context),
            )],
        )
        .with_is_staticmethod(true)
        .with_defining_class(get_class_ref("test", "MyClass", context))
    },
);

exported_function_testcase!(
    test_export_classmethod,
    r#"
class MyClass:
    @classmethod
    def class_method(cls):
        pass
"#,
    &|context: &ModuleContext| {
        create_function_definition(
            "class_method",
            ScopeParent::Class {
                location: create_location(2, 7, 2, 14),
            },
            /* overloads */
            vec![create_simple_signature(
                vec![FunctionParameter::Pos {
                    name: "cls".to_owned(),
                    annotation: PysaType::from_type(
                        &Type::Type(Box::new(Type::ClassType(ClassType::new(
                            get_class("test", "MyClass", context),
                            Default::default(),
                        )))),
                        context,
                    ),
                    required: true,
                }],
                PysaType::none(),
            )],
        )
        .with_is_classmethod(true)
        .with_defining_class(get_class_ref("test", "MyClass", context))
    },
);

exported_function_testcase!(
    test_export_type_overloads,
    r#"
from typing import overload
@overload
def foo(x: int) -> int:
    ...
@overload
def foo(x: str) -> str:
    ...
def foo(x: str | int) -> str | int:
    return x
"#,
    &|context: &ModuleContext| {
        create_function_definition(
            "foo",
            ScopeParent::TopLevel,
            /* overloads */
            vec![
                create_simple_signature(
                    vec![FunctionParameter::Pos {
                        name: "x".to_owned(),
                        annotation: PysaType::from_class_type(context.stdlib.int(), context),
                        required: true,
                    }],
                    PysaType::from_class_type(context.stdlib.int(), context),
                ),
                create_simple_signature(
                    vec![FunctionParameter::Pos {
                        name: "x".to_owned(),
                        annotation: PysaType::from_class_type(context.stdlib.str(), context),
                        required: true,
                    }],
                    PysaType::from_class_type(context.stdlib.str(), context),
                ),
            ],
        )
    },
);

exported_function_testcase!(
    test_export_stub,
    r#"
def foo() -> None:
    ...
"#,
    &|_: &ModuleContext| {
        create_function_definition(
            "foo",
            ScopeParent::TopLevel,
            /* overloads */
            vec![create_simple_signature(vec![], PysaType::none())],
        )
        .with_is_stub(true)
    },
);

exported_functions_testcase!(
    test_export_property_getter_setter,
    r#"
class MyClass:
    @property
    def foo(self) -> int:
        return 0
    @foo.setter
    def foo(self, value: int) -> None:
        return
"#,
    &|context: &ModuleContext| {
        vec![
            create_function_definition(
                "foo",
                ScopeParent::Class {
                    location: create_location(2, 7, 2, 14),
                },
                /* overloads */
                vec![create_simple_signature(
                    vec![FunctionParameter::Pos {
                        name: "self".to_owned(),
                        annotation: PysaType::from_class(
                            &get_class("test", "MyClass", context),
                            context,
                        ),
                        required: true,
                    }],
                    PysaType::from_class_type(context.stdlib.int(), context),
                )],
            )
            .with_is_property_getter(true)
            .with_defining_class(get_class_ref("test", "MyClass", context)),
            create_function_definition(
                "foo",
                ScopeParent::Class {
                    location: create_location(2, 7, 2, 14),
                },
                /* overloads */
                vec![create_simple_signature(
                    vec![
                        FunctionParameter::Pos {
                            name: "self".to_owned(),
                            annotation: PysaType::from_class(
                                &get_class("test", "MyClass", context),
                                context,
                            ),
                            required: true,
                        },
                        FunctionParameter::Pos {
                            name: "value".to_owned(),
                            annotation: PysaType::from_class_type(context.stdlib.int(), context),
                            required: true,
                        },
                    ],
                    PysaType::none(),
                )],
            )
            .with_is_property_setter(true)
            .with_defining_class(get_class_ref("test", "MyClass", context)),
        ]
    },
);

exported_functions_testcase!(
    test_export_nested_functions,
    r#"
def foo():
    def bar():
        pass
    return
"#,
    &|_: &ModuleContext| {
        vec![
            create_function_definition(
                "foo",
                ScopeParent::TopLevel,
                /* overloads */
                vec![create_simple_signature(vec![], PysaType::none())],
            ),
            create_function_definition(
                "bar",
                ScopeParent::Function {
                    location: create_location(2, 5, 2, 8),
                },
                /* overloads */
                vec![create_simple_signature(vec![], PysaType::none())],
            ),
        ]
    },
);
