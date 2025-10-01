/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;

use pretty_assertions::assert_eq;
use pyrefly_types::class::ClassType;
use pyrefly_types::types::Type;

use crate::report::pysa::class::ClassDefinition;
use crate::report::pysa::class::ClassId;
use crate::report::pysa::class::PysaClassField;
use crate::report::pysa::class::PysaClassMro;
use crate::report::pysa::class::export_all_classes;
use crate::report::pysa::context::ModuleContext;
use crate::report::pysa::module::ModuleIds;
use crate::report::pysa::scope::ScopeParent;
use crate::report::pysa::types::PysaType;
use crate::test::pysa::utils::create_location;
use crate::test::pysa::utils::create_state;
use crate::test::pysa::utils::get_class;
use crate::test::pysa::utils::get_class_ref;
use crate::test::pysa::utils::get_handle_for_module_name;

fn create_simple_class(name: &str, id: u32, parent: ScopeParent) -> ClassDefinition {
    ClassDefinition {
        class_id: ClassId::from_int(id),
        name: name.to_owned(),
        bases: Vec::new(),
        mro: PysaClassMro::Resolved(Vec::new()),
        parent,
        is_synthesized: false,
        fields: HashMap::new(),
    }
}

fn test_exported_classes(
    module_name: &str,
    python_code: &str,
    create_expected_class_definitions: &dyn Fn(&ModuleContext) -> Vec<ClassDefinition>,
) {
    let state = create_state(module_name, python_code);
    let transaction = state.transaction();
    let handles = transaction.handles();
    let module_ids = ModuleIds::new(&handles);

    let test_module_handle = get_handle_for_module_name(module_name, &transaction);

    let context = ModuleContext::create(&test_module_handle, &transaction, &module_ids).unwrap();

    let expected_class_definitions = create_expected_class_definitions(&context);

    let actual_class_definitions = export_all_classes(&context);

    // Sort definitions by location.
    let mut actual_class_definitions = actual_class_definitions.into_iter().collect::<Vec<_>>();
    actual_class_definitions.sort_by_key(|(location, _)| location.clone());
    let actual_class_definitions = actual_class_definitions
        .into_iter()
        .map(|(_, class_definition)| class_definition)
        .collect::<Vec<_>>();

    assert_eq!(expected_class_definitions, actual_class_definitions);
}

#[macro_export]
macro_rules! exported_classes_testcase {
    ($name:ident, $code:literal, $expected:expr,) => {
        #[test]
        fn $name() {
            $crate::test::pysa::classes::test_exported_classes("test", $code, $expected);
        }
    };
}

#[macro_export]
macro_rules! exported_class_testcase {
    ($name:ident, $code:literal, $expected:expr,) => {
        #[test]
        fn $name() {
            let expected_closure = $expected;
            $crate::test::pysa::classes::test_exported_classes(
                "test",
                $code,
                &|context: &ModuleContext| vec![expected_closure(context)],
            );
        }
    };
}

exported_class_testcase!(
    test_export_simple_class,
    r#"
class Foo:
    pass
"#,
    &|_: &ModuleContext| { create_simple_class("Foo", 0, ScopeParent::TopLevel) },
);

exported_classes_testcase!(
    test_export_simple_derived_class,
    r#"
class Foo:
    pass
class Bar(Foo):
    pass
"#,
    &|context: &ModuleContext| {
        vec![
            create_simple_class("Foo", 0, ScopeParent::TopLevel),
            create_simple_class("Bar", 1, ScopeParent::TopLevel)
                .with_bases(vec![get_class_ref("test", "Foo", context)])
                .with_mro(PysaClassMro::Resolved(vec![get_class_ref(
                    "test", "Foo", context,
                )])),
        ]
    },
);

exported_classes_testcase!(
    test_export_multiple_inheritance,
    r#"
class A:
    pass
class B:
    pass
class C(A, B):
    pass
"#,
    &|context: &ModuleContext| {
        vec![
            create_simple_class("A", 0, ScopeParent::TopLevel),
            create_simple_class("B", 1, ScopeParent::TopLevel),
            create_simple_class("C", 2, ScopeParent::TopLevel)
                .with_bases(vec![
                    get_class_ref("test", "A", context),
                    get_class_ref("test", "B", context),
                ])
                .with_mro(PysaClassMro::Resolved(vec![
                    get_class_ref("test", "A", context),
                    get_class_ref("test", "B", context),
                ])),
        ]
    },
);

exported_classes_testcase!(
    test_export_diamond_inheritance,
    r#"
class A:
    pass
class B(A):
    pass
class C(A):
    pass
class D(B, C):
    pass
"#,
    &|context: &ModuleContext| {
        vec![
            create_simple_class("A", 0, ScopeParent::TopLevel),
            create_simple_class("B", 1, ScopeParent::TopLevel)
                .with_bases(vec![get_class_ref("test", "A", context)])
                .with_mro(PysaClassMro::Resolved(vec![get_class_ref(
                    "test", "A", context,
                )])),
            create_simple_class("C", 2, ScopeParent::TopLevel)
                .with_bases(vec![get_class_ref("test", "A", context)])
                .with_mro(PysaClassMro::Resolved(vec![get_class_ref(
                    "test", "A", context,
                )])),
            create_simple_class("D", 3, ScopeParent::TopLevel)
                .with_bases(vec![
                    get_class_ref("test", "B", context),
                    get_class_ref("test", "C", context),
                ])
                .with_mro(PysaClassMro::Resolved(vec![
                    get_class_ref("test", "B", context),
                    get_class_ref("test", "C", context),
                    get_class_ref("test", "A", context),
                ])),
        ]
    },
);

exported_classes_testcase!(
    test_export_nested_classes,
    r#"
class Foo:
    class Bar:
        pass
"#,
    &|context: &ModuleContext| {
        vec![
            create_simple_class("Foo", 0, ScopeParent::TopLevel).with_fields(HashMap::from([(
                "Bar".to_owned(),
                PysaClassField {
                    type_: PysaType::from_type(
                        &Type::Type(Box::new(Type::ClassType(ClassType::new(
                            get_class("test", "Bar", context),
                            Default::default(),
                        )))),
                        context,
                    ),
                    explicit_annotation: None,
                    location: Some(create_location(3, 11, 3, 14)),
                },
            )])),
            create_simple_class(
                "Bar",
                1,
                ScopeParent::Class {
                    location: create_location(2, 7, 2, 10),
                },
            ),
        ]
    },
);

exported_class_testcase!(
    test_export_class_nested_in_function,
    r#"
def foo():
    class Foo:
        pass
    return Foo
"#,
    &|_: &ModuleContext| {
        create_simple_class(
            "Foo",
            0,
            ScopeParent::Function {
                location: create_location(2, 5, 2, 8),
            },
        )
    },
);

exported_class_testcase!(
    test_export_namedtuple_class,
    r#"
from collections import namedtuple
Point = namedtuple('Point', ['x', 'y'])
"#,
    &|context: &ModuleContext| {
        ClassDefinition {
            class_id: ClassId::from_int(0),
            name: "Point".to_owned(),
            bases: vec![get_class_ref(
                "_typeshed._type_checker_internals",
                "NamedTupleFallback",
                context,
            )],
            mro: PysaClassMro::Resolved(vec![
                get_class_ref(
                    "_typeshed._type_checker_internals",
                    "NamedTupleFallback",
                    context,
                ),
                get_class_ref("builtins", "tuple", context),
                get_class_ref("typing", "Sequence", context),
                get_class_ref("typing", "Reversible", context),
                get_class_ref("typing", "Collection", context),
                get_class_ref("typing", "Iterable", context),
                get_class_ref("typing", "Container", context),
            ]),
            parent: ScopeParent::TopLevel,
            is_synthesized: true,
            fields: HashMap::from([
                (
                    "x".to_owned(),
                    PysaClassField {
                        type_: PysaType::any_implicit(),
                        explicit_annotation: None,
                        location: Some(create_location(3, 30, 3, 33)),
                    },
                ),
                (
                    "y".to_owned(),
                    PysaClassField {
                        type_: PysaType::any_implicit(),
                        explicit_annotation: None,
                        location: Some(create_location(3, 35, 3, 38)),
                    },
                ),
            ]),
        }
    },
);

exported_class_testcase!(
    test_export_class_fields_declared_by_annotation,
    r#"
import typing
class Foo:
    x: int
    y: str
    z: typing.Annotated[bool, "annotation for z"]
"#,
    &|context: &ModuleContext| {
        create_simple_class("Foo", 0, ScopeParent::TopLevel).with_fields(HashMap::from([
            (
                "x".to_owned(),
                PysaClassField {
                    type_: PysaType::from_class_type(context.stdlib.int(), context),
                    explicit_annotation: Some("int".to_owned()),
                    location: Some(create_location(4, 5, 4, 6)),
                },
            ),
            (
                "y".to_owned(),
                PysaClassField {
                    type_: PysaType::from_class_type(context.stdlib.str(), context),
                    explicit_annotation: Some("str".to_owned()),
                    location: Some(create_location(5, 5, 5, 6)),
                },
            ),
            (
                "z".to_owned(),
                PysaClassField {
                    type_: PysaType::from_class_type(context.stdlib.bool(), context),
                    explicit_annotation: Some(
                        "typing.Annotated[bool, \"annotation for z\"]".to_owned(),
                    ),
                    location: Some(create_location(6, 5, 6, 6)),
                },
            ),
        ]))
    },
);

exported_class_testcase!(
    test_export_class_fields_assigned_in_body,
    r#"
import typing
class Foo:
    def __init__(self, x: int, y: str, z: bool) -> None:
        self.x: int = x
        self.y: str = y
        self.z: typing.Annotated[bool, "annotation for z"] = z
"#,
    &|context: &ModuleContext| {
        create_simple_class("Foo", 0, ScopeParent::TopLevel).with_fields(HashMap::from([
            (
                "x".to_owned(),
                PysaClassField {
                    type_: PysaType::from_class_type(context.stdlib.int(), context),
                    explicit_annotation: Some("int".to_owned()),
                    location: Some(create_location(5, 14, 5, 15)),
                },
            ),
            (
                "y".to_owned(),
                PysaClassField {
                    type_: PysaType::from_class_type(context.stdlib.str(), context),
                    explicit_annotation: Some("str".to_owned()),
                    location: Some(create_location(6, 14, 6, 15)),
                },
            ),
            (
                "z".to_owned(),
                PysaClassField {
                    type_: PysaType::from_class_type(context.stdlib.bool(), context),
                    explicit_annotation: Some(
                        "typing.Annotated[bool, \"annotation for z\"]".to_owned(),
                    ),
                    location: Some(create_location(7, 14, 7, 15)),
                },
            ),
        ]))
    },
);

exported_class_testcase!(
    test_export_dataclass,
    r#"
from dataclasses import dataclass
@dataclass
class Foo:
    x: int
    y: str
"#,
    &|context: &ModuleContext| {
        create_simple_class("Foo", 0, ScopeParent::TopLevel).with_fields(HashMap::from([
            (
                "x".to_owned(),
                PysaClassField {
                    type_: PysaType::from_class_type(context.stdlib.int(), context),
                    explicit_annotation: Some("int".to_owned()),
                    location: Some(create_location(5, 5, 5, 6)),
                },
            ),
            (
                "y".to_owned(),
                PysaClassField {
                    type_: PysaType::from_class_type(context.stdlib.str(), context),
                    explicit_annotation: Some("str".to_owned()),
                    location: Some(create_location(6, 5, 6, 6)),
                },
            ),
        ]))
    },
);
