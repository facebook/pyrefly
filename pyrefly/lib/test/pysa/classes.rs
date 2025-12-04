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

use crate::report::pysa::call_graph::Target;
use crate::report::pysa::class::ClassDefinition;
use crate::report::pysa::class::ClassId;
use crate::report::pysa::class::PysaClassField;
use crate::report::pysa::class::PysaClassFieldDeclaration;
use crate::report::pysa::class::PysaClassMro;
use crate::report::pysa::class::export_all_classes;
use crate::report::pysa::context::ModuleContext;
use crate::report::pysa::function::collect_function_base_definitions;
use crate::report::pysa::module::ModuleIds;
use crate::report::pysa::override_graph::WholeProgramReversedOverrideGraph;
use crate::report::pysa::scope::ScopeParent;
use crate::report::pysa::types::PysaType;
use crate::test::pysa::utils::create_location;
use crate::test::pysa::utils::create_state;
use crate::test::pysa::utils::get_class;
use crate::test::pysa::utils::get_class_ref;
use crate::test::pysa::utils::get_function_ref;
use crate::test::pysa::utils::get_handle_for_module_name;

fn create_simple_class(name: &str, id: u32, parent: ScopeParent) -> ClassDefinition {
    ClassDefinition {
        class_id: ClassId::from_int(id),
        name: name.to_owned(),
        bases: Vec::new(),
        mro: PysaClassMro::Resolved(Vec::new()),
        parent,
        is_synthesized: false,
        is_dataclass: false,
        is_named_tuple: false,
        is_typed_dict: false,
        fields: HashMap::new(),
        decorator_callees: HashMap::new(),
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

    let context = ModuleContext::create(test_module_handle, &transaction, &module_ids).unwrap();

    let expected_class_definitions = create_expected_class_definitions(&context);

    let reversed_override_graph = WholeProgramReversedOverrideGraph::new();
    let actual_class_definitions = export_all_classes(
        &collect_function_base_definitions(
            &handles,
            &transaction,
            &module_ids,
            &reversed_override_graph,
        ),
        &context,
    );

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
                "Bar".into(),
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
                    declaration_kind: Some(PysaClassFieldDeclaration::DefinedWithoutAssign),
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
            is_dataclass: false,
            is_named_tuple: true,
            is_typed_dict: false,
            fields: HashMap::from([
                (
                    "x".into(),
                    PysaClassField {
                        type_: PysaType::any_implicit(),
                        explicit_annotation: None,
                        location: Some(create_location(3, 30, 3, 33)),
                        declaration_kind: Some(
                            PysaClassFieldDeclaration::DeclaredWithoutAnnotation,
                        ),
                    },
                ),
                (
                    "y".into(),
                    PysaClassField {
                        type_: PysaType::any_implicit(),
                        explicit_annotation: None,
                        location: Some(create_location(3, 35, 3, 38)),
                        declaration_kind: Some(
                            PysaClassFieldDeclaration::DeclaredWithoutAnnotation,
                        ),
                    },
                ),
                (
                    "__match_args__".into(),
                    PysaClassField {
                        type_: PysaType::from_type(
                            &Type::concrete_tuple(vec![
                                context.stdlib.str().clone().to_type(),
                                context.stdlib.str().clone().to_type(),
                            ]),
                            context,
                        ),
                        explicit_annotation: None,
                        location: None,
                        declaration_kind: None,
                    },
                ),
            ]),
            decorator_callees: HashMap::new(),
        }
    },
);

exported_class_testcase!(
    test_export_typed_dict,
    r#"
from typing import TypedDict
class Point(TypedDict):
    x: int
    y: int
"#,
    &|context: &ModuleContext| {
        ClassDefinition {
            class_id: ClassId::from_int(0),
            name: "Point".to_owned(),
            bases: vec![get_class_ref(
                "_typeshed._type_checker_internals",
                "TypedDictFallback",
                context,
            )],
            mro: PysaClassMro::Resolved(vec![
                get_class_ref(
                    "_typeshed._type_checker_internals",
                    "TypedDictFallback",
                    context,
                ),
                get_class_ref("typing", "Mapping", context),
                get_class_ref("typing", "Collection", context),
                get_class_ref("typing", "Iterable", context),
                get_class_ref("typing", "Container", context),
            ]),
            parent: ScopeParent::TopLevel,
            is_synthesized: false,
            is_dataclass: false,
            is_named_tuple: false,
            is_typed_dict: true,
            fields: HashMap::from([
                (
                    "x".into(),
                    PysaClassField {
                        type_: PysaType::from_class_type(context.stdlib.int(), context),
                        explicit_annotation: Some("int".to_owned()),
                        location: Some(create_location(4, 5, 4, 6)),
                        declaration_kind: Some(PysaClassFieldDeclaration::DeclaredByAnnotation),
                    },
                ),
                (
                    "y".into(),
                    PysaClassField {
                        type_: PysaType::from_class_type(context.stdlib.int(), context),
                        explicit_annotation: Some("int".to_owned()),
                        location: Some(create_location(5, 5, 5, 6)),
                        declaration_kind: Some(PysaClassFieldDeclaration::DeclaredByAnnotation),
                    },
                ),
            ]),
            decorator_callees: HashMap::new(),
        }
    },
);

exported_class_testcase!(
    test_export_non_total_typed_dict,
    r#"
from typing import TypedDict
class Point(TypedDict, total=False):
    x: int
    y: int
"#,
    &|context: &ModuleContext| {
        ClassDefinition {
            class_id: ClassId::from_int(0),
            name: "Point".to_owned(),
            bases: vec![get_class_ref(
                "_typeshed._type_checker_internals",
                "TypedDictFallback",
                context,
            )],
            mro: PysaClassMro::Resolved(vec![
                get_class_ref(
                    "_typeshed._type_checker_internals",
                    "TypedDictFallback",
                    context,
                ),
                get_class_ref("typing", "Mapping", context),
                get_class_ref("typing", "Collection", context),
                get_class_ref("typing", "Iterable", context),
                get_class_ref("typing", "Container", context),
            ]),
            parent: ScopeParent::TopLevel,
            is_synthesized: false,
            is_dataclass: false,
            is_named_tuple: false,
            is_typed_dict: true,
            fields: HashMap::from([
                (
                    "x".into(),
                    PysaClassField {
                        type_: PysaType::from_class_type(context.stdlib.int(), context),
                        explicit_annotation: Some("int".to_owned()),
                        location: Some(create_location(4, 5, 4, 6)),
                        declaration_kind: Some(PysaClassFieldDeclaration::DeclaredByAnnotation),
                    },
                ),
                (
                    "y".into(),
                    PysaClassField {
                        type_: PysaType::from_class_type(context.stdlib.int(), context),
                        explicit_annotation: Some("int".to_owned()),
                        location: Some(create_location(5, 5, 5, 6)),
                        declaration_kind: Some(PysaClassFieldDeclaration::DeclaredByAnnotation),
                    },
                ),
            ]),
            decorator_callees: HashMap::new(),
        }
    },
);

exported_class_testcase!(
    test_export_typing_namedtuple_class,
    r#"
import typing
class Foo(typing.NamedTuple):
    x: int
    y: str
"#,
    &|context: &ModuleContext| {
        ClassDefinition {
            class_id: ClassId::from_int(0),
            name: "Foo".to_owned(),
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
            is_synthesized: false,
            is_dataclass: false,
            is_named_tuple: true,
            is_typed_dict: false,
            fields: HashMap::from([
                (
                    "x".into(),
                    PysaClassField {
                        type_: PysaType::from_class_type(context.stdlib.int(), context),
                        explicit_annotation: Some("int".into()),
                        location: Some(create_location(4, 5, 4, 6)),
                        declaration_kind: Some(PysaClassFieldDeclaration::DeclaredByAnnotation),
                    },
                ),
                (
                    "y".into(),
                    PysaClassField {
                        type_: PysaType::from_class_type(context.stdlib.str(), context),
                        explicit_annotation: Some("str".into()),
                        location: Some(create_location(5, 5, 5, 6)),
                        declaration_kind: Some(PysaClassFieldDeclaration::DeclaredByAnnotation),
                    },
                ),
                (
                    "__match_args__".into(),
                    PysaClassField {
                        type_: PysaType::from_type(
                            &Type::concrete_tuple(vec![
                                context.stdlib.str().clone().to_type(),
                                context.stdlib.str().clone().to_type(),
                            ]),
                            context,
                        ),
                        explicit_annotation: None,
                        location: None,
                        declaration_kind: None,
                    },
                ),
            ]),
            decorator_callees: HashMap::new(),
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
                "x".into(),
                PysaClassField {
                    type_: PysaType::from_class_type(context.stdlib.int(), context),
                    explicit_annotation: Some("int".to_owned()),
                    location: Some(create_location(4, 5, 4, 6)),
                    declaration_kind: Some(PysaClassFieldDeclaration::DeclaredByAnnotation),
                },
            ),
            (
                "y".into(),
                PysaClassField {
                    type_: PysaType::from_class_type(context.stdlib.str(), context),
                    explicit_annotation: Some("str".to_owned()),
                    location: Some(create_location(5, 5, 5, 6)),
                    declaration_kind: Some(PysaClassFieldDeclaration::DeclaredByAnnotation),
                },
            ),
            (
                "z".into(),
                PysaClassField {
                    type_: PysaType::from_class_type(context.stdlib.bool(), context),
                    explicit_annotation: Some(
                        "typing.Annotated[bool, \"annotation for z\"]".to_owned(),
                    ),
                    location: Some(create_location(6, 5, 6, 6)),
                    declaration_kind: Some(PysaClassFieldDeclaration::DeclaredByAnnotation),
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
                "x".into(),
                PysaClassField {
                    type_: PysaType::from_class_type(context.stdlib.int(), context),
                    explicit_annotation: Some("int".to_owned()),
                    location: Some(create_location(5, 14, 5, 15)),
                    declaration_kind: Some(PysaClassFieldDeclaration::DefinedInMethod),
                },
            ),
            (
                "y".into(),
                PysaClassField {
                    type_: PysaType::from_class_type(context.stdlib.str(), context),
                    explicit_annotation: Some("str".to_owned()),
                    location: Some(create_location(6, 14, 6, 15)),
                    declaration_kind: Some(PysaClassFieldDeclaration::DefinedInMethod),
                },
            ),
            (
                "z".into(),
                PysaClassField {
                    type_: PysaType::from_class_type(context.stdlib.bool(), context),
                    explicit_annotation: Some(
                        "typing.Annotated[bool, \"annotation for z\"]".to_owned(),
                    ),
                    location: Some(create_location(7, 14, 7, 15)),
                    declaration_kind: Some(PysaClassFieldDeclaration::DefinedInMethod),
                },
            ),
        ]))
    },
);

exported_class_testcase!(
    test_export_dataclass,
    r#"
from dataclasses import dataclass
@dataclass(frozen=True)
class Foo:
    x: int
    y: str

    def get(self) -> int:
        return self.x
"#,
    &|context: &ModuleContext| {
        create_simple_class("Foo", 0, ScopeParent::TopLevel)
            .with_is_dataclass(true)
            .with_fields(HashMap::from([
                (
                    "x".into(),
                    PysaClassField {
                        type_: PysaType::from_class_type(context.stdlib.int(), context),
                        explicit_annotation: Some("int".to_owned()),
                        location: Some(create_location(5, 5, 5, 6)),
                        declaration_kind: Some(PysaClassFieldDeclaration::DeclaredByAnnotation),
                    },
                ),
                (
                    "y".into(),
                    PysaClassField {
                        type_: PysaType::from_class_type(context.stdlib.str(), context),
                        explicit_annotation: Some("str".to_owned()),
                        location: Some(create_location(6, 5, 6, 6)),
                        declaration_kind: Some(PysaClassFieldDeclaration::DeclaredByAnnotation),
                    },
                ),
                (
                    "__dataclass_fields__".into(),
                    PysaClassField {
                        type_: PysaType::from_type(
                            &Type::ClassType(context.stdlib.dict(
                                context.stdlib.str().clone().to_type(),
                                Type::any_implicit(),
                            )),
                            context,
                        ),
                        explicit_annotation: None,
                        location: None,
                        declaration_kind: None,
                    },
                ),
                (
                    "__match_args__".into(),
                    PysaClassField {
                        type_: PysaType::from_type(
                            &Type::concrete_tuple(vec![
                                context.stdlib.str().clone().to_type(),
                                context.stdlib.str().clone().to_type(),
                            ]),
                            context,
                        ),
                        explicit_annotation: None,
                        location: None,
                        declaration_kind: None,
                    },
                ),
            ]))
            .with_decorator_callees(HashMap::from([(
                create_location(3, 2, 3, 11),
                vec![Target::Function(get_function_ref(
                    "dataclasses",
                    "dataclass",
                    context,
                ))],
            )]))
    },
);

exported_class_testcase!(
    test_export_class_decorator,
    r#"
def decorator(c):
    return c

@decorator
class Foo:
    pass
"#,
    &|context: &ModuleContext| {
        create_simple_class("Foo", 0, ScopeParent::TopLevel).with_decorator_callees(HashMap::from(
            [(
                create_location(5, 2, 5, 11),
                vec![Target::Function(get_function_ref(
                    "test",
                    "decorator",
                    context,
                ))],
            )],
        ))
    },
);

exported_class_testcase!(
    test_export_class_decorator_factory,
    r#"
def decorator(x):
    return lambda f: f

@decorator(1)
class Foo:
    pass
"#,
    &|context: &ModuleContext| {
        create_simple_class("Foo", 0, ScopeParent::TopLevel).with_decorator_callees(HashMap::from(
            [(
                create_location(5, 2, 5, 11),
                vec![Target::Function(get_function_ref(
                    "test",
                    "decorator",
                    context,
                ))],
            )],
        ))
    },
);

exported_class_testcase!(
    test_export_class_multiple_decorators,
    r#"
def d1(f):
    return f

def d2(f):
    return f

@d1
@d2
class Foo:
    pass
"#,
    &|context: &ModuleContext| {
        create_simple_class("Foo", 0, ScopeParent::TopLevel).with_decorator_callees(HashMap::from(
            [
                (
                    create_location(8, 2, 8, 4),
                    vec![Target::Function(get_function_ref("test", "d1", context))],
                ),
                (
                    create_location(9, 2, 9, 4),
                    vec![Target::Function(get_function_ref("test", "d2", context))],
                ),
            ],
        ))
    },
);
