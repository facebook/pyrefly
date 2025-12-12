/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use pretty_assertions::assert_eq;
use pyrefly_types::callable::Callable;
use pyrefly_types::callable::ParamList;
use pyrefly_types::class::ClassType;
use pyrefly_types::lit_int::LitInt;
use pyrefly_types::literal::Lit;
use pyrefly_types::simplify::unions;
use pyrefly_types::types::Type;

use crate::report::pysa::class::ClassRef;
use crate::report::pysa::context::ModuleContext;
use crate::report::pysa::module::ModuleIds;
use crate::report::pysa::types::ClassNamesFromType;
use crate::report::pysa::types::PysaType;
use crate::report::pysa::types::TypeModifier;
use crate::test::pysa::utils::create_state;
use crate::test::pysa::utils::get_class;
use crate::test::pysa::utils::get_class_ref;
use crate::test::pysa::utils::get_handle_for_module_name;

#[test]
fn test_pysa_type() {
    let state = create_state(
        "test",
        r#"
import enum

class MyEnum(enum.Enum):
    A = 1

class MyClass:
    pass

class A:
    pass
class B:
    pass
class C:
    pass
"#,
    );
    let transaction = state.transaction();
    let handles = transaction.handles();
    let module_ids = ModuleIds::new(&handles);

    let test_module_handle = get_handle_for_module_name("test", &transaction);
    let context = ModuleContext::create(test_module_handle, &transaction, &module_ids).unwrap();

    // Builtin types

    assert_eq!(
        PysaType::new(
            "int".to_owned(),
            ClassNamesFromType::from_class(context.stdlib.int().class_object(), &context),
        )
        .with_is_int(true),
        PysaType::from_type(&Type::ClassType(context.stdlib.int().clone()), &context),
    );

    assert_eq!(
        PysaType::new(
            "str".to_owned(),
            ClassNamesFromType::from_class(context.stdlib.str().class_object(), &context),
        ),
        PysaType::from_type(&Type::ClassType(context.stdlib.str().clone()), &context),
    );

    assert_eq!(
        PysaType::new(
            "bool".to_owned(),
            ClassNamesFromType::from_class(context.stdlib.bool().class_object(), &context),
        )
        .with_is_bool(true)
        .with_is_int(true),
        PysaType::from_type(&Type::ClassType(context.stdlib.bool().clone()), &context),
    );

    assert_eq!(
        PysaType::new(
            "float".to_owned(),
            ClassNamesFromType::from_class(context.stdlib.float().class_object(), &context),
        )
        .with_is_float(true),
        PysaType::from_type(&Type::ClassType(context.stdlib.float().clone()), &context),
    );

    assert_eq!(
        PysaType::new("None".to_owned(), ClassNamesFromType::not_a_class()),
        PysaType::from_type(&Type::None, &context),
    );

    assert_eq!(
        PysaType::new("Unknown".to_owned(), ClassNamesFromType::not_a_class()),
        PysaType::from_type(&Type::any_implicit(), &context),
    );

    assert_eq!(
        PysaType::new("typing.Any".to_owned(), ClassNamesFromType::not_a_class()),
        PysaType::from_type(&Type::any_explicit(), &context),
    );

    assert_eq!(
        PysaType::new(
            "test.MyEnum".to_owned(),
            ClassNamesFromType::from_class(&get_class("test", "MyEnum", &context), &context),
        )
        .with_is_enum(true),
        PysaType::from_type(
            &Type::ClassType(ClassType::new(
                get_class("test", "MyEnum", &context),
                Default::default()
            )),
            &context
        ),
    );

    assert_eq!(
        PysaType::new(
            "test.MyClass".to_owned(),
            ClassNamesFromType::from_class(&get_class("test", "MyClass", &context), &context),
        ),
        PysaType::from_type(
            &Type::ClassType(ClassType::new(
                get_class("test", "MyClass", &context),
                Default::default()
            )),
            &context
        ),
    );

    // Types wrapped into optionals

    assert_eq!(
        PysaType::new(
            "int | None".to_owned(),
            ClassNamesFromType::from_class(context.stdlib.int().class_object(), &context)
                .prepend_optional(),
        )
        .with_is_int(true),
        PysaType::from_type(
            &Type::optional(Type::ClassType(context.stdlib.int().clone())),
            &context
        ),
    );

    assert_eq!(
        PysaType::new(
            "str | None".to_owned(),
            ClassNamesFromType::from_class(context.stdlib.str().class_object(), &context)
                .prepend_optional(),
        ),
        PysaType::from_type(
            &Type::optional(Type::ClassType(context.stdlib.str().clone())),
            &context
        ),
    );

    assert_eq!(
        PysaType::new(
            "bool | None".to_owned(),
            ClassNamesFromType::from_class(context.stdlib.bool().class_object(), &context)
                .prepend_optional(),
        )
        .with_is_bool(true)
        .with_is_int(true),
        PysaType::from_type(
            &Type::optional(Type::ClassType(context.stdlib.bool().clone())),
            &context
        ),
    );

    assert_eq!(
        PysaType::new(
            "float | None".to_owned(),
            ClassNamesFromType::from_class(context.stdlib.float().class_object(), &context)
                .prepend_optional(),
        )
        .with_is_float(true),
        PysaType::from_type(
            &Type::optional(Type::ClassType(context.stdlib.float().clone())),
            &context
        ),
    );

    assert_eq!(
        PysaType::new(
            "test.MyEnum | None".to_owned(),
            ClassNamesFromType::from_class(&get_class("test", "MyEnum", &context), &context)
                .prepend_optional(),
        )
        .with_is_enum(true),
        PysaType::from_type(
            &Type::optional(Type::ClassType(ClassType::new(
                get_class("test", "MyEnum", &context),
                Default::default()
            ))),
            &context
        ),
    );

    assert_eq!(
        PysaType::new(
            "test.MyClass | None".to_owned(),
            ClassNamesFromType::from_class(&get_class("test", "MyClass", &context), &context)
                .prepend_optional(),
        ),
        PysaType::from_type(
            &Type::optional(Type::ClassType(ClassType::new(
                get_class("test", "MyClass", &context),
                Default::default()
            ))),
            &context
        ),
    );

    // Union of types

    assert_eq!(
        PysaType::new(
            "test.A | test.B".to_owned(),
            ClassNamesFromType::from_classes(
                vec![
                    get_class_ref("test", "A", &context),
                    get_class_ref("test", "B", &context),
                ],
                /* is_exhaustive */ true
            ),
        ),
        PysaType::from_type(
            &unions(vec![
                Type::ClassType(ClassType::new(
                    get_class("test", "A", &context),
                    Default::default()
                )),
                Type::ClassType(ClassType::new(
                    get_class("test", "B", &context),
                    Default::default()
                )),
            ]),
            &context
        ),
    );

    assert_eq!(
        PysaType::new(
            "(() -> None) | test.A".to_owned(),
            ClassNamesFromType::from_classes(
                vec![get_class_ref("test", "A", &context),],
                /* is_exhaustive */ false
            ),
        ),
        PysaType::from_type(
            &unions(vec![
                Type::ClassType(ClassType::new(
                    get_class("test", "A", &context),
                    Default::default()
                )),
                Type::Callable(Box::new(Callable::list(
                    ParamList::new(Vec::new()),
                    Type::None
                ))),
            ]),
            &context
        ),
    );

    assert_eq!(
        PysaType::new(
            "float | int".to_owned(),
            ClassNamesFromType::from_classes(
                vec![
                    ClassRef::from_class(context.stdlib.int().class_object(), context.module_ids),
                    ClassRef::from_class(context.stdlib.float().class_object(), context.module_ids),
                ],
                /* is_exhaustive */ true
            ),
        ),
        PysaType::from_type(
            &unions(vec![
                Type::ClassType(context.stdlib.float().clone()),
                Type::ClassType(context.stdlib.int().clone()),
            ]),
            &context
        ),
    );

    // Promote Literal types to their base type
    assert_eq!(
        PysaType::new(
            "int".to_owned(),
            ClassNamesFromType::from_class(context.stdlib.int().class_object(), &context),
        )
        .with_is_int(true),
        PysaType::from_type(&Type::Literal(Lit::Int(LitInt::new(0))), &context),
    );

    // Strip self type
    assert_eq!(
        PysaType::new(
            "test.MyClass".to_owned(),
            ClassNamesFromType::from_class(&get_class("test", "MyClass", &context), &context),
        ),
        PysaType::from_type(
            &Type::SelfType(ClassType::new(
                get_class("test", "MyClass", &context),
                Default::default()
            )),
            &context
        ),
    );

    // Strip awaitable
    assert_eq!(
        PysaType::new(
            "typing.Awaitable[int]".to_owned(),
            ClassNamesFromType::from_class(context.stdlib.int().class_object(), &context)
                .prepend_awaitable(),
        )
        .with_is_int(true),
        PysaType::from_type(
            &Type::ClassType(
                context
                    .stdlib
                    .awaitable(Type::ClassType(context.stdlib.int().clone()))
            ),
            &context
        ),
    );

    // Strip optional awaitable
    assert_eq!(
        PysaType::new(
            "typing.Awaitable[int] | None".to_owned(),
            ClassNamesFromType::from_class(context.stdlib.int().class_object(), &context)
                .prepend_awaitable()
                .prepend_optional(),
        )
        .with_is_int(true),
        PysaType::from_type(
            &Type::optional(Type::ClassType(
                context
                    .stdlib
                    .awaitable(Type::ClassType(context.stdlib.int().clone()))
            )),
            &context
        ),
    );

    assert_eq!(
        PysaType::new(
            "typing.Awaitable[test.A | test.B]".to_owned(),
            ClassNamesFromType::from_classes(
                vec![
                    get_class_ref("test", "A", &context),
                    get_class_ref("test", "B", &context),
                ],
                /* is_exhaustive */ true
            )
            .prepend_awaitable(),
        ),
        PysaType::from_type(
            &Type::ClassType(context.stdlib.awaitable(unions(vec![
                Type::ClassType(ClassType::new(
                    get_class("test", "A", &context),
                    Default::default()
                )),
                Type::ClassType(ClassType::new(
                    get_class("test", "B", &context),
                    Default::default()
                )),
            ]))),
            &context
        ),
    );

    // Handle type[A]
    assert_eq!(
        PysaType::new(
            "type[test.MyClass]".to_owned(),
            ClassNamesFromType::from_class(&get_class("test", "MyClass", &context), &context)
                .prepend_modifier(TypeModifier::Type),
        ),
        PysaType::from_type(
            &Type::ClassDef(get_class("test", "MyClass", &context),),
            &context
        ),
    );
    assert_eq!(
        PysaType::new(
            "type[test.MyClass]".to_owned(),
            ClassNamesFromType::from_class(&get_class("test", "MyClass", &context), &context)
                .prepend_modifier(TypeModifier::Type),
        ),
        PysaType::from_type(
            &Type::Type(Box::new(Type::ClassType(ClassType::new(
                get_class("test", "MyClass", &context),
                Default::default()
            )))),
            &context
        ),
    );

    assert_eq!(
        PysaType::new(
            "type[test.A] | type[test.B]".to_owned(),
            ClassNamesFromType::from_classes(
                vec![
                    get_class_ref("test", "A", &context),
                    get_class_ref("test", "B", &context),
                ],
                /* is_exhaustive */ true
            )
            .prepend_modifier(TypeModifier::Type),
        ),
        PysaType::from_type(
            &unions(vec![
                Type::Type(Box::new(Type::ClassType(ClassType::new(
                    get_class("test", "A", &context),
                    Default::default()
                )))),
                Type::Type(Box::new(Type::ClassType(ClassType::new(
                    get_class("test", "B", &context),
                    Default::default()
                )))),
            ]),
            &context
        ),
    );
}
