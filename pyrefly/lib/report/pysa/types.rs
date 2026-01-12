/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::ops::Not;

#[cfg(test)]
use dupe::Dupe;
use pyrefly_types::class::Class;
#[cfg(test)]
use pyrefly_types::class::ClassType;
use pyrefly_types::types::Type;
use pyrefly_types::types::Union;
use serde::Serialize;

use crate::report::pysa::ModuleContext;
use crate::report::pysa::class::ClassRef;
use crate::types::display::TypeDisplayContext;

/// Modifier that was stripped from a type to extract the underlying class.
#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[allow(dead_code)]
pub enum TypeModifier {
    Optional,          // Optional[T]
    Coroutine,         // Coroutine[<...>]
    Awaitable,         // Awaitable[T]
    TypeVariableBound, // TypeVar(.., bound=T)
    Type,              // type[T]
}

/// A class reference along with the modifiers that were stripped to extract it.
#[derive(Debug, Clone, Serialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct ClassWithModifiers {
    pub class: ClassRef,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub modifiers: Vec<TypeModifier>,
}

impl ClassWithModifiers {
    pub fn new(class: ClassRef) -> ClassWithModifiers {
        ClassWithModifiers {
            class,
            modifiers: Vec::new(),
        }
    }
}

// List of class names that a type refers to, after stripping Optional and Awaitable.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ClassNamesFromType {
    classes: Vec<ClassWithModifiers>,
    // Is there an element (after stripping) that isn't a class name?
    #[serde(skip_serializing_if = "<&bool>::not")]
    is_exhaustive: bool,
}

/// Information needed from Pysa about a type.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct PysaType {
    // Pretty string representation of the type. Usually meant for the user.
    string: String,

    // Whether the type is a bool/int/float/enum, after stripping Optional and Awaitable.
    #[serde(flatten)]
    scalar_type_properties: ScalarTypeProperties,

    #[serde(skip_serializing_if = "ClassNamesFromType::skip_serializing")]
    class_names: ClassNamesFromType,
}

impl ClassNamesFromType {
    pub fn from_class(class: &Class, context: &ModuleContext) -> ClassNamesFromType {
        ClassNamesFromType {
            classes: vec![ClassWithModifiers::new(ClassRef::from_class(
                class,
                context.module_ids,
            ))],
            is_exhaustive: true,
        }
    }

    #[cfg(test)]
    pub fn from_classes(class_names: Vec<ClassRef>, is_exhaustive: bool) -> ClassNamesFromType {
        ClassNamesFromType {
            classes: class_names
                .into_iter()
                .map(ClassWithModifiers::new)
                .collect(),
            is_exhaustive,
        }
    }

    pub fn not_a_class() -> ClassNamesFromType {
        ClassNamesFromType {
            classes: vec![],
            is_exhaustive: false,
        }
    }

    fn skip_serializing(&self) -> bool {
        self.classes.is_empty()
    }

    pub fn prepend_optional(self) -> ClassNamesFromType {
        self.prepend_modifier(TypeModifier::Optional)
    }

    pub fn prepend_awaitable(self) -> ClassNamesFromType {
        self.prepend_modifier(TypeModifier::Awaitable)
    }

    pub fn prepend_coroutine(self) -> ClassNamesFromType {
        self.prepend_modifier(TypeModifier::Coroutine)
    }

    pub fn prepend_modifier(mut self, modifier: TypeModifier) -> ClassNamesFromType {
        for class in &mut self.classes {
            class.modifiers.insert(0, modifier);
        }
        self
    }

    fn join_with(mut self, other: ClassNamesFromType) -> ClassNamesFromType {
        self.classes.extend(other.classes);
        self.is_exhaustive &= other.is_exhaustive;
        self
    }

    fn sort_and_dedup(mut self) -> ClassNamesFromType {
        self.classes.sort();
        self.classes.dedup();
        self
    }
}

pub fn string_for_type(type_: &Type) -> String {
    let mut ctx = TypeDisplayContext::new(&[type_]);
    ctx.always_display_module_name_except_builtins();
    ctx.display(type_).to_string()
}

fn strip_self_type(mut ty: Type) -> Type {
    ty.transform_mut(&mut |t| {
        if let Type::SelfType(cls) = t {
            *t = Type::ClassType(cls.clone());
        }
    });
    ty
}

fn strip_optional(type_: &Type) -> Option<&Type> {
    match type_ {
        Type::Union(box Union {
            members: elements, ..
        }) if elements.len() == 2 && elements[0] == Type::None => Some(&elements[1]),
        Type::Union(box Union {
            members: elements, ..
        }) if elements.len() == 2 && elements[1] == Type::None => Some(&elements[0]),
        _ => None,
    }
}

fn strip_awaitable<'a>(type_: &'a Type, context: &ModuleContext) -> Option<&'a Type> {
    match type_ {
        Type::ClassType(class_type)
            if class_type.class_object() == context.stdlib.awaitable_object()
                && class_type.targs().as_slice().len() == 1 =>
        {
            Some(&class_type.targs().as_slice()[0])
        }
        _ => None,
    }
}

fn strip_coroutine<'a>(type_: &'a Type, context: &ModuleContext) -> Option<&'a Type> {
    match type_ {
        Type::ClassType(class_type)
            if class_type.class_object() == context.stdlib.coroutine_object()
                && class_type.targs().as_slice().len() >= 3 =>
        {
            Some(&class_type.targs().as_slice()[2])
        }
        _ => None,
    }
}

pub fn has_superclass(class: &Class, want: &Class, context: &ModuleContext) -> bool {
    context
        .transaction
        .ad_hoc_solve(&context.handle, |solver| {
            solver.type_order().has_superclass(class, want)
        })
        .unwrap()
}

fn is_scalar_type(get: &Type, want: &Class, context: &ModuleContext) -> bool {
    if let Some(inner) = strip_optional(get) {
        return is_scalar_type(inner, want, context);
    }
    if let Some(inner) = strip_awaitable(get, context) {
        return is_scalar_type(inner, want, context);
    }
    if let Some(inner) = strip_coroutine(get, context) {
        return is_scalar_type(inner, want, context);
    }
    match get {
        Type::ClassType(class_type) => has_superclass(class_type.class_object(), want, context),
        Type::TypeAlias(alias) => is_scalar_type(&alias.as_type(), want, context),
        _ => false,
    }
}

fn get_classes_of_type(type_: &Type, context: &ModuleContext) -> ClassNamesFromType {
    if let Some(inner) = strip_optional(type_) {
        return get_classes_of_type(inner, context).prepend_optional();
    }
    if let Some(inner) = strip_awaitable(type_, context) {
        return get_classes_of_type(inner, context).prepend_awaitable();
    }
    if let Some(inner) = strip_coroutine(type_, context) {
        return get_classes_of_type(inner, context).prepend_coroutine();
    }
    // No need to strip ReadOnly[], it is already stripped by pyrefly.
    match type_ {
        Type::ClassType(class_type) => {
            ClassNamesFromType::from_class(class_type.class_object(), context)
        }
        Type::ClassDef(class) => {
            ClassNamesFromType::from_class(class, context).prepend_modifier(TypeModifier::Type)
        }
        Type::Type(box Type::ClassType(class_type)) => {
            ClassNamesFromType::from_class(class_type.class_object(), context)
                .prepend_modifier(TypeModifier::Type)
        }
        Type::Tuple(_) => ClassNamesFromType::from_class(context.stdlib.tuple_object(), context),
        Type::Union(box Union {
            members: elements, ..
        }) if !elements.is_empty() => elements
            .iter()
            .map(|inner| get_classes_of_type(inner, context))
            .reduce(|acc, next| acc.join_with(next))
            .unwrap()
            .sort_and_dedup(),
        Type::TypeAlias(alias) => get_classes_of_type(&alias.as_type(), context),
        _ => ClassNamesFromType::not_a_class(),
    }
}

/// Apply normalization to a type before exporting it to Pysa.
pub fn preprocess_type(type_: &Type, context: &ModuleContext) -> Type {
    // Promote `Literal[..]` into `str` or `int`.
    let type_ = type_.clone().promote_literals(&context.stdlib);
    strip_self_type(type_)
}

impl PysaType {
    #[cfg(test)]
    pub fn new(string: String, class_names: ClassNamesFromType) -> PysaType {
        PysaType {
            string,
            scalar_type_properties: ScalarTypeProperties {
                is_bool: false,
                is_int: false,
                is_float: false,
                is_enum: false,
            },
            class_names,
        }
    }

    #[cfg(test)]
    pub fn with_is_bool(mut self, is_bool: bool) -> PysaType {
        self.scalar_type_properties.is_bool = is_bool;
        self
    }

    #[cfg(test)]
    pub fn with_is_int(mut self, is_int: bool) -> PysaType {
        self.scalar_type_properties.is_int = is_int;
        self
    }

    #[cfg(test)]
    pub fn with_is_float(mut self, is_float: bool) -> PysaType {
        self.scalar_type_properties.is_float = is_float;
        self
    }

    #[cfg(test)]
    pub fn with_is_enum(mut self, is_enum: bool) -> PysaType {
        self.scalar_type_properties.is_enum = is_enum;
        self
    }

    pub fn from_type(type_: &Type, context: &ModuleContext) -> PysaType {
        let type_ = preprocess_type(type_, context);
        let string = string_for_type(&type_);

        PysaType {
            string,
            scalar_type_properties: ScalarTypeProperties::from_preprocessed_type(&type_, context),
            class_names: get_classes_of_type(&type_, context),
        }
    }

    #[cfg(test)]
    pub fn from_class_type(class_type: &ClassType, context: &ModuleContext) -> PysaType {
        PysaType::from_type(&Type::ClassType(class_type.clone()), context)
    }

    #[cfg(test)]
    pub fn from_class(class: &Class, context: &ModuleContext) -> PysaType {
        PysaType::from_type(
            &Type::ClassType(ClassType::new(class.dupe(), Default::default())),
            context,
        )
    }

    #[cfg(test)]
    pub fn any_implicit() -> PysaType {
        PysaType {
            string: "Unknown".to_owned(),
            scalar_type_properties: ScalarTypeProperties::none(),
            class_names: ClassNamesFromType::not_a_class(),
        }
    }

    #[cfg(test)]
    pub fn none() -> PysaType {
        PysaType {
            string: "None".to_owned(),
            scalar_type_properties: ScalarTypeProperties::none(),
            class_names: ClassNamesFromType::not_a_class(),
        }
    }
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq, Copy, Hash, PartialOrd, Ord)]
pub struct ScalarTypeProperties {
    // Whether the type is a bool/int/float/enum, after stripping Optional and Awaitable.
    #[serde(skip_serializing_if = "<&bool>::not")]
    is_bool: bool,
    #[serde(skip_serializing_if = "<&bool>::not")]
    is_int: bool,
    #[serde(skip_serializing_if = "<&bool>::not")]
    is_float: bool,
    #[serde(skip_serializing_if = "<&bool>::not")]
    is_enum: bool,
}

impl ScalarTypeProperties {
    pub fn from_type(type_: &Type, context: &ModuleContext) -> ScalarTypeProperties {
        let type_ = preprocess_type(type_, context);
        Self::from_preprocessed_type(&type_, context)
    }

    fn from_preprocessed_type(type_: &Type, context: &ModuleContext) -> ScalarTypeProperties {
        ScalarTypeProperties {
            is_bool: is_scalar_type(type_, context.stdlib.bool().class_object(), context),
            is_int: is_scalar_type(type_, context.stdlib.int().class_object(), context),
            is_float: is_scalar_type(type_, context.stdlib.float().class_object(), context),
            is_enum: is_scalar_type(type_, context.stdlib.enum_class().class_object(), context),
        }
    }

    pub fn none() -> ScalarTypeProperties {
        ScalarTypeProperties {
            is_bool: false,
            is_int: false,
            is_float: false,
            is_enum: false,
        }
    }

    #[allow(clippy::trivially_copy_pass_by_ref)]
    pub fn is_none(&self) -> bool {
        *self == Self::none()
    }

    #[cfg(test)]
    pub fn int() -> ScalarTypeProperties {
        ScalarTypeProperties {
            is_bool: false,
            is_int: true,
            is_float: false,
            is_enum: false,
        }
    }

    #[cfg(test)]
    pub fn bool() -> ScalarTypeProperties {
        ScalarTypeProperties {
            is_bool: true,
            is_int: true,
            is_float: false,
            is_enum: false,
        }
    }

    #[cfg(test)]
    pub fn float() -> ScalarTypeProperties {
        ScalarTypeProperties {
            is_bool: false,
            is_int: false,
            is_float: true,
            is_enum: false,
        }
    }
}

pub fn is_callable_like(ty: &Type) -> bool {
    match ty {
        Type::Function(_) => true,
        Type::Callable(_) => true,
        Type::BoundMethod(_) => true,
        Type::Overload(_) => true,
        Type::Union(box Union {
            members: elements, ..
        }) => {
            elements.iter().any(is_callable_like)
                && elements
                    .iter()
                    .all(|ty| ty.is_none() || ty.is_any() || is_callable_like(ty))
        }
        _ => false,
    }
}

pub fn is_bound_method_like(ty: &Type) -> bool {
    match ty {
        Type::BoundMethod(_) => true,
        Type::Overload(_) => true,
        Type::Union(box Union {
            members: elements, ..
        }) => {
            elements.iter().any(is_bound_method_like)
                && elements
                    .iter()
                    .all(|ty| ty.is_none() || ty.is_any() || is_bound_method_like(ty))
        }
        _ => false,
    }
}
