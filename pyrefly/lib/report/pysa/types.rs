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
use serde::Serialize;

use crate::report::pysa::ModuleContext;
use crate::report::pysa::class::ClassRef;
use crate::types::display::TypeDisplayContext;

// List of class names that a type refers to, after stripping Optional and Awaitable.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ClassNamesFromType {
    class_names: Vec<ClassRef>,
    #[serde(skip_serializing_if = "<&bool>::not")]
    stripped_coroutine: bool,
    #[serde(skip_serializing_if = "<&bool>::not")]
    stripped_optional: bool,
    #[serde(skip_serializing_if = "<&bool>::not")]
    stripped_readonly: bool,
    #[serde(skip_serializing_if = "<&bool>::not")]
    unbound_type_variable: bool,
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
    #[serde(skip_serializing_if = "<&bool>::not")]
    is_bool: bool,
    #[serde(skip_serializing_if = "<&bool>::not")]
    is_int: bool,
    #[serde(skip_serializing_if = "<&bool>::not")]
    is_float: bool,
    #[serde(skip_serializing_if = "<&bool>::not")]
    is_enum: bool,

    #[serde(skip_serializing_if = "ClassNamesFromType::skip_serializing")]
    class_names: ClassNamesFromType,
}

impl ClassNamesFromType {
    pub fn from_class(class: &Class, context: &ModuleContext) -> ClassNamesFromType {
        ClassNamesFromType {
            class_names: vec![ClassRef::from_class(class, context.module_ids)],
            stripped_coroutine: false,
            stripped_optional: false,
            stripped_readonly: false,
            unbound_type_variable: false,
            is_exhaustive: true,
        }
    }

    #[cfg(test)]
    pub fn from_classes(class_names: Vec<ClassRef>, is_exhaustive: bool) -> ClassNamesFromType {
        ClassNamesFromType {
            class_names,
            stripped_coroutine: false,
            stripped_optional: false,
            stripped_readonly: false,
            unbound_type_variable: false,
            is_exhaustive,
        }
    }

    pub fn not_a_class() -> ClassNamesFromType {
        ClassNamesFromType {
            class_names: vec![],
            stripped_coroutine: false,
            stripped_optional: false,
            stripped_readonly: false,
            unbound_type_variable: false,
            is_exhaustive: false,
        }
    }

    fn skip_serializing(&self) -> bool {
        self.class_names.is_empty()
    }

    pub fn with_strip_optional(mut self, stripped_optional: bool) -> ClassNamesFromType {
        self.stripped_optional = stripped_optional;
        self
    }

    pub fn with_strip_coroutine(mut self, stripped_coroutine: bool) -> ClassNamesFromType {
        self.stripped_coroutine = stripped_coroutine;
        self
    }

    fn join_with(mut self, other: ClassNamesFromType) -> ClassNamesFromType {
        self.class_names.extend(other.class_names);
        self.stripped_coroutine |= other.stripped_coroutine;
        self.stripped_optional |= other.stripped_optional;
        self.stripped_readonly |= other.stripped_readonly;
        self.unbound_type_variable |= other.unbound_type_variable;
        self.is_exhaustive &= other.is_exhaustive;
        self
    }

    fn sort_and_dedup(mut self) -> ClassNamesFromType {
        self.class_names.sort();
        self.class_names.dedup();
        self
    }
}

fn string_for_type(type_: &Type) -> String {
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
        Type::Union(elements) if elements.len() == 2 && elements[0] == Type::None => {
            Some(&elements[1])
        }
        Type::Union(elements) if elements.len() == 2 && elements[1] == Type::None => {
            Some(&elements[0])
        }
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

fn has_superclass(class: &Class, want: &Class, context: &ModuleContext) -> bool {
    context
        .transaction
        .ad_hoc_solve(context.handle, |solver| {
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
        return get_classes_of_type(inner, context).with_strip_optional(true);
    }
    if let Some(inner) = strip_awaitable(type_, context) {
        return get_classes_of_type(inner, context).with_strip_coroutine(true);
    }
    if let Some(inner) = strip_coroutine(type_, context) {
        return get_classes_of_type(inner, context).with_strip_coroutine(true);
    }
    // No need to strip ReadOnly[], it is already stripped by pyrefly.
    match type_ {
        Type::ClassType(class_type) => {
            ClassNamesFromType::from_class(class_type.class_object(), context)
        }
        Type::Tuple(_) => ClassNamesFromType::from_class(context.stdlib.tuple_object(), context),
        Type::Union(elements) if !elements.is_empty() => elements
            .iter()
            .map(|inner| get_classes_of_type(inner, context))
            .reduce(|acc, next| acc.join_with(next))
            .unwrap()
            .sort_and_dedup(),
        Type::TypeAlias(alias) => get_classes_of_type(&alias.as_type(), context),
        _ => ClassNamesFromType::not_a_class(),
    }
}

impl PysaType {
    #[cfg(test)]
    pub fn new(string: String, class_names: ClassNamesFromType) -> PysaType {
        PysaType {
            string,
            is_bool: false,
            is_int: false,
            is_float: false,
            is_enum: false,
            class_names,
        }
    }

    #[cfg(test)]
    pub fn with_is_bool(mut self, is_bool: bool) -> PysaType {
        self.is_bool = is_bool;
        self
    }

    #[cfg(test)]
    pub fn with_is_int(mut self, is_int: bool) -> PysaType {
        self.is_int = is_int;
        self
    }

    #[cfg(test)]
    pub fn with_is_float(mut self, is_float: bool) -> PysaType {
        self.is_float = is_float;
        self
    }

    #[cfg(test)]
    pub fn with_is_enum(mut self, is_enum: bool) -> PysaType {
        self.is_enum = is_enum;
        self
    }

    pub fn from_type(type_: &Type, context: &ModuleContext) -> PysaType {
        // Promote `Literal[..]` into `str` or `int`.
        let type_ = type_.clone().promote_literals(&context.stdlib);
        let type_ = strip_self_type(type_);

        let string = string_for_type(&type_);

        PysaType {
            string,
            is_bool: is_scalar_type(&type_, context.stdlib.bool().class_object(), context),
            is_int: is_scalar_type(&type_, context.stdlib.int().class_object(), context),
            is_float: is_scalar_type(&type_, context.stdlib.float().class_object(), context),
            is_enum: is_scalar_type(&type_, context.stdlib.enum_class().class_object(), context),
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
            is_bool: false,
            is_int: false,
            is_float: false,
            is_enum: false,
            class_names: ClassNamesFromType::not_a_class(),
        }
    }

    #[cfg(test)]
    pub fn none() -> PysaType {
        PysaType {
            string: "None".to_owned(),
            is_bool: false,
            is_int: false,
            is_float: false,
            is_enum: false,
            class_names: ClassNamesFromType::not_a_class(),
        }
    }
}
