/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use pyrefly_derive::TypeEq;
use pyrefly_derive::Visit;
use pyrefly_derive::VisitMut;
use pyrefly_util::visit::Visit;
use pyrefly_util::visit::VisitMut;
use ruff_python_ast::name::Name;
use starlark_map::ordered_map::OrderedMap;

use crate::callable::FuncMetadata;
use crate::callable::FunctionKind;
use crate::literal::Lit;
use crate::tuple::Tuple;
use crate::types::CalleeKind;
use crate::types::Type;

#[derive(Debug, Clone, PartialEq, Eq, TypeEq, PartialOrd, Ord, Hash)]
pub struct TypeMap(pub OrderedMap<Name, Type>);

impl TypeMap {
    pub fn new() -> Self {
        Self(OrderedMap::new())
    }

    pub fn get_bool(&self, name: &Name) -> Option<bool> {
        self.0.get(name).and_then(|t| t.as_bool())
    }

    pub fn get_string(&self, name: &Name) -> Option<&str> {
        self.0.get(name).and_then(|t| match t {
            Type::Literal(Lit::Str(s)) => Some(&**s),
            _ => None,
        })
    }
}

impl Visit<Type> for TypeMap {
    fn recurse<'a>(&'a self, f: &mut dyn FnMut(&'a Type)) {
        for (_, ty) in self.0.iter() {
            ty.visit(f);
        }
    }
}

impl VisitMut<Type> for TypeMap {
    fn recurse_mut(&mut self, f: &mut dyn FnMut(&mut Type)) {
        for (_, ty) in self.0.iter_mut() {
            ty.visit_mut(f);
        }
    }
}

/// Wraps the result of a function call whose keyword arguments have typing effects.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(Visit, VisitMut, TypeEq)]
pub struct KwCall {
    /// The metadata of the called function.
    pub func_metadata: FuncMetadata,
    /// The keyword arguments that the function was called with.
    pub keywords: TypeMap,
    /// The return type of the call.
    pub return_ty: Type,
}

impl KwCall {
    pub fn has_function_kind(&self, kind: FunctionKind) -> bool {
        self.func_metadata.kind == kind
    }
}

/// Parameters to `typing.dataclass_transform`.
/// See https://typing.python.org/en/latest/spec/dataclasses.html#dataclass-transform-parameters.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(Visit, VisitMut, TypeEq)]
pub struct DataclassTransformKeywords {
    pub eq_default: bool,
    pub order_default: bool,
    pub kw_only_default: bool,
    pub frozen_default: bool,
    pub field_specifiers: Vec<CalleeKind>,
}

impl DataclassTransformKeywords {
    const EQ_DEFAULT: Name = Name::new_static("eq_default");
    const ORDER_DEFAULT: Name = Name::new_static("order_default");
    const KW_ONLY_DEFAULT: Name = Name::new_static("kw_only_default");
    const FROZEN_DEFAULT: Name = Name::new_static("frozen_default");
    const FIELD_SPECIFIERS: Name = Name::new_static("field_specifiers");

    pub fn from_type_map(map: &TypeMap) -> Self {
        Self {
            eq_default: map.get_bool(&Self::EQ_DEFAULT).unwrap_or(true),
            order_default: map.get_bool(&Self::ORDER_DEFAULT).unwrap_or(false),
            kw_only_default: map.get_bool(&Self::KW_ONLY_DEFAULT).unwrap_or(false),
            frozen_default: map.get_bool(&Self::FROZEN_DEFAULT).unwrap_or(false),
            field_specifiers: match map.0.get(&Self::FIELD_SPECIFIERS) {
                Some(Type::Tuple(Tuple::Concrete(elts))) => {
                    elts.iter().filter_map(|e| e.callee_kind()).collect()
                }
                _ => Vec::new(),
            },
        }
    }

    pub fn new() -> Self {
        Self::from_type_map(&TypeMap::new())
    }
}

/// Parameters to dataclass field specifiers.
/// See https://typing.python.org/en/latest/spec/dataclasses.html#field-specifier-parameters.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(Visit, VisitMut, TypeEq)]
pub struct DataclassFieldKeywords {
    pub init: bool,
    /// Whether this field has a default. Note that this is derived from the various
    /// default-related parameters but does not correspond directly to any of them
    pub default: bool,
    /// None means that kw_only was not explicitly set
    pub kw_only: Option<bool>,
    /// Alias that is used in `__init__` rather than the field name. None means no alias
    pub alias: Option<Name>,
    /// If a converter callable is passed in, its first positional parameter
    pub converter_param: Option<Type>,
}

impl DataclassFieldKeywords {
    pub const INIT: Name = Name::new_static("init");
    /// We combine default, default_factory, and factory into a single "default" keyword indicating
    /// whether the field has a default. The default value isn't stored.
    pub const DEFAULT: Name = Name::new_static("default");
    pub const DEFAULT_FACTORY: Name = Name::new_static("default_factory");
    pub const FACTORY: Name = Name::new_static("factory");
    pub const KW_ONLY: Name = Name::new_static("kw_only");
    pub const ALIAS: Name = Name::new_static("alias");
    /// We extract and store only the first positional parameter to the converter callable.
    pub const CONVERTER: Name = Name::new_static("converter");

    pub fn new() -> Self {
        Self {
            init: true,
            default: false,
            kw_only: None,
            alias: None,
            converter_param: None,
        }
    }

    pub fn is_kw_only(&self) -> bool {
        self.kw_only == Some(true)
    }
}

/// Dataclass parameters.
/// See https://typing.python.org/en/latest/spec/dataclasses.html#decorator-function-and-class-metaclass-parameters.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(Visit, VisitMut, TypeEq)]
pub struct DataclassKeywords {
    pub init: bool,
    pub order: bool,
    pub frozen: bool,
    pub match_args: bool,
    pub kw_only: bool,
    pub eq: bool,
    pub unsafe_hash: bool,
    pub slots: bool,
}

impl DataclassKeywords {
    const INIT: Name = Name::new_static("init");
    const ORDER: Name = Name::new_static("order");
    const FROZEN: Name = Name::new_static("frozen");
    const MATCH_ARGS: Name = Name::new_static("match_args");
    const KW_ONLY: Name = Name::new_static("kw_only");
    const EQ: Name = Name::new_static("eq");
    const UNSAFE_HASH: Name = Name::new_static("unsafe_hash");
    const SLOTS: Name = Name::new_static("slots");

    pub fn from_type_map(map: &TypeMap, defaults: &DataclassTransformKeywords) -> Self {
        Self {
            init: map.get_bool(&Self::INIT).unwrap_or(true),
            order: map.get_bool(&Self::ORDER).unwrap_or(defaults.order_default),
            frozen: map
                .get_bool(&Self::FROZEN)
                .unwrap_or(defaults.frozen_default),
            match_args: map.get_bool(&Self::MATCH_ARGS).unwrap_or(true),
            kw_only: map
                .get_bool(&Self::KW_ONLY)
                .unwrap_or(defaults.kw_only_default),
            eq: map.get_bool(&Self::EQ).unwrap_or(defaults.eq_default),
            unsafe_hash: map.get_bool(&Self::UNSAFE_HASH).unwrap_or(false),
            slots: map.get_bool(&Self::SLOTS).unwrap_or(false),
        }
    }

    pub fn new() -> Self {
        Self::from_type_map(&TypeMap::new(), &DataclassTransformKeywords::new())
    }
}
