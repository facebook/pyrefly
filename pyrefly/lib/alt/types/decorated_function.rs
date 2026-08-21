/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::fmt;
use std::fmt::Debug;
use std::fmt::Display;
use std::path::PathBuf;
use std::sync::Arc;

use pyrefly_derive::TypeEq;
use pyrefly_derive::Visit;
use pyrefly_derive::VisitMut;
use pyrefly_python::module::Module;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePath;
use pyrefly_python::short_identifier::ShortIdentifier;
use pyrefly_types::callable::Param;
use pyrefly_types::class::Class;
use pyrefly_types::function::Deprecation;
use pyrefly_types::function::FuncDefIndex;
use pyrefly_types::keywords::TypeMap;
use pyrefly_types::quantified::Quantified;
use pyrefly_types::type_level_dsl::ParsedTypeShapeDslFunction;
use pyrefly_types::types::TParams;
use ruff_python_ast::Identifier;
use ruff_python_ast::name::Name;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use starlark_map::small_map::SmallMap;

use crate::types::function::FuncMetadata;
use crate::types::types::Type;

/// Information about the function def before decorators are applied. The metadata stored here
/// includes information from decorators, like @classmethod.
#[derive(Clone, Debug, Visit, VisitMut, TypeEq, PartialEq, Eq)]
pub struct UndecoratedFunction {
    pub def_index: FuncDefIndex,
    pub identifier: ShortIdentifier,
    pub metadata: FuncMetadata,
    pub decorators: Box<[(Type, TextRange)]>,
    pub tparams: Arc<TParams>,
    pub params: Vec<Param>,
    /// Alias-preserving parameter types used to construct the hover signature.
    pub display_param_types: SmallMap<Name, Type>,
    pub paramspec: Option<Quantified>,
    pub defining_cls: Option<Class>,
    pub type_shape_dsl_def: Option<Arc<ParsedTypeShapeDslFunction>>,
    /// Maps parameter names to their resolved types - used to connect
    /// FunctionParameter and KeyUndecoratedFunction.
    pub resolved_param_types: SmallMap<Name, Type>,
}

/// Answer for BindingDecorator
#[derive(Clone, Debug, Visit, VisitMut, TypeEq, PartialEq, Eq)]
pub struct Decorator {
    pub ty: Type,
    pub deprecation: Option<Deprecation>,
}

impl Display for Decorator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Decorator[{}]", self.ty)
    }
}

/// Decorators that need special handling
pub enum SpecialDecorator<'a> {
    Overload,
    StaticMethod(Name),
    ClassMethod(Name),
    Property(Name),
    CachedProperty(Name),
    EnumMember,
    Override,
    Final,
    Deprecated(&'a Deprecation),
    PropertySetter(&'a Type),
    PropertyDeleter(&'a Type),
    DataclassTransformCall(&'a TypeMap),
    EnumNonmember,
    AbstractMethod,
    NoTypeCheck,
    UsesShapeDsl,
    DefinesAssertShape,
    DisjointBase,
}

impl UndecoratedFunction {
    pub fn recursive() -> Self {
        UndecoratedFunction {
            def_index: FuncDefIndex(u32::MAX),
            identifier: ShortIdentifier::new(&Identifier::new(
                Name::default(),
                TextRange::default(),
            )),
            metadata: FuncMetadata::synthesized(
                &Module::new(
                    ModuleName::from_str("__undecorated_function_recursive__"),
                    ModulePath::filesystem(PathBuf::default()),
                    Arc::new("".to_owned()),
                ),
                None,
                Name::default(),
            ),
            decorators: Box::from([]),
            tparams: Arc::new(TParams::default()),
            params: Vec::new(),
            display_param_types: SmallMap::new(),
            paramspec: None,
            defining_cls: None,
            type_shape_dsl_def: None,
            resolved_param_types: SmallMap::new(),
        }
    }

    pub fn id_range(&self) -> TextRange {
        self.identifier.range()
    }
}

impl Display for UndecoratedFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "def {}: ...", self.metadata.kind.function_name())
    }
}
