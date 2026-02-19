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
use pyrefly_graph::index::Idx;
use pyrefly_python::module::Module;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePath;
use pyrefly_python::short_identifier::ShortIdentifier;
use pyrefly_types::callable::Deprecation;
use pyrefly_types::callable::FuncFlags;
use pyrefly_types::callable::FuncId;
use pyrefly_types::callable::FunctionKind;
use pyrefly_types::callable::Param;
use pyrefly_types::class::Class;
use pyrefly_types::keywords::TypeMap;
use pyrefly_types::quantified::Quantified;
use pyrefly_types::types::TParams;
use ruff_python_ast::Identifier;
use ruff_python_ast::name::Name;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;

use crate::alt::answers::Answers;
use crate::alt::answers::LookupAnswer;
use crate::alt::answers_solver::AnswersSolver;
use crate::binding::binding::FunctionStubOrImpl;
use crate::binding::binding::KeyDecoratedFunction;
use crate::binding::bindings::Bindings;
use crate::types::callable::FuncMetadata;
use crate::types::types::Type;

/// Information about the function def before decorators are applied. The metadata stored here
/// includes information from decorators, like @classmethod.
#[derive(Clone, Debug, Visit, VisitMut, TypeEq, PartialEq, Eq)]
pub struct UndecoratedFunction {
    pub identifier: ShortIdentifier,
    pub metadata: FuncMetadata,
    pub decorators: Box<[(Type, TextRange)]>,
    pub tparams: Arc<TParams>,
    pub params: Vec<Param>,
    pub paramspec: Option<Quantified>,
    pub stub_or_impl: FunctionStubOrImpl,
    pub defining_cls: Option<Class>,
}

/// A value that combines the metadata of a function def and also provides the type of the function
/// after decorators are applied. Note that the type might not be a function at all, since
/// decorators can produce any type.
#[derive(Clone, Debug)]
pub struct DecoratedFunction {
    pub idx: Idx<KeyDecoratedFunction>,
    pub ty: Arc<Type>,
    pub undecorated: Arc<UndecoratedFunction>,
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
}

impl UndecoratedFunction {
    pub fn recursive() -> Self {
        UndecoratedFunction {
            identifier: ShortIdentifier::new(&Identifier::new(
                Name::default(),
                TextRange::default(),
            )),
            metadata: FuncMetadata {
                kind: FunctionKind::Def(Box::new(FuncId {
                    module: Module::new(
                        ModuleName::from_str("__undecorated_function_recursive__"),
                        ModulePath::filesystem(PathBuf::default()),
                        Arc::new("".to_owned()),
                    ),
                    cls: None,
                    name: Name::default(),
                })),
                flags: FuncFlags::default(),
            },
            decorators: Box::from([]),
            tparams: Arc::new(TParams::default()),
            params: Vec::new(),
            paramspec: None,
            stub_or_impl: FunctionStubOrImpl::Stub,
            defining_cls: None,
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

impl DecoratedFunction {
    pub fn from_bindings_answers(
        idx: Idx<KeyDecoratedFunction>,
        bindings: &Bindings,
        answers: &Answers,
    ) -> Self {
        let binding = bindings.get(idx);
        let undecorated = answers.get_idx(binding.undecorated_idx).unwrap();
        let ty = answers.get_idx(idx).unwrap();
        DecoratedFunction {
            ty,
            idx,
            undecorated,
        }
    }

    pub fn metadata(&self) -> &FuncMetadata {
        &self.undecorated.metadata
    }

    pub fn id_range(&self) -> TextRange {
        self.undecorated.identifier.range()
    }

    pub fn defining_cls(&self) -> Option<&Class> {
        self.undecorated.defining_cls.as_ref()
    }

    pub fn is_stub(&self) -> bool {
        self.undecorated.stub_or_impl == FunctionStubOrImpl::Stub
    }

    pub fn is_impl(&self) -> bool {
        self.undecorated.stub_or_impl == FunctionStubOrImpl::Impl
    }

    pub fn is_overload(&self) -> bool {
        self.undecorated.metadata.flags.is_overload
    }
}

impl<'a, Ans: LookupAnswer> AnswersSolver<'a, Ans> {
    pub fn get_decorated_function(&self, idx: Idx<KeyDecoratedFunction>) -> DecoratedFunction {
        let binding = self.bindings().get(idx);
        let undecorated = self.get_idx(binding.undecorated_idx);
        let ty = self.get_idx(idx);
        DecoratedFunction {
            ty,
            idx,
            undecorated,
        }
    }

    pub fn get_function_successor(
        &self,
        def: &DecoratedFunction,
    ) -> Option<Idx<KeyDecoratedFunction>> {
        self.bindings().get(def.idx).successor
    }
}
