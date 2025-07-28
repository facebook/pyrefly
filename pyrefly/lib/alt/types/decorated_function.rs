/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::fmt;
use std::fmt::Debug;
use std::fmt::Display;

use pyrefly_derive::TypeEq;
use pyrefly_derive::Visit;
use pyrefly_derive::VisitMut;
use pyrefly_python::module_name::ModuleName;
use pyrefly_types::class::Class;
use ruff_python_ast::name::Name;
use ruff_text_size::TextRange;

use crate::binding::binding::FunctionStubOrImpl;
use crate::types::callable::FuncFlags;
use crate::types::callable::FuncId;
use crate::types::callable::FuncMetadata;
use crate::types::callable::FunctionKind;
use crate::types::types::Type;

/// The type of a function definition after decorators are applied. Metadata arising from the
/// decorators can be stored here. Note that the type might not be a function at all, since
/// decorators can produce any type.
#[derive(Clone, Debug, Visit, VisitMut, TypeEq, PartialEq, Eq)]
pub struct DecoratedFunction {
    pub id_range: TextRange,
    pub ty: Type,
    pub metadata: FuncMetadata,
    pub stub_or_impl: FunctionStubOrImpl,
    pub defining_cls: Option<Class>,
}

impl Display for DecoratedFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Display::fmt(&self.ty, f)
    }
}

impl DecoratedFunction {
    pub fn recursive() -> Self {
        DecoratedFunction {
            id_range: TextRange::default(),
            ty: Type::any_implicit(),
            metadata: FuncMetadata {
                kind: FunctionKind::Def(Box::new(FuncId {
                    module: ModuleName::from_str("__decorated_function_recursive__"),
                    cls: None,
                    func: Name::default(),
                })),
                flags: FuncFlags::default(),
            },
            stub_or_impl: FunctionStubOrImpl::Stub,
            defining_cls: None,
        }
    }
}
