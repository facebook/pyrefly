/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::fmt;

use pyrefly_derive::TypeEq;
use pyrefly_derive::Visit;
use pyrefly_derive::VisitMut;
use pyrefly_util::display::commas_iter;
use ruff_python_ast::name::Name;

use crate::stdlib::Stdlib;
use crate::type_output::TypeOutput;
use crate::types::TParams;
use crate::types::Type;

/// The style of a type alias declaration.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(Visit, VisitMut, TypeEq)]
pub enum TypeAliasStyle {
    /// A type alias declared with the `type` keyword
    Scoped,
    /// A type alias declared with a `: TypeAlias` annotation
    LegacyExplicit,
    /// An unannotated assignment that may be either an implicit type alias or an untyped value
    LegacyImplicit,
}

/// A type alias, which may be scoped (PEP 695) or legacy.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(Visit, VisitMut, TypeEq)]
pub struct TypeAlias {
    pub name: Box<Name>,
    ty: Box<Type>,
    pub style: TypeAliasStyle,
    annotated_metadata: Box<[Type]>,
}

impl TypeAlias {
    pub fn new(name: Name, ty: Type, style: TypeAliasStyle, annotated_metadata: Vec<Type>) -> Self {
        Self {
            name: Box::new(name),
            ty: Box::new(ty),
            style,
            annotated_metadata: annotated_metadata.into_boxed_slice(),
        }
    }

    pub fn annotated_metadata(&self) -> &[Type] {
        &self.annotated_metadata
    }

    /// Gets the type contained within the type alias for use in a value
    /// position - for example, for a function call or attribute access.
    pub fn as_value(&self, stdlib: &Stdlib) -> Type {
        if self.style == TypeAliasStyle::Scoped {
            stdlib.type_alias_type().clone().to_type()
        } else {
            *self.ty.clone()
        }
    }

    /// Gets the type contained within the type alias for use in a type
    /// position - for example, in a variable type annotation. Note that
    /// the caller is still responsible for untyping the type. That is,
    /// `type X = int` is represented as `TypeAlias(X, type[int])`, and
    /// `as_type` returns `type[int]`; the caller must turn it into `int`.
    pub fn as_type(&self) -> Type {
        *self.ty.clone()
    }

    pub fn as_type_mut(&mut self) -> &mut Type {
        &mut self.ty
    }

    pub fn fmt_with_type<O: TypeOutput>(
        &self,
        output: &mut O,
        write_type: &impl Fn(&Type, &mut O) -> fmt::Result,
        tparams: Option<&TParams>,
    ) -> fmt::Result {
        match (&self.style, tparams) {
            (TypeAliasStyle::LegacyImplicit, _) => write_type(&self.ty, output),
            (_, None) => {
                output.write_str("TypeAlias[")?;
                output.write_str(self.name.as_str())?;
                output.write_str(", ")?;
                write_type(&self.ty, output)?;
                output.write_str("]")
            }
            (_, Some(tparams)) => {
                output.write_str("TypeAlias[")?;
                output.write_str(self.name.as_str())?;
                output.write_str("[")?;
                output.write_str(&format!("{}", commas_iter(|| tparams.iter())))?;
                output.write_str("], ")?;
                write_type(&self.ty, output)?;
                output.write_str("]")
            }
        }
    }

    pub fn error(name: Name, style: TypeAliasStyle) -> Self {
        Self::new(name, Type::any_error(), style, Vec::new())
    }
}
