/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::fmt;
use std::fmt::Display;
use std::hash::Hash;

use dupe::Dupe;
use pyrefly_derive::TypeEq;
use pyrefly_derive::Visit;
use pyrefly_derive::VisitMut;
use pyrefly_util::arc_id::ArcId;
use pyrefly_util::visit::Visit;
use pyrefly_util::visit::VisitMut;
use ruff_python_ast::Identifier;

use crate::module::module_info::ModuleInfo;
use crate::types::equality::TypeEq;
use crate::types::equality::TypeEqCtx;
use crate::types::qname::QName;
use crate::types::types::Type;

/// Used to represent TypeVar calls. Each TypeVar is unique, so use the ArcId to separate them.
#[derive(Clone, Dupe, Debug, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct TypeVar(ArcId<TypeVarInner>);

// This is a lie, we do have types in the bound position
impl Visit<Type> for TypeVar {
    const RECURSE_CONTAINS: bool = false;
    fn recurse<'a>(&'a self, _: &mut dyn FnMut(&'a Type)) {}
}

impl VisitMut<Type> for TypeVar {
    const RECURSE_CONTAINS: bool = false;
    fn recurse_mut(&mut self, _: &mut dyn FnMut(&mut Type)) {}
}

impl Display for TypeVar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0.qname.id())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Ord, PartialOrd, Hash)]
#[derive(Visit, VisitMut, TypeEq)]
pub enum Restriction {
    Constraints(Vec<Type>),
    Bound(Type),
    Unrestricted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Ord, PartialOrd, Hash)]
#[derive(Visit, VisitMut, TypeEq)]
pub enum PreInferenceVariance {
    PCovariant,
    PContravariant,
    PInvariant,
    PUndefined,
}

impl Display for PreInferenceVariance {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PreInferenceVariance::PCovariant => write!(f, "PCovariant"),
            PreInferenceVariance::PContravariant => write!(f, "PContravariant"),
            PreInferenceVariance::PInvariant => write!(f, "PInvariant"),
            PreInferenceVariance::PUndefined => write!(f, "PUndefined"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Ord, PartialOrd, Hash)]
#[derive(Visit, VisitMut, TypeEq)]
pub enum Variance {
    Covariant,
    Contravariant,
    Invariant,
    #[allow(dead_code)]
    Bivariant,
}

impl Display for Variance {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Variance::Covariant => write!(f, "covariant"),
            Variance::Contravariant => write!(f, "contravariant"),
            Variance::Invariant => write!(f, "invariant"),
            Variance::Bivariant => write!(f, "bivariant"),
        }
    }
}

#[derive(Debug, PartialEq, TypeEq, Eq, Ord, PartialOrd)]
struct TypeVarInner {
    qname: QName,
    restriction: Restriction,
    default: Option<Type>,
    /// The variance if known, or None for infer_variance=True
    variance: PreInferenceVariance,
}

impl TypeVar {
    pub fn new(
        name: Identifier,
        module: ModuleInfo,
        restriction: Restriction,
        default: Option<Type>,
        variance: PreInferenceVariance,
    ) -> Self {
        Self(ArcId::new(TypeVarInner {
            qname: QName::new(name, module),
            restriction,
            default,
            variance,
        }))
    }

    pub fn qname(&self) -> &QName {
        &self.0.qname
    }

    pub fn restriction(&self) -> &Restriction {
        &self.0.restriction
    }

    pub fn default(&self) -> Option<&Type> {
        self.0.default.as_ref()
    }

    pub fn variance(&self) -> PreInferenceVariance {
        self.0.variance
    }

    pub fn to_type(&self) -> Type {
        Type::TypeVar(self.dupe())
    }

    pub fn type_eq_inner(&self, other: &Self, ctx: &mut TypeEqCtx) -> bool {
        self.0.type_eq(&other.0, ctx)
    }
}
