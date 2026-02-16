/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::cmp::Ordering;
use std::fmt;
use std::fmt::Display;
use std::hash::Hash;
use std::hash::Hasher;

use parse_display::Display;
use pyrefly_derive::TypeEq;
use pyrefly_derive::Visit;
use pyrefly_derive::VisitMut;
use pyrefly_util::display::Fmt;
use pyrefly_util::uniques::Unique;
use pyrefly_util::uniques::UniqueFactory;
use ruff_python_ast::name::Name;

use crate::class::ClassType;
use crate::heap::TypeHeap;
use crate::stdlib::Stdlib;
use crate::type_var::PreInferenceVariance;
use crate::type_var::Restriction;
use crate::type_var::TypeVar;
use crate::types::Type;

#[derive(Debug, Clone, Eq)]
#[derive(Visit, VisitMut, TypeEq)]
pub struct Quantified {
    /// Unique identifier
    unique: Unique,
    pub name: Name,
    pub kind: QuantifiedKind,
    pub default: Option<Type>,
    pub restriction: Restriction,
    /// The *declared* variance of this type parameter, as specified by the user
    /// For function type parameters, variance has no meaning
    /// We store it here for convenience of our variance inference and checking
    /// infrastructure so it can directly read it from the type
    variance: PreInferenceVariance,
}

impl Quantified {
    pub fn with_restriction(self, restriction: Restriction) -> Self {
        Self {
            restriction,
            unique: self.unique,
            name: self.name,
            kind: self.kind,
            default: self.default,
            variance: self.variance,
        }
    }
}

impl PartialEq for Quantified {
    fn eq(&self, other: &Self) -> bool {
        self.unique == other.unique
    }
}

impl Hash for Quantified {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.unique.hash(state);
    }
}

impl Ord for Quantified {
    fn cmp(&self, other: &Self) -> Ordering {
        // This function wants to serve two purposes, and currently we can't do both,
        // so we compromise. The Ord is used to order the types in a union. Problems:
        //
        // 1. The `Unique` is non-deterministic, so if you sort on it, types like
        //    Q.a and Q.b will not be sorted consistently.
        // 2. For a union we deduplicate adjacent elements, meaning we do need to sort
        //    on the unique to deduplicate (see test_quantified_accumulation for if)
        //    we don't.
        //
        // So we sort on unique last, which is slightly better, solves 2. but leaves
        // 1. as a partial problem.
        self.name
            .cmp(&other.name)
            .then_with(|| self.kind.cmp(&other.kind))
            .then_with(|| self.default.cmp(&other.default))
            .then_with(|| self.restriction.cmp(&other.restriction))
            .then_with(|| self.variance.cmp(&other.variance))
            .then_with(|| self.unique.cmp(&other.unique))
    }
}

impl PartialOrd for Quantified {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Display)]
#[derive(Visit, VisitMut, TypeEq)]
pub enum QuantifiedKind {
    TypeVar,
    ParamSpec,
    TypeVarTuple,
}

impl QuantifiedKind {
    fn empty_value(self) -> Type {
        match self {
            QuantifiedKind::TypeVar => Type::any_implicit(),
            QuantifiedKind::ParamSpec => Type::Ellipsis,
            QuantifiedKind::TypeVarTuple => Type::any_tuple(),
        }
    }

    fn class_type(self, stdlib: &Stdlib) -> &ClassType {
        match self {
            QuantifiedKind::TypeVar => stdlib.type_var(),
            QuantifiedKind::ParamSpec => stdlib.param_spec(),
            QuantifiedKind::TypeVarTuple => stdlib.type_var_tuple(),
        }
    }
}

impl Display for Quantified {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

impl Quantified {
    pub fn new(
        unique: Unique,
        name: Name,
        kind: QuantifiedKind,
        default: Option<Type>,
        restriction: Restriction,
        variance: PreInferenceVariance,
    ) -> Self {
        Quantified {
            unique,
            name,
            kind,
            default,
            restriction,
            variance,
        }
    }

    pub fn type_var(
        name: Name,
        uniques: &UniqueFactory,
        default: Option<Type>,
        restriction: Restriction,
        variance: PreInferenceVariance,
    ) -> Self {
        Self::new(
            uniques.fresh(),
            name,
            QuantifiedKind::TypeVar,
            default,
            restriction,
            variance,
        )
    }

    /// Creates a Quantified from a TypeVar, extracting all relevant fields.
    pub fn from_type_var(tv: &TypeVar, uniques: &UniqueFactory) -> Self {
        Self::type_var(
            tv.qname().id().clone(),
            uniques,
            tv.default().cloned(),
            tv.restriction().clone(),
            tv.variance(),
        )
    }

    pub fn param_spec(name: Name, uniques: &UniqueFactory, default: Option<Type>) -> Self {
        Self::new(
            uniques.fresh(),
            name,
            QuantifiedKind::ParamSpec,
            default,
            Restriction::Unrestricted,
            PreInferenceVariance::Invariant,
        )
    }

    pub fn type_var_tuple(name: Name, uniques: &UniqueFactory, default: Option<Type>) -> Self {
        Self::new(
            uniques.fresh(),
            name,
            QuantifiedKind::TypeVarTuple,
            default,
            Restriction::Unrestricted,
            PreInferenceVariance::Invariant,
        )
    }

    pub fn to_type(self, heap: &TypeHeap) -> Type {
        heap.mk_quantified(self)
    }

    pub fn to_value(self) -> Type {
        Type::QuantifiedValue(Box::new(self))
    }

    pub fn class_type<'a>(&self, stdlib: &'a Stdlib) -> &'a ClassType {
        self.kind.class_type(stdlib)
    }

    pub fn name(&self) -> &Name {
        &self.name
    }

    pub fn kind(&self) -> QuantifiedKind {
        self.kind
    }

    pub fn default(&self) -> Option<&Type> {
        self.default.as_ref()
    }

    pub fn restriction(&self) -> &Restriction {
        &self.restriction
    }

    /// Display this type parameter with its bounds/constraints and default,
    /// in the format used for type parameter lists (e.g. `T: int = str`).
    pub fn display_with_bounds(&self) -> impl Display + '_ {
        Fmt(move |f| {
            write!(f, "{}", self.name)?;
            match self.restriction() {
                Restriction::Bound(t) => write!(f, ": {}", t)?,
                Restriction::Constraints(ts) => {
                    write!(f, ": (")?;
                    for (i, t) in ts.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{}", t)?;
                    }
                    write!(f, ")")?;
                }
                Restriction::Unrestricted => {}
            }
            if let Some(default) = self.default() {
                write!(f, " = {}", default)?;
            }
            Ok(())
        })
    }

    pub fn variance(&self) -> PreInferenceVariance {
        self.variance
    }

    pub fn is_type_var(&self) -> bool {
        matches!(self.kind, QuantifiedKind::TypeVar)
    }

    pub fn is_param_spec(&self) -> bool {
        matches!(self.kind, QuantifiedKind::ParamSpec)
    }

    pub fn is_type_var_tuple(&self) -> bool {
        matches!(self.kind, QuantifiedKind::TypeVarTuple)
    }

    fn as_gradual_type_helper(kind: QuantifiedKind, default: Option<&Type>) -> Type {
        default.map_or_else(
            || kind.empty_value(),
            |default| {
                default.clone().transform(&mut |default| match default {
                    Type::TypeVar(t) => {
                        *default =
                            Self::as_gradual_type_helper(QuantifiedKind::TypeVar, t.default())
                    }
                    Type::TypeVarTuple(t) => {
                        *default =
                            Self::as_gradual_type_helper(QuantifiedKind::TypeVarTuple, t.default())
                    }
                    Type::ParamSpec(p) => {
                        *default =
                            Self::as_gradual_type_helper(QuantifiedKind::ParamSpec, p.default())
                    }
                    Type::Quantified(q) => {
                        *default = q.as_gradual_type();
                    }
                    _ => {}
                })
            },
        )
    }

    pub fn as_gradual_type(&self) -> Type {
        Self::as_gradual_type_helper(self.kind(), self.default())
    }
}
