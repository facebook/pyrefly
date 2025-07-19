/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::cmp::Ordering;
use std::fmt;
use std::fmt::Display;

use parse_display::Display;
use pyrefly_derive::TypeEq;
use pyrefly_derive::Visit;
use pyrefly_derive::VisitMut;
use pyrefly_util::uniques::Unique;
use pyrefly_util::uniques::UniqueFactory;
use ruff_python_ast::name::Name;

use crate::class::ClassType;
use crate::stdlib::Stdlib;
use crate::type_var::Restriction;
use crate::types::Type;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Ord, PartialOrd)]
#[derive(Visit, VisitMut, TypeEq)]
pub struct QuantifiedInfo {
    pub name: Name,
    pub kind: QuantifiedKind,
    pub default: Option<Type>,
    pub restriction: Restriction,
}

impl QuantifiedInfo {
    fn as_gradual_type_helper(&self, default: Option<&Type>) -> Type {
        default.map_or_else(
            || self.kind.empty_value(),
            |default| {
                default.clone().transform(&mut |default| match default {
                    Type::TypeVar(t) => {
                        *default = Self::type_var(
                            t.qname().id().clone(),
                            t.default().cloned(),
                            t.restriction().clone(),
                        )
                        .as_gradual_type();
                    }
                    Type::TypeVarTuple(t) => {
                        *default =
                            Self::type_var_tuple(t.qname().id().clone(), t.default().cloned())
                                .as_gradual_type();
                    }
                    Type::ParamSpec(p) => {
                        *default = Self::param_spec(p.qname().id().clone(), p.default().cloned())
                            .as_gradual_type();
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
        self.as_gradual_type_helper(self.default.as_ref())
    }

    fn type_var(name: Name, default: Option<Type>, restriction: Restriction) -> Self {
        Self {
            name,
            kind: QuantifiedKind::TypeVar,
            restriction,
            default,
        }
    }

    fn param_spec(name: Name, default: Option<Type>) -> Self {
        Self {
            name,
            kind: QuantifiedKind::ParamSpec,
            restriction: Restriction::Unrestricted,
            default,
        }
    }

    fn type_var_tuple(name: Name, default: Option<Type>) -> Self {
        Self {
            name,
            kind: QuantifiedKind::TypeVarTuple,
            restriction: Restriction::Unrestricted,
            default,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[derive(Visit, VisitMut, TypeEq)]
pub struct Quantified {
    /// Unique identifier
    unique: Unique,
    info: Box<QuantifiedInfo>,
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
        self.info
            .cmp(&other.info)
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
}

impl Display for Quantified {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.info.name)
    }
}

impl Quantified {
    pub fn new(unique: Unique, info: QuantifiedInfo) -> Self {
        Quantified {
            unique,
            info: Box::new(info),
        }
    }

    pub fn type_var(
        name: Name,
        uniques: &UniqueFactory,
        default: Option<Type>,
        restriction: Restriction,
    ) -> Self {
        Self::new(
            uniques.fresh(),
            QuantifiedInfo::type_var(name, default, restriction),
        )
    }

    pub fn param_spec(name: Name, uniques: &UniqueFactory, default: Option<Type>) -> Self {
        Self::new(uniques.fresh(), QuantifiedInfo::param_spec(name, default))
    }

    pub fn type_var_tuple(name: Name, uniques: &UniqueFactory, default: Option<Type>) -> Self {
        Self::new(
            uniques.fresh(),
            QuantifiedInfo::type_var_tuple(name, default),
        )
    }

    pub fn to_type(self) -> Type {
        Type::Quantified(self)
    }

    pub fn as_value<'a>(&self, stdlib: &'a Stdlib) -> &'a ClassType {
        match self.info.kind {
            QuantifiedKind::TypeVar => stdlib.type_var(),
            QuantifiedKind::ParamSpec => stdlib.param_spec(),
            QuantifiedKind::TypeVarTuple => stdlib.type_var_tuple(),
        }
    }

    pub fn name(&self) -> &Name {
        &self.info.name
    }

    pub fn kind(&self) -> QuantifiedKind {
        self.info.kind
    }

    pub fn default(&self) -> Option<&Type> {
        self.info.default.as_ref()
    }

    pub fn restriction(&self) -> &Restriction {
        &self.info.restriction
    }

    pub fn is_type_var(&self) -> bool {
        matches!(self.info.kind, QuantifiedKind::TypeVar)
    }

    pub fn is_param_spec(&self) -> bool {
        matches!(self.info.kind, QuantifiedKind::ParamSpec)
    }

    pub fn is_type_var_tuple(&self) -> bool {
        matches!(self.info.kind, QuantifiedKind::TypeVarTuple)
    }

    pub fn as_gradual_type(&self) -> Type {
        self.info.as_gradual_type()
    }
}
