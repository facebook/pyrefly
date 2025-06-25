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

use crate::alt::solve::TypeFormContext;
use crate::types::class::ClassType;
use crate::types::stdlib::Stdlib;
use crate::types::type_var::Restriction;
use crate::types::types::Type;

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
        // This is a bit dubious, as now the Ord and Eq instances don't quite align.
        // But we happen to know that we only use the Ord for sorting union,
        // and we'd really like to have a deterministic sort order between runs.
        self.info.cmp(&other.info)
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

    pub fn type_form_context_for_default(self) -> TypeFormContext {
        match self {
            QuantifiedKind::TypeVar => TypeFormContext::TypeVarDefault,
            QuantifiedKind::ParamSpec => TypeFormContext::ParamSpecDefault,
            QuantifiedKind::TypeVarTuple => TypeFormContext::TypeVarTupleDefault,
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
