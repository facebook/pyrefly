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
use pyrefly_python::module_name::ModuleName;
use pyrefly_util::display::Fmt;
use pyrefly_util::visit::Visit;
use pyrefly_util::visit::VisitMut;
use ruff_python_ast::name::Name;
use ruff_text_size::TextRange;

use crate::class::ClassType;
use crate::heap::TypeHeap;
use crate::stdlib::Stdlib;
use crate::type_var::PreInferenceVariance;
use crate::type_var::Restriction;
use crate::type_var::TypeVar;
use crate::types::Type;

/// Discriminator for the origin of a `Quantified`, making collisions structurally impossible
/// between quantifieds that share the same anchor range but have different origins.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum QuantifiedOrigin {
    /// Legacy TypeVar scoped to a function/class/alias owner.
    ScopedLegacy,
    /// PEP 695 type parameter — has its own definition range, no ambiguity.
    Pep695,
    /// Synthetic Self quantified synthesized for `__new__` on a class.
    SyntheticSelf,
    /// Synthetic binder created during callable/tuple instantiation (TypeVarTuple residual).
    SyntheticCallableResidual,
}

/// A source range plus an index that disambiguates multiple quantifieds sharing the same range.
/// Index 0 is used when a range produces exactly one quantified; higher indices are used when
/// a single range produces several (e.g. multiple type parameters anchored to the same scope).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AnchorIndex {
    pub range: TextRange,
    pub index: u32,
}

impl AnchorIndex {
    pub fn new(range: TextRange, index: u32) -> Self {
        Self { range, index }
    }

    pub fn first(range: TextRange) -> Self {
        Self { range, index: 0 }
    }
}

impl PartialOrd for AnchorIndex {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for AnchorIndex {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // TextRange doesn't implement Ord, so compare start/end as u32.
        u32::from(self.range.start())
            .cmp(&u32::from(other.range.start()))
            .then_with(|| u32::from(self.range.end()).cmp(&u32::from(other.range.end())))
            .then_with(|| self.index.cmp(&other.index))
    }
}

/// Deterministic identity for a `Quantified`, derived from source locations rather than
/// allocation order. Two quantifieds are the same iff their identity is the same.
///
/// Globally unique because `module` distinguishes cross-module collisions, `anchor` pins
/// the source location and index, and `origin` prevents collisions between quantifieds
/// of different kinds at the same anchor.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct QuantifiedIdentity {
    pub module: ModuleName,
    pub anchor: AnchorIndex,
    pub origin: QuantifiedOrigin,
}

impl PartialOrd for QuantifiedIdentity {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for QuantifiedIdentity {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.module
            .cmp(&other.module)
            .then_with(|| self.anchor.cmp(&other.anchor))
            .then_with(|| self.origin.cmp(&other.origin))
    }
}

impl QuantifiedIdentity {
    pub fn new(module: ModuleName, anchor: AnchorIndex, origin: QuantifiedOrigin) -> Self {
        Self {
            module,
            anchor,
            origin,
        }
    }
}

// None of these types contain Type values; they are visit leaves.
impl<To> Visit<To> for QuantifiedOrigin {
    const RECURSE_CONTAINS: bool = false;
    fn recurse<'a>(&'a self, _: &mut dyn FnMut(&'a To)) {}
}
impl<To> VisitMut<To> for QuantifiedOrigin {
    const RECURSE_CONTAINS: bool = false;
    fn recurse_mut(&mut self, _: &mut dyn FnMut(&mut To)) {}
}
impl<To> Visit<To> for AnchorIndex {
    const RECURSE_CONTAINS: bool = false;
    fn recurse<'a>(&'a self, _: &mut dyn FnMut(&'a To)) {}
}
impl<To> VisitMut<To> for AnchorIndex {
    const RECURSE_CONTAINS: bool = false;
    fn recurse_mut(&mut self, _: &mut dyn FnMut(&mut To)) {}
}
impl<To> Visit<To> for QuantifiedIdentity {
    const RECURSE_CONTAINS: bool = false;
    fn recurse<'a>(&'a self, _: &mut dyn FnMut(&'a To)) {}
}
impl<To> VisitMut<To> for QuantifiedIdentity {
    const RECURSE_CONTAINS: bool = false;
    fn recurse_mut(&mut self, _: &mut dyn FnMut(&mut To)) {}
}

#[derive(Debug, Clone, Eq)]
#[derive(Visit, VisitMut, TypeEq)]
pub struct Quantified {
    /// Deterministic identity based on source location.
    identity: QuantifiedIdentity,
    pub name: Name,
    pub kind: QuantifiedKind,
    pub default: Option<Type>,
    pub restriction: Restriction,
    /// The *declared* variance of this type parameter, as specified by the user.
    /// For function type parameters, variance has no meaning.
    /// We store it here for convenience of our variance inference and checking
    /// infrastructure so it can directly read it from the type.
    variance: PreInferenceVariance,
    /// Qualified owner, e.g. `"mod.func"`, set for function type params to enable
    /// disambiguation in display (e.g. `T@mod.func`).
    pub owner: Option<Name>,
}

impl Quantified {
    pub fn with_restriction(self, restriction: Restriction) -> Self {
        Self {
            restriction,
            identity: self.identity,
            name: self.name,
            kind: self.kind,
            default: self.default,
            variance: self.variance,
            owner: self.owner,
        }
    }
}

impl PartialEq for Quantified {
    fn eq(&self, other: &Self) -> bool {
        self.identity == other.identity
    }
}

impl Hash for Quantified {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.identity.hash(state);
    }
}

impl Ord for Quantified {
    fn cmp(&self, other: &Self) -> Ordering {
        // Identity is fully deterministic, so sort on it directly.
        // This is used to order types in a union (and deduplication relies on Ord).
        self.identity.cmp(&other.identity)
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

impl Display for QuantifiedIdentity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}@{}[{}]:{:?}",
            self.module,
            u32::from(self.anchor.range.start()),
            self.anchor.index,
            self.origin,
        )
    }
}

impl Display for Quantified {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

impl Quantified {
    pub fn new(
        identity: QuantifiedIdentity,
        name: Name,
        kind: QuantifiedKind,
        default: Option<Type>,
        restriction: Restriction,
        variance: PreInferenceVariance,
    ) -> Self {
        Quantified {
            identity,
            name,
            kind,
            default,
            restriction,
            variance,
            owner: None,
        }
    }

    pub fn with_owner(mut self, owner: Name) -> Self {
        self.owner = Some(owner);
        self
    }

    pub fn type_var(
        name: Name,
        identity: QuantifiedIdentity,
        default: Option<Type>,
        restriction: Restriction,
        variance: PreInferenceVariance,
    ) -> Self {
        Self::new(
            identity,
            name,
            QuantifiedKind::TypeVar,
            default,
            restriction,
            variance,
        )
    }

    /// Creates a Quantified from a TypeVar, extracting all relevant fields.
    pub fn from_type_var(tv: &TypeVar, identity: QuantifiedIdentity) -> Self {
        Self::type_var(
            tv.qname().id().clone(),
            identity,
            tv.default().cloned(),
            tv.restriction().clone(),
            tv.variance(),
        )
    }

    pub fn param_spec(name: Name, identity: QuantifiedIdentity, default: Option<Type>) -> Self {
        Self::new(
            identity,
            name,
            QuantifiedKind::ParamSpec,
            default,
            Restriction::Unrestricted,
            PreInferenceVariance::Invariant,
        )
    }

    pub fn type_var_tuple(name: Name, identity: QuantifiedIdentity, default: Option<Type>) -> Self {
        Self::new(
            identity,
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

    /// The upper bound of this type parameter as a type, accounting for the parameter's kind.
    /// For TypeVar the bound is `object`, for ParamSpec it's `...` (any params), and for
    /// TypeVarTuple it's an unbounded tuple. Explicit bounds and constraints are used as-is.
    pub fn bound_type(&self, stdlib: &Stdlib, heap: &TypeHeap) -> Type {
        match &self.restriction {
            Restriction::Unrestricted => match self.kind {
                QuantifiedKind::TypeVar => stdlib.object().clone().to_type(),
                QuantifiedKind::ParamSpec => Type::Ellipsis,
                QuantifiedKind::TypeVarTuple => Type::any_tuple(),
            },
            r => r.as_type(stdlib, heap),
        }
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

    pub fn identity(&self) -> &QuantifiedIdentity {
        &self.identity
    }

    pub fn as_gradual_type_helper(kind: QuantifiedKind, default: Option<&Type>) -> Type {
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
