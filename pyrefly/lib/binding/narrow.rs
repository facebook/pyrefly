/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::fmt;

use pyrefly_python::ast::Ast;
use pyrefly_util::assert_words;
use pyrefly_util::display::DisplayWith;
use pyrefly_util::display::DisplayWithCtx;
use pyrefly_util::display::commas_iter;
use pyrefly_util::prelude::SliceExt;
use ruff_python_ast::Arguments;
use ruff_python_ast::BoolOp;
use ruff_python_ast::CmpOp;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprAttribute;
use ruff_python_ast::ExprBoolOp;
use ruff_python_ast::ExprCall;
use ruff_python_ast::ExprCompare;
use ruff_python_ast::ExprNamed;
use ruff_python_ast::ExprNumberLiteral;
use ruff_python_ast::ExprStringLiteral;
use ruff_python_ast::ExprSubscript;
use ruff_python_ast::ExprUnaryOp;
use ruff_python_ast::Identifier;
use ruff_python_ast::Number;
use ruff_python_ast::UnaryOp;
use ruff_python_ast::name::Name;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use starlark_map::Hashed;
use starlark_map::small_map::Entry;
use starlark_map::small_map::SmallMap;
use starlark_map::small_set::SmallSet;
use vec1::Vec1;

use crate::binding::binding::Binding;
use crate::binding::binding::Key;
use crate::binding::bindings::BindingsBuilder;
use crate::binding::scope::NameReadInfo;
use crate::export::special::SpecialExport;
use crate::graph::index::Idx;
use crate::module::module_info::ModuleInfo;
use crate::types::facet::FacetChain;
use crate::types::facet::FacetKind;
use crate::types::types::Type;

assert_words!(AtomicNarrowOp, 11);
assert_words!(NarrowOp, 16);

#[derive(Clone, Debug)]
pub enum AtomicNarrowOp {
    Is(Expr),
    IsNot(Expr),
    Eq(Expr),
    NotEq(Expr),
    IsInstance(Expr),
    IsNotInstance(Expr),
    IsSubclass(Expr),
    IsNotSubclass(Expr),
    HasAttr(Name),
    NotHasAttr(Name),
    GetAttr(Name, Option<Box<Expr>>),
    NotGetAttr(Name, Option<Box<Expr>>),
    TypeGuard(Type, Arguments),
    NotTypeGuard(Type, Arguments),
    TypeIs(Type, Arguments),
    NotTypeIs(Type, Arguments),
    // type(x) == y or type(x) is y
    TypeEq(Expr),
    TypeNotEq(Expr),
    In(Expr),
    NotIn(Expr),
    /// Used to narrow tuple types based on length
    LenEq(Expr),
    LenNotEq(Expr),
    LenGt(Expr),
    LenGte(Expr),
    LenLt(Expr),
    LenLte(Expr),
    /// (func, args) for a function call that may narrow the type of its first argument.
    Call(Box<Expr>, Arguments),
    NotCall(Box<Expr>, Arguments),
    /// A narrow op applies to a name; these operations mean we are narrowing to the case
    /// when that name evaluates to a truthy or falsy value.
    IsTruthy,
    IsFalsy,
    /// An operation that might be true or false, but does not narrow the name
    /// currently under consideration (for example, if we are modeling the
    /// narrowing for name `x` from `x is None or y is None`). We need to
    /// preserve its existence in order to handle control flow and negation
    Placeholder,
}

#[derive(Clone, Debug)]
pub enum NarrowOp {
    Atomic(Option<FacetSubject>, AtomicNarrowOp),
    And(Vec<NarrowOp>),
    Or(Vec<NarrowOp>),
}

impl DisplayWith<ModuleInfo> for AtomicNarrowOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>, ctx: &ModuleInfo) -> fmt::Result {
        match self {
            AtomicNarrowOp::Is(expr) => write!(f, "Is({})", expr.display_with(ctx)),
            AtomicNarrowOp::IsNot(expr) => write!(f, "IsNot({})", expr.display_with(ctx)),
            AtomicNarrowOp::Eq(expr) => write!(f, "Eq({})", expr.display_with(ctx)),
            AtomicNarrowOp::NotEq(expr) => write!(f, "NotEq({})", expr.display_with(ctx)),
            AtomicNarrowOp::IsInstance(expr) => write!(f, "IsInstance({})", expr.display_with(ctx)),
            AtomicNarrowOp::IsNotInstance(expr) => {
                write!(f, "IsNotInstance({})", expr.display_with(ctx))
            }
            AtomicNarrowOp::IsSubclass(expr) => write!(f, "IsSubclass({})", expr.display_with(ctx)),
            AtomicNarrowOp::IsNotSubclass(expr) => {
                write!(f, "IsNotSubclass({})", expr.display_with(ctx))
            }
            AtomicNarrowOp::HasAttr(attr) => {
                write!(f, "HasAttr({attr})")
            }
            AtomicNarrowOp::NotHasAttr(attr) => {
                write!(f, "NotHasAttr({attr})")
            }
            AtomicNarrowOp::GetAttr(attr, Some(default)) => {
                write!(f, "GetAttr({}, {})", attr, default.display_with(ctx))
            }
            AtomicNarrowOp::NotGetAttr(attr, Some(default)) => {
                write!(f, "NotGetAttr({}, {})", attr, default.display_with(ctx))
            }
            AtomicNarrowOp::GetAttr(attr, None) => {
                write!(f, "GetAttr({attr}, None)")
            }
            AtomicNarrowOp::NotGetAttr(attr, None) => {
                write!(f, "NotGetAttr({attr}, None)")
            }
            AtomicNarrowOp::TypeGuard(t, arguments) => {
                write!(f, "TypeGuard({t}, {})", arguments.display_with(ctx))
            }
            AtomicNarrowOp::NotTypeGuard(t, arguments) => {
                write!(f, "NotTypeGuard({t}, {})", arguments.display_with(ctx))
            }
            AtomicNarrowOp::TypeIs(t, arguments) => {
                write!(f, "TypeIs({t}, {})", arguments.display_with(ctx))
            }
            AtomicNarrowOp::NotTypeIs(t, arguments) => {
                write!(f, "NotTypeIs({t}, {})", arguments.display_with(ctx))
            }
            AtomicNarrowOp::TypeEq(expr) => write!(f, "TypeEq({})", expr.display_with(ctx)),
            AtomicNarrowOp::TypeNotEq(expr) => write!(f, "TypeNotEq({})", expr.display_with(ctx)),
            AtomicNarrowOp::In(expr) => write!(f, "In({})", expr.display_with(ctx)),
            AtomicNarrowOp::NotIn(expr) => write!(f, "NotIn({})", expr.display_with(ctx)),
            AtomicNarrowOp::LenEq(expr) => write!(f, "LenEq({})", expr.display_with(ctx)),
            AtomicNarrowOp::LenNotEq(expr) => write!(f, "LenNotEq({})", expr.display_with(ctx)),
            AtomicNarrowOp::LenGt(expr) => write!(f, "LenGt({})", expr.display_with(ctx)),
            AtomicNarrowOp::LenGte(expr) => write!(f, "LenGte({})", expr.display_with(ctx)),
            AtomicNarrowOp::LenLt(expr) => write!(f, "LenLt({})", expr.display_with(ctx)),
            AtomicNarrowOp::LenLte(expr) => write!(f, "LenLte({})", expr.display_with(ctx)),
            AtomicNarrowOp::Call(expr, arguments) => write!(
                f,
                "Call({}, {})",
                expr.display_with(ctx),
                arguments.display_with(ctx)
            ),
            AtomicNarrowOp::NotCall(expr, arguments) => write!(
                f,
                "NotCall({}, {})",
                expr.display_with(ctx),
                arguments.display_with(ctx)
            ),
            AtomicNarrowOp::IsTruthy => write!(f, "IsTruthy"),
            AtomicNarrowOp::IsFalsy => write!(f, "IsFalsy"),
            AtomicNarrowOp::Placeholder => write!(f, "Placeholder"),
        }
    }
}

impl DisplayWith<ModuleInfo> for NarrowOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>, ctx: &ModuleInfo) -> fmt::Result {
        match self {
            Self::Atomic(prop, op) => match prop {
                None => write!(f, "{}", op.display_with(ctx)),
                Some(prop) => {
                    write!(f, "[")?;
                    prop.fmt_chain(f, ctx)?;
                    write!(f, "] {}", op.display_with(ctx))
                }
            },
            Self::And(ops) => {
                write!(
                    f,
                    "And({})",
                    commas_iter(|| ops.iter().map(|op| op.display_with(ctx)))
                )
            }
            Self::Or(ops) => {
                write!(
                    f,
                    "Or({})",
                    commas_iter(|| ops.iter().map(|op| op.display_with(ctx)))
                )
            }
        }
    }
}

impl AtomicNarrowOp {
    pub fn negate(&self) -> Self {
        match self {
            Self::Is(v) => Self::IsNot(v.clone()),
            Self::IsNot(v) => Self::Is(v.clone()),
            Self::IsInstance(v) => Self::IsNotInstance(v.clone()),
            Self::IsNotInstance(v) => Self::IsInstance(v.clone()),
            Self::IsSubclass(v) => Self::IsNotSubclass(v.clone()),
            Self::IsNotSubclass(v) => Self::IsSubclass(v.clone()),
            Self::HasAttr(attr) => Self::NotHasAttr(attr.clone()),
            Self::NotHasAttr(attr) => Self::HasAttr(attr.clone()),
            Self::GetAttr(attr, default) => Self::NotGetAttr(attr.clone(), default.clone()),
            Self::NotGetAttr(attr, default) => Self::GetAttr(attr.clone(), default.clone()),
            Self::Eq(v) => Self::NotEq(v.clone()),
            Self::NotEq(v) => Self::Eq(v.clone()),
            Self::In(v) => Self::NotIn(v.clone()),
            Self::NotIn(v) => Self::In(v.clone()),
            Self::LenEq(v) => Self::LenNotEq(v.clone()),
            Self::LenGt(v) => Self::LenLte(v.clone()),
            Self::LenGte(v) => Self::LenLt(v.clone()),
            Self::LenLte(v) => Self::LenGt(v.clone()),
            Self::LenLt(v) => Self::LenGte(v.clone()),
            Self::LenNotEq(v) => Self::LenEq(v.clone()),
            Self::TypeGuard(ty, args) => Self::NotTypeGuard(ty.clone(), args.clone()),
            Self::NotTypeGuard(ty, args) => Self::TypeGuard(ty.clone(), args.clone()),
            Self::TypeIs(ty, args) => Self::NotTypeIs(ty.clone(), args.clone()),
            Self::NotTypeIs(ty, args) => Self::TypeIs(ty.clone(), args.clone()),
            Self::TypeEq(v) => Self::TypeNotEq(v.clone()),
            Self::TypeNotEq(v) => Self::TypeEq(v.clone()),
            Self::Call(f, args) => Self::NotCall(f.clone(), args.clone()),
            Self::NotCall(f, args) => Self::Call(f.clone(), args.clone()),
            Self::IsTruthy => Self::IsFalsy,
            Self::IsFalsy => Self::IsTruthy,
            Self::Placeholder => Self::Placeholder,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FacetOrigin {
    // This facet is a direct access, like `x.y`, `x[0]`, `x["key"]`
    Direct,
    // This facet came from a call to a `get` method, like `x.get("key")`
    GetMethod,
}

#[derive(Clone, Debug)]
pub struct FacetSubject {
    pub resolved: Vec<FacetKind>,
    pub pending: Option<Box<Expr>>,
    pub origin: FacetOrigin,
}

impl FacetSubject {
    pub fn to_chain(&self) -> Option<FacetChain> {
        if self.pending.is_some() {
            return None;
        }
        Vec1::try_from_vec(self.resolved.clone())
            .ok()
            .map(FacetChain::new)
    }

    fn fmt_chain(&self, f: &mut fmt::Formatter<'_>, ctx: &ModuleInfo) -> fmt::Result {
        for facet in &self.resolved {
            write!(f, "{facet}")?;
        }
        if let Some(pending) = self.pending.as_deref() {
            write!(f, "[{}]", pending.display_with(ctx))?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub enum NarrowingSubject {
    Name(Name),
    Facets(Name, FacetSubject),
}

impl NarrowingSubject {
    pub fn with_facet(&self, prop: FacetKind) -> Self {
        match self {
            Self::Name(name) => Self::Facets(
                name.clone(),
                FacetSubject {
                    resolved: vec![prop],
                    pending: None,
                    origin: FacetOrigin::Direct,
                },
            ),
            Self::Facets(name, facets) => {
                let mut resolved = facets.resolved.clone();
                if facets.pending.is_none() {
                    resolved.push(prop);
                }
                Self::Facets(
                    name.clone(),
                    FacetSubject {
                        resolved,
                        pending: facets.pending.clone(),
                        origin: facets.origin,
                    },
                )
            }
        }
    }
}

impl NarrowOp {
    pub fn negate(&self) -> Self {
        match self {
            Self::Atomic(attr, op) => Self::Atomic(attr.clone(), op.negate()),
            Self::And(ops) => Self::Or(ops.map(|op| op.negate())),
            Self::Or(ops) => Self::And(ops.map(|op| op.negate())),
        }
    }

    fn and(&mut self, other: Self) {
        match self {
            Self::And(ops) => ops.push(other),
            _ => *self = Self::And(vec![self.clone(), other]),
        }
    }

    fn or(&mut self, other: Self) {
        match self {
            Self::Or(ops) => ops.push(other),
            _ => *self = Self::Or(vec![self.clone(), other]),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct NarrowOps(pub SmallMap<Name, (NarrowOp, TextRange)>);

impl NarrowOps {
    pub fn new() -> Self {
        Self(SmallMap::new())
    }

    pub fn negate(&self) -> Self {
        Self(
            self.0
                .iter()
                .map(|(name, (op, range))| (name.clone(), (op.negate(), *range)))
                .collect(),
        )
    }

    fn get_or_placeholder(&mut self, name: Name, range: TextRange) -> &mut NarrowOp {
        &mut self
            .0
            .entry(name)
            .or_insert((NarrowOp::Atomic(None, AtomicNarrowOp::Placeholder), range))
            .0
    }

    fn and(&mut self, name: Name, op: NarrowOp, range: TextRange) {
        let existing_op = self.get_or_placeholder(name, range);
        existing_op.and(op)
    }

    pub fn and_all(&mut self, other: Self) {
        let mut seen = SmallSet::new();
        for (name, (op, range)) in other.0 {
            seen.insert(name.clone());
            self.and(name, op, range);
        }
        // For names present in `self` but not `other`, `And` their narrows with a placeholder
        let unmerged_names: Vec<_> = self
            .0
            .keys()
            .filter_map(|name| {
                if seen.contains(name) {
                    None
                } else {
                    Some(name.clone())
                }
            })
            .collect();
        for name in unmerged_names {
            if let Entry::Occupied(mut entry) = self.0.entry(name) {
                entry
                    .get_mut()
                    .0
                    .and(NarrowOp::Atomic(None, AtomicNarrowOp::Placeholder));
            }
        }
    }

    fn or(&mut self, name: Name, op: NarrowOp, range: TextRange) {
        let existing_op = self.get_or_placeholder(name, range);
        existing_op.or(op)
    }

    pub fn or_all(&mut self, other: Self) {
        let mut seen = SmallSet::new();
        for (name, (op, range)) in other.0 {
            seen.insert(name.clone());
            self.or(name, op, range);
        }
        // For names present in `self` but not `other`, `Or` their narrows with a placeholder
        let unmerged_names: Vec<_> = self
            .0
            .keys()
            .filter_map(|name| {
                if seen.contains(name) {
                    None
                } else {
                    Some(name.clone())
                }
            })
            .collect();
        for name in unmerged_names {
            if let Entry::Occupied(mut entry) = self.0.entry(name) {
                entry
                    .get_mut()
                    .0
                    .or(NarrowOp::Atomic(None, AtomicNarrowOp::Placeholder));
            }
        }
    }

    pub fn from_single_narrow_op(left: &Expr, op: AtomicNarrowOp, range: TextRange) -> Self {
        let mut narrow_ops = Self::new();
        for subject in expr_to_subjects(left) {
            let (name, prop) = match subject {
                NarrowingSubject::Name(name) => (name, None),
                NarrowingSubject::Facets(name, facets) => (name, Some(facets)),
            };
            if let Some((existing, _)) = narrow_ops.0.get_mut(&name) {
                existing.and(NarrowOp::Atomic(prop, op.clone()));
            } else {
                narrow_ops
                    .0
                    .insert(name, (NarrowOp::Atomic(prop, op.clone()), range));
            }
        }
        narrow_ops
    }

    pub fn from_single_narrow_op_for_subject(
        subject: NarrowingSubject,
        op: AtomicNarrowOp,
        range: TextRange,
    ) -> Self {
        let mut narrow_ops = Self::new();
        let (name, prop) = match subject {
            NarrowingSubject::Name(name) => (name, None),
            NarrowingSubject::Facets(name, facets) => (name, Some(facets)),
        };
        if let Some((existing, _)) = narrow_ops.0.get_mut(&name) {
            existing.and(NarrowOp::Atomic(prop, op.clone()));
        } else {
            narrow_ops
                .0
                .insert(name, (NarrowOp::Atomic(prop, op.clone()), range));
        }
        narrow_ops
    }

    pub fn from_expr(builder: &BindingsBuilder, test: Option<&Expr>) -> Self {
        Self::from_expr_helper(builder, test, SmallSet::new())
    }

    fn from_expr_helper<'a>(
        builder: &BindingsBuilder,
        test: Option<&'a Expr>,
        mut seen: SmallSet<&'a Name>,
    ) -> Self {
        let Some(test) = test else {
            return Self::new();
        };
        match test {
            Expr::Compare(ExprCompare {
                node_index: _,
                range: _,
                left,
                ops: cmp_ops,
                comparators,
            }) => {
                // If the left expression is a call to `len()` or `getattr()`, we're narrowing the first argument
                let mut left = &**left;
                // If the left expression is a call to `getattr()` we store attribute name and default
                let mut getattr_name = None;
                let mut special_export = None;
                if let Expr::Call(ExprCall {
                    func, arguments, ..
                }) = left
                    && arguments.args.len() == 1
                    && arguments.keywords.is_empty()
                {
                    special_export = builder.as_special_export(func);
                    if matches!(
                        special_export,
                        Some(SpecialExport::Len | SpecialExport::BuiltinsType)
                    ) {
                        left = &arguments.args[0];
                    }
                }
                // If we have something like `getattr(x, "attr") != None` or `getattr(x, "attr", None) is not None`
                // we can perform a `hasattr(x, "attr")` narrow.
                if let Expr::Call(ExprCall {
                    func, arguments, ..
                }) = left
                    && arguments.keywords.is_empty()
                    && (arguments.args.len() == 2
                        || (arguments.args.len() == 3
                            && matches!(arguments.args[2], Expr::NoneLiteral(_))))
                    && let Expr::StringLiteral(ExprStringLiteral { value, .. }) = &arguments.args[1]
                {
                    special_export = builder.as_special_export(func);
                    if matches!(special_export, Some(SpecialExport::GetAttr)) {
                        left = &arguments.args[0];
                        getattr_name = Some(Name::new(value.to_string()));
                    }
                }
                let mut ops = cmp_ops
                    .iter()
                    .zip(comparators)
                    .filter_map(|(cmp_op, right)| {
                        let range = right.range();
                        let op = match (cmp_op, special_export) {
                            (CmpOp::Is | CmpOp::Eq, Some(SpecialExport::BuiltinsType)) => {
                                AtomicNarrowOp::TypeEq(right.clone())
                            }
                            (CmpOp::IsNot | CmpOp::NotEq, Some(SpecialExport::BuiltinsType)) => {
                                AtomicNarrowOp::TypeNotEq(right.clone())
                            }
                            (CmpOp::IsNot | CmpOp::NotEq, Some(SpecialExport::GetAttr))
                                if matches!(right, Expr::NoneLiteral(_))
                                    && let Some(attr) = &getattr_name =>
                            {
                                AtomicNarrowOp::HasAttr(attr.clone())
                            }
                            (_, Some(SpecialExport::GetAttr)) => {
                                return None;
                            }
                            (CmpOp::Is, None) => AtomicNarrowOp::Is(right.clone()),
                            (CmpOp::IsNot, None) => AtomicNarrowOp::IsNot(right.clone()),
                            (CmpOp::Eq, Some(SpecialExport::Len)) => {
                                AtomicNarrowOp::LenEq(right.clone())
                            }
                            (CmpOp::NotEq, Some(SpecialExport::Len)) => {
                                AtomicNarrowOp::LenNotEq(right.clone())
                            }
                            (CmpOp::Gt, Some(SpecialExport::Len)) => {
                                AtomicNarrowOp::LenGt(right.clone())
                            }
                            (CmpOp::GtE, Some(SpecialExport::Len)) => {
                                AtomicNarrowOp::LenGte(right.clone())
                            }
                            (CmpOp::Lt, Some(SpecialExport::Len)) => {
                                AtomicNarrowOp::LenLt(right.clone())
                            }
                            (CmpOp::LtE, Some(SpecialExport::Len)) => {
                                AtomicNarrowOp::LenLte(right.clone())
                            }
                            (CmpOp::Eq, _) => AtomicNarrowOp::Eq(right.clone()),
                            (CmpOp::NotEq, _) => AtomicNarrowOp::NotEq(right.clone()),
                            (CmpOp::In, None) => AtomicNarrowOp::In(right.clone()),
                            (CmpOp::NotIn, None) => AtomicNarrowOp::NotIn(right.clone()),
                            _ => {
                                return None;
                            }
                        };
                        Some((op, range))
                    });
                match ops.next() {
                    None => Self::new(),
                    Some((op, range)) => {
                        let mut narrow_ops = NarrowOps::from_single_narrow_op(left, op, range);
                        for (op, range) in ops {
                            narrow_ops.and_all(NarrowOps::from_single_narrow_op(left, op, range));
                        }
                        narrow_ops
                    }
                }
            }
            Expr::BoolOp(ExprBoolOp {
                node_index: _,
                range: _,
                op,
                values,
            }) => {
                let extend = match op {
                    BoolOp::And => NarrowOps::and_all,
                    BoolOp::Or => NarrowOps::or_all,
                };
                let mut exprs = values.iter();
                let mut narrow_ops = Self::from_expr_helper(builder, exprs.next(), seen.clone());
                for next_val in exprs {
                    extend(
                        &mut narrow_ops,
                        Self::from_expr_helper(builder, Some(next_val), seen.clone()),
                    )
                }
                narrow_ops
            }
            Expr::UnaryOp(ExprUnaryOp {
                node_index: _,
                range: _,
                op: UnaryOp::Not,
                operand: e,
            }) => Self::from_expr_helper(builder, Some(e), seen).negate(),
            Expr::Call(ExprCall {
                node_index: _,
                range,
                func,
                arguments,
            }) if builder.as_special_export(func) == Some(SpecialExport::Bool)
                && arguments.args.len() == 1
                && arguments.keywords.is_empty() =>
            {
                Self::from_single_narrow_op(&arguments.args[0], AtomicNarrowOp::IsTruthy, *range)
            }
            Expr::Call(ExprCall {
                node_index: _,
                range,
                func,
                arguments,
            }) if builder.as_special_export(func) == Some(SpecialExport::HasAttr)
                && arguments.args.len() == 2
                && arguments.keywords.is_empty()
                && let Expr::StringLiteral(ExprStringLiteral { value, .. }) =
                    &arguments.args[1] =>
            {
                Self::from_single_narrow_op(
                    &arguments.args[0],
                    AtomicNarrowOp::HasAttr(Name::new(value.to_string())),
                    *range,
                )
            }
            Expr::Call(ExprCall {
                node_index: _,
                range,
                func,
                arguments,
            }) if builder.as_special_export(func) == Some(SpecialExport::GetAttr)
                && (arguments.args.len() == 2 || arguments.args.len() == 3)
                && arguments.keywords.is_empty()
                && let Expr::StringLiteral(ExprStringLiteral { value, .. }) =
                    &arguments.args[1] =>
            {
                Self::from_single_narrow_op(
                    &arguments.args[0],
                    AtomicNarrowOp::GetAttr(
                        Name::new(value.to_string()),
                        if arguments.args.len() == 2 {
                            None
                        } else {
                            Some(Box::new(arguments.args[2].clone()))
                        },
                    ),
                    *range,
                )
            }
            e @ Expr::Call(call) if dict_get_subject_for_call_expr(call).is_some() => {
                // When the guard is something like `x.get("key")`, we narrow it like `x["key"]` if `x` resolves to a dict
                // in the answers step.
                // This cannot be a TypeGuard/TypeIs function call, since the first argument is a string literal
                Self::from_single_narrow_op(e, AtomicNarrowOp::IsTruthy, e.range())
            }
            Expr::Call(ExprCall {
                node_index: _,
                range,
                func,
                arguments: args @ Arguments { args: posargs, .. },
            }) if !posargs.is_empty() => {
                // This may be a function call that narrows the type of its first argument. Record
                // it as a possible narrowing operation that we'll resolve in the answers phase.
                Self::from_single_narrow_op(
                    &posargs[0],
                    AtomicNarrowOp::Call(Box::new((**func).clone()), args.clone()),
                    *range,
                )
            }
            Expr::Named(named) => {
                let mut target_narrow = Self::from_single_narrow_op(
                    &named.target,
                    AtomicNarrowOp::IsTruthy,
                    named.target.range(),
                );
                let value_narrow =
                    Self::from_expr_helper(builder, Some(*named.value.clone()).as_ref(), seen);
                // Merge the entries from the two `NarrowOps`
                // We don't use `and_all` because it always generates placeholders when the entry is not present.
                // This causes `Or` ops to be generated when the narrowing is negated, which is correct for
                // unrelated narrows but undesirable here because we know these two narrows are either both true or both false.
                for (name, (op, range)) in value_narrow.0 {
                    let existing_entry = target_narrow.0.entry(name);
                    match existing_entry {
                        Entry::Occupied(mut entry) => {
                            entry.get_mut().0.and(op.clone());
                        }
                        Entry::Vacant(entry) => {
                            entry.insert((op, range));
                        }
                    };
                }
                target_narrow
            }
            e @ Expr::Name(name) => {
                if !seen.insert(name.id()) {
                    Self::new()
                } else {
                    // Look up the definition of `name`.
                    let original_expr = match Self::get_original_binding(builder, &name.id) {
                        Some((
                            _,
                            Some(Binding::NameAssign {
                                expr: original_expr,
                                ..
                            }),
                        )) => Some(&**original_expr),
                        _ => None,
                    };
                    let mut ops = Self::from_expr_helper(builder, original_expr, seen);
                    ops.0.retain(|name, (op, op_range)| {
                        Self::op_is_still_valid(builder, name, op, *op_range)
                    });
                    // Merge the narrow ops from the original definition with IsTruthy(name).
                    ops.0.insert(
                        name.id.clone(),
                        (NarrowOp::Atomic(None, AtomicNarrowOp::IsTruthy), e.range()),
                    );
                    ops
                }
            }
            e => Self::from_single_narrow_op(e, AtomicNarrowOp::IsTruthy, e.range()),
        }
    }

    fn get_original_binding<'a>(
        builder: &'a BindingsBuilder,
        name: &Name,
    ) -> Option<(Idx<Key>, Option<&'a Binding>)> {
        let name_read_info = builder.scopes.look_up_name_for_read(Hashed::new(name));
        match name_read_info {
            NameReadInfo::Flow { idx, .. } => builder.get_original_binding(idx),
            _ => None,
        }
    }

    fn op_is_still_valid(
        builder: &BindingsBuilder,
        name: &Name,
        op: &NarrowOp,
        op_range: TextRange,
    ) -> bool {
        // Check (1) if `op` checks a property of `name` that can't be invalidated without
        // reassigning `name` and (2) whether `name` is reassigned after `op` is computed.
        match op {
            NarrowOp::And(ops) | NarrowOp::Or(ops) => ops
                .iter()
                .all(|op| Self::op_is_still_valid(builder, name, op, op_range)),
            // A non-None facet subject means we're narrowing something like an attribute or a dict item.
            NarrowOp::Atomic(Some(_), _) => false,
            NarrowOp::Atomic(None, op) => match op {
                AtomicNarrowOp::Is(..)
                | AtomicNarrowOp::IsNot(..)
                | AtomicNarrowOp::Eq(..)
                | AtomicNarrowOp::NotEq(..)
                // Technically the `__class__` attribute can be mutated, but code that does that
                // probably isn't statically analyzable anyway.
                | AtomicNarrowOp::IsInstance(..)
                | AtomicNarrowOp::IsNotInstance(..)
                | AtomicNarrowOp::IsSubclass(..)
                | AtomicNarrowOp::IsNotSubclass(..)
                | AtomicNarrowOp::TypeEq(..)
                | AtomicNarrowOp::TypeNotEq(..)
                // The len ops are only applied to tuples, which are immutable.
                | AtomicNarrowOp::LenEq(..)
                | AtomicNarrowOp::LenNotEq(..)
                | AtomicNarrowOp::LenGt(..)
                | AtomicNarrowOp::LenGte(..)
                | AtomicNarrowOp::LenLt(..)
                | AtomicNarrowOp::LenLte(..)
                // This is technically unsafe, because it marks arbitrary TypeGuard/TypeIs results
                // as still valid, but we need to allow this for `isinstance` and friends to work.
                | AtomicNarrowOp::Call(..)
                | AtomicNarrowOp::NotCall(..)
                // The only objects that have different truthy and falsy types
                // (True vs. False, empty vs. non-empty tuple, etc.) are immutable.
                | AtomicNarrowOp::IsTruthy
                | AtomicNarrowOp::IsFalsy
                | AtomicNarrowOp::Placeholder => match builder.scopes.binding_idx_for_name(name) {
                    // Make sure the last definition of `name` is before the narrowing operation,
                    // so we know that `name` hasn't been redefined post-narrowing.
                    Some((idx, _)) => builder.idx_to_key(idx).range().end() <= op_range.start(),
                    None => true,
                },
                _ => false,
            },
        }
    }
}

/// Given an expression, determine whether it is a chain of properties (attribute/concrete index)
/// rooted at a name, and if so, return the name and the facets (allowing a trailing unresolved subscript).
/// For example: x.y[0].z, or x["key"], or x[key] (pending facet).
pub(crate) fn identifier_and_facet_subject_for_expr(
    expr: &Expr,
) -> Option<(Identifier, FacetSubject)> {
    #[derive(Clone)]
    enum FacetStep<'a> {
        Attribute(&'a ExprAttribute),
        Subscript(&'a ExprSubscript),
    }

    fn collect_steps<'a>(expr: &'a Expr, steps: &mut Vec<FacetStep<'a>>) -> Option<Identifier> {
        match expr {
            Expr::Name(name) => Some(Ast::expr_name_identifier(name.clone())),
            Expr::Attribute(attr) => {
                let ident = collect_steps(&attr.value, steps)?;
                steps.push(FacetStep::Attribute(attr));
                Some(ident)
            }
            Expr::Subscript(subscript) => {
                let ident = collect_steps(&subscript.value, steps)?;
                steps.push(FacetStep::Subscript(subscript));
                Some(ident)
            }
            _ => None,
        }
    }

    fn literal_facet_from_expr(expr: &Expr) -> Option<FacetKind> {
        match expr {
            Expr::StringLiteral(ExprStringLiteral { value, .. }) => {
                Some(FacetKind::Key(value.to_string()))
            }
            Expr::NumberLiteral(ExprNumberLiteral {
                value: Number::Int(idx),
                ..
            }) => idx.as_usize().map(FacetKind::Index),
            _ => None,
        }
    }

    let mut steps = Vec::new();
    let identifier = collect_steps(expr, &mut steps)?;
    if steps.is_empty() {
        return None;
    }
    let mut resolved = Vec::new();
    let mut pending = None;
    for (idx, step) in steps.iter().enumerate() {
        match step {
            FacetStep::Attribute(attr) => {
                if pending.is_some() {
                    return None;
                }
                resolved.push(FacetKind::Attribute(attr.attr.id.clone()));
            }
            FacetStep::Subscript(subscript) => {
                if pending.is_some() {
                    return None;
                }
                if let Some(facet) = literal_facet_from_expr(&subscript.slice) {
                    resolved.push(facet);
                } else if idx == steps.len() - 1 {
                    pending = Some(subscript.slice.clone());
                } else {
                    return None;
                }
            }
        }
    }
    if resolved.is_empty() && pending.is_none() {
        return None;
    }
    Some((
        identifier,
        FacetSubject {
            resolved,
            pending,
            origin: FacetOrigin::Direct,
        },
    ))
}

pub fn identifier_and_chain_for_expr(expr: &Expr) -> Option<(Identifier, FacetChain)> {
    let (identifier, subject) = identifier_and_facet_subject_for_expr(expr)?;
    subject.to_chain().map(|chain| (identifier, chain))
}

fn literal_string_from_expr(expr: &Expr) -> Option<String> {
    if let Expr::StringLiteral(ExprStringLiteral { value, .. }) = expr {
        Some(value.to_string())
    } else {
        None
    }
}

/// Similar to identifier_and_chain_for_expr, except if we encounter a non-concrete subscript in the chain
/// we only return the prefix before that location.
/// For example: w.x[y].z -> w.x
/// Variant of `identifier_and_chain_prefix_for_expr` that allows a custom literal resolver,
/// so builders/solvers can plug in logic that resolves Literal-typed variables.
pub(crate) fn identifier_and_chain_prefix_for_expr_with_resolver<F>(
    expr: &Expr,
    resolver: &mut F,
) -> Option<(Identifier, Vec<FacetKind>)>
where
    F: FnMut(&Expr) -> Option<String>,
{
    fn f<F>(
        expr: &Expr,
        mut rev_chain: Vec<FacetKind>,
        resolver: &mut F,
    ) -> Option<(Identifier, Vec<FacetKind>)>
    where
        F: FnMut(&Expr) -> Option<String>,
    {
        if let Expr::Attribute(attr) = expr {
            match &*attr.value {
                Expr::Name(name) => {
                    rev_chain.push(FacetKind::Attribute(attr.attr.id.clone()));
                    rev_chain.reverse();
                    Some((Ast::expr_name_identifier(name.clone()), rev_chain))
                }
                parent @ (Expr::Attribute(_) | Expr::Subscript(_)) => {
                    rev_chain.push(FacetKind::Attribute(attr.attr.id.clone()));
                    f(parent, rev_chain, resolver)
                }
                _ => None,
            }
        } else if let Expr::Subscript(subscript @ ExprSubscript { slice, .. }) = expr {
            if let Expr::NumberLiteral(ExprNumberLiteral {
                value: Number::Int(idx),
                ..
            }) = &**slice
                && let Some(idx) = idx.as_usize()
            {
                match &*subscript.value {
                    Expr::Name(name) => {
                        rev_chain.push(FacetKind::Index(idx));
                        rev_chain.reverse();
                        Some((Ast::expr_name_identifier(name.clone()), rev_chain))
                    }
                    parent @ (Expr::Attribute(_) | Expr::Subscript(_)) => {
                        rev_chain.push(FacetKind::Index(idx));
                        f(parent, rev_chain, resolver)
                    }
                    _ => None,
                }
            } else if let Some(key) = resolver(slice) {
                match &*subscript.value {
                    Expr::Name(name) => {
                        rev_chain.push(FacetKind::Key(key));
                        rev_chain.reverse();
                        Some((Ast::expr_name_identifier(name.clone()), rev_chain))
                    }
                    parent @ (Expr::Attribute(_) | Expr::Subscript(_)) => {
                        rev_chain.push(FacetKind::Key(key));
                        f(parent, rev_chain, resolver)
                    }
                    _ => None,
                }
            } else {
                // The subscript does not contain an integer or string literal, so we drop everything that we encountered so far
                match &*subscript.value {
                    Expr::Name(name) => Some((Ast::expr_name_identifier(name.clone()), Vec::new())),
                    parent @ (Expr::Attribute(_) | Expr::Subscript(_)) => {
                        rev_chain.clear();
                        f(parent, rev_chain, resolver)
                    }
                    _ => None,
                }
            }
        } else {
            None
        }
    }
    f(expr, Vec::new(), resolver)
}

pub fn identifier_and_chain_prefix_for_expr(expr: &Expr) -> Option<(Identifier, Vec<FacetKind>)> {
    identifier_and_chain_prefix_for_expr_with_resolver(expr, &mut literal_string_from_expr)
}

// Handle narrowing on `dict.get("key")`. During solving, if the resolved
// type of the object is not a subtype of `dict`, we will not perform any narrowing.
fn dict_get_subject_for_call_expr(call_expr: &ExprCall) -> Option<NarrowingSubject> {
    let func = &call_expr.func;
    let arguments = &call_expr.arguments;
    if arguments.keywords.is_empty()
        && arguments.args.len() == 1
        && let Some(first_arg) = arguments.args.first()
        && let Expr::Attribute(attr) = &**func
        && attr.attr.id.as_str() == "get"
        && let Expr::StringLiteral(ExprStringLiteral { value, .. }) = first_arg
    {
        let key = value.to_string();
        if let Some((identifier, facets)) = identifier_and_chain_for_expr(&attr.value) {
            // x.y.z.get("key")
            let mut resolved = facets.facets().to_vec();
            resolved.push(FacetKind::Key(key.clone()));
            return Some(NarrowingSubject::Facets(
                identifier.id,
                FacetSubject {
                    resolved,
                    pending: None,
                    origin: FacetOrigin::GetMethod,
                },
            ));
        } else if let Expr::Name(name) = &*attr.value {
            // x.get("key")
            return Some(NarrowingSubject::Facets(
                name.id.clone(),
                FacetSubject {
                    resolved: vec![FacetKind::Key(key)],
                    pending: None,
                    origin: FacetOrigin::GetMethod,
                },
            ));
        }
    }
    None
}

pub fn expr_to_subjects(expr: &Expr) -> Vec<NarrowingSubject> {
    fn f(expr: &Expr, res: &mut Vec<NarrowingSubject>) {
        match expr {
            Expr::Name(name) => res.push(NarrowingSubject::Name(name.id.clone())),
            Expr::Attribute(_) | Expr::Subscript(_) => {
                if let Some((identifier, subject)) = identifier_and_facet_subject_for_expr(expr) {
                    res.push(NarrowingSubject::Facets(identifier.id, subject));
                }
            }
            Expr::Call(call) => {
                if let Some(subject) = dict_get_subject_for_call_expr(call) {
                    res.push(subject);
                }
            }
            Expr::Named(ExprNamed { target, value, .. }) => {
                f(target, res);
                f(value, res);
            }
            _ => {}
        }
    }
    let mut res = Vec::new();
    f(expr, &mut res);
    res
}
