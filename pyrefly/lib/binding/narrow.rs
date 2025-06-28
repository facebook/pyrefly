/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use pyrefly_python::ast::Ast;
use pyrefly_util::assert_words;
use pyrefly_util::prelude::SliceExt;
use ruff_python_ast::Arguments;
use ruff_python_ast::BoolOp;
use ruff_python_ast::CmpOp;
use ruff_python_ast::Expr;
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
use starlark_map::small_map::SmallMap;
use vec1::Vec1;

use crate::binding::bindings::BindingsBuilder;
use crate::export::special::SpecialExport;
use crate::types::facet::FacetChain;
use crate::types::facet::FacetKind;
use crate::types::types::Type;

assert_words!(AtomicNarrowOp, 10);
assert_words!(NarrowOp, 11);

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
    TypeGuard(Type, Arguments),
    NotTypeGuard(Type, Arguments),
    TypeIs(Type, Arguments),
    NotTypeIs(Type, Arguments),
    In(Expr),
    NotIn(Expr),
    /// Used to narrow tuple types based on length
    LenEq(Expr),
    LenNotEq(Expr),
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
    Atomic(Option<FacetChain>, AtomicNarrowOp),
    And(Vec<NarrowOp>),
    Or(Vec<NarrowOp>),
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
            Self::Eq(v) => Self::NotEq(v.clone()),
            Self::NotEq(v) => Self::Eq(v.clone()),
            Self::In(v) => Self::NotIn(v.clone()),
            Self::NotIn(v) => Self::In(v.clone()),
            Self::LenEq(v) => Self::LenNotEq(v.clone()),
            Self::LenNotEq(v) => Self::LenEq(v.clone()),
            Self::TypeGuard(ty, args) => Self::NotTypeGuard(ty.clone(), args.clone()),
            Self::NotTypeGuard(ty, args) => Self::TypeGuard(ty.clone(), args.clone()),
            Self::TypeIs(ty, args) => Self::NotTypeIs(ty.clone(), args.clone()),
            Self::NotTypeIs(ty, args) => Self::TypeIs(ty.clone(), args.clone()),
            Self::Call(f, args) => Self::NotCall(f.clone(), args.clone()),
            Self::NotCall(f, args) => Self::Call(f.clone(), args.clone()),
            Self::IsTruthy => Self::IsFalsy,
            Self::IsFalsy => Self::IsTruthy,
            Self::Placeholder => Self::Placeholder,
        }
    }
}

#[derive(Clone, Debug)]
pub enum NarrowingSubject {
    Name(Name),
    Facets(Name, FacetChain),
}

impl NarrowingSubject {
    pub fn with_facet(&self, prop: FacetKind) -> Self {
        match self {
            Self::Name(name) => Self::Facets(name.clone(), FacetChain::new(Vec1::new(prop))),
            Self::Facets(name, props) => {
                let props = Vec1::from_vec_push(props.facets().to_vec(), prop);
                Self::Facets(name.clone(), FacetChain::new(props))
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
        for (name, (op, range)) in other.0 {
            self.and(name, op, range);
        }
    }

    fn or(&mut self, name: Name, op: NarrowOp, range: TextRange) {
        let existing_op = self.get_or_placeholder(name, range);
        existing_op.or(op)
    }

    pub fn or_all(&mut self, other: Self) {
        for (name, (op, range)) in other.0 {
            self.or(name, op, range);
        }
    }

    pub fn from_single_narrow_op(left: &Expr, op: AtomicNarrowOp, range: TextRange) -> Self {
        let mut narrow_ops = Self::new();
        for subject in expr_to_subjects(left) {
            let (name, prop) = match subject {
                NarrowingSubject::Name(name) => (name, None),
                NarrowingSubject::Facets(name, prop) => (name, Some(prop)),
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
            NarrowingSubject::Facets(name, prop) => (name, Some(prop)),
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
        match test {
            Some(Expr::Compare(ExprCompare {
                range: _,
                left,
                ops: cmp_ops,
                comparators,
            })) => {
                // If the left expression is a call to len(), we're narrowing the argument
                let mut left = &**left;
                let mut lhs_is_len = false;
                if let Expr::Call(ExprCall {
                    func, arguments, ..
                }) = left
                    && builder.as_special_export(func) == Some(SpecialExport::Len)
                    && arguments.args.len() == 1
                    && arguments.keywords.is_empty()
                {
                    lhs_is_len = true;
                    left = arguments.args.first().unwrap();
                };
                let mut ops = cmp_ops
                    .iter()
                    .zip(comparators)
                    .filter_map(|(cmp_op, right)| {
                        let range = right.range();
                        let op = match cmp_op {
                            CmpOp::Is if !lhs_is_len => AtomicNarrowOp::Is(right.clone()),
                            CmpOp::IsNot if !lhs_is_len => AtomicNarrowOp::IsNot(right.clone()),
                            CmpOp::Eq if lhs_is_len => AtomicNarrowOp::LenEq(right.clone()),
                            CmpOp::NotEq if lhs_is_len => AtomicNarrowOp::LenNotEq(right.clone()),
                            CmpOp::Eq => AtomicNarrowOp::Eq(right.clone()),
                            CmpOp::NotEq => AtomicNarrowOp::NotEq(right.clone()),
                            CmpOp::In if !lhs_is_len => AtomicNarrowOp::In(right.clone()),
                            CmpOp::NotIn if !lhs_is_len => AtomicNarrowOp::NotIn(right.clone()),
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
            Some(Expr::BoolOp(ExprBoolOp {
                range: _,
                op,
                values,
            })) => {
                let extend = match op {
                    BoolOp::And => NarrowOps::and_all,
                    BoolOp::Or => NarrowOps::or_all,
                };
                let mut exprs = values.iter();
                let mut narrow_ops = Self::from_expr(builder, exprs.next());
                for next_val in exprs {
                    extend(&mut narrow_ops, Self::from_expr(builder, Some(next_val)))
                }
                narrow_ops
            }
            Some(Expr::UnaryOp(ExprUnaryOp {
                range: _,
                op: UnaryOp::Not,
                operand: e,
            })) => Self::from_expr(builder, Some(e)).negate(),
            Some(Expr::Call(ExprCall {
                range,
                func,
                arguments:
                    args @ Arguments {
                        range: _,
                        args: posargs,
                        keywords: _,
                    },
            })) if !posargs.is_empty() => {
                // This may be a function call that narrows the type of its first argument. Record
                // it as a possible narrowing operation that we'll resolve in the answers phase.
                Self::from_single_narrow_op(
                    &posargs[0],
                    AtomicNarrowOp::Call(Box::new((**func).clone()), args.clone()),
                    *range,
                )
            }
            Some(e) => Self::from_single_narrow_op(e, AtomicNarrowOp::IsTruthy, e.range()),
            None => Self::new(),
        }
    }
}

/// Given an expression, determine whether it is a chain of properties (attribute/concrete index) rooted at a name,
/// and if so, return the name and the chain of properties.
/// For example: x.y.[0].z
pub fn identifier_and_chain_for_expr(expr: &Expr) -> Option<(Identifier, FacetChain)> {
    fn f(expr: &Expr, mut rev_chain: Vec<FacetKind>) -> Option<(Identifier, FacetChain)> {
        if let Expr::Attribute(attr) = expr {
            match &*attr.value {
                Expr::Name(name) => {
                    let mut final_chain =
                        Vec1::from_vec_push(rev_chain, FacetKind::Attribute(attr.attr.id.clone()));
                    final_chain.reverse();
                    Some((
                        Ast::expr_name_identifier(name.clone()),
                        FacetChain::new(final_chain),
                    ))
                }
                parent @ (Expr::Attribute(_) | Expr::Subscript(_)) => {
                    rev_chain.push(FacetKind::Attribute(attr.attr.id.clone()));
                    f(parent, rev_chain)
                }
                _ => None,
            }
        } else if let Expr::Subscript(subscript @ ExprSubscript { slice, .. }) = expr
            && let Expr::NumberLiteral(ExprNumberLiteral {
                value: Number::Int(idx),
                ..
            }) = &**slice
            && let Some(idx) = idx.as_usize()
        {
            match &*subscript.value {
                Expr::Name(name) => {
                    let mut final_chain = Vec1::from_vec_push(rev_chain, FacetKind::Index(idx));
                    final_chain.reverse();
                    Some((
                        Ast::expr_name_identifier(name.clone()),
                        FacetChain::new(final_chain),
                    ))
                }
                parent @ (Expr::Attribute(_) | Expr::Subscript(_)) => {
                    rev_chain.push(FacetKind::Index(idx));
                    f(parent, rev_chain)
                }
                _ => None,
            }
        } else if let Expr::Subscript(subscript @ ExprSubscript { slice, .. }) = expr
            && let Expr::StringLiteral(ExprStringLiteral { value: key, .. }) = &**slice
        {
            match &*subscript.value {
                Expr::Name(name) => {
                    let mut final_chain =
                        Vec1::from_vec_push(rev_chain, FacetKind::Key(key.to_string()));
                    final_chain.reverse();
                    Some((
                        Ast::expr_name_identifier(name.clone()),
                        FacetChain::new(final_chain),
                    ))
                }
                parent @ (Expr::Attribute(_) | Expr::Subscript(_)) => {
                    rev_chain.push(FacetKind::Key(key.to_string()));
                    f(parent, rev_chain)
                }
                _ => None,
            }
        } else {
            None
        }
    }
    f(expr, Vec::new())
}

/// Similar to identifier_and_chain_for_expr, except if we encounter a non-concrete subscript in the chain
/// we only return the prefix before that location.
/// For example: w.x[y].z -> w.x
pub fn identifier_and_chain_prefix_for_expr(expr: &Expr) -> Option<(Identifier, Vec<FacetKind>)> {
    fn f(expr: &Expr, mut rev_chain: Vec<FacetKind>) -> Option<(Identifier, Vec<FacetKind>)> {
        if let Expr::Attribute(attr) = expr {
            match &*attr.value {
                Expr::Name(name) => {
                    rev_chain.push(FacetKind::Attribute(attr.attr.id.clone()));
                    rev_chain.reverse();
                    Some((Ast::expr_name_identifier(name.clone()), rev_chain))
                }
                parent @ (Expr::Attribute(_) | Expr::Subscript(_)) => {
                    rev_chain.push(FacetKind::Attribute(attr.attr.id.clone()));
                    f(parent, rev_chain)
                }
                _ => None,
            }
        } else if let Expr::Subscript(subscript @ ExprSubscript { slice, .. }) = expr
            && let Expr::NumberLiteral(ExprNumberLiteral {
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
                    f(parent, rev_chain)
                }
                _ => None,
            }
        } else if let Expr::Subscript(subscript @ ExprSubscript { slice, .. }) = expr
            && let Expr::StringLiteral(ExprStringLiteral { value: key, .. }) = &**slice
        {
            match &*subscript.value {
                Expr::Name(name) => {
                    rev_chain.push(FacetKind::Key(key.to_string()));
                    rev_chain.reverse();
                    Some((Ast::expr_name_identifier(name.clone()), rev_chain))
                }
                parent @ (Expr::Attribute(_) | Expr::Subscript(_)) => {
                    rev_chain.push(FacetKind::Key(key.to_string()));
                    f(parent, rev_chain)
                }
                _ => None,
            }
        } else if let Expr::Subscript(subscript) = expr {
            // The subscript does not contain an integer or string literal, so we drop everything that we encountered so far
            match &*subscript.value {
                Expr::Name(name) => Some((Ast::expr_name_identifier(name.clone()), Vec::new())),
                parent @ (Expr::Attribute(_) | Expr::Subscript(_)) => {
                    rev_chain.clear();
                    f(parent, rev_chain)
                }
                _ => None,
            }
        } else {
            None
        }
    }
    f(expr, Vec::new())
}

fn subject_for_expr(expr: &Expr) -> Option<NarrowingSubject> {
    identifier_and_chain_for_expr(expr)
        .map(|(identifier, attr)| NarrowingSubject::Facets(identifier.id, attr))
}

pub fn expr_to_subjects(expr: &Expr) -> Vec<NarrowingSubject> {
    fn f(expr: &Expr, res: &mut Vec<NarrowingSubject>) {
        match expr {
            Expr::Name(name) => res.push(NarrowingSubject::Name(name.id.clone())),
            Expr::Attribute(_) | Expr::Subscript(_) => res.extend(subject_for_expr(expr)),
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
