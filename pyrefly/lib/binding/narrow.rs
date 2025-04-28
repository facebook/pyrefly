/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use ruff_python_ast::Arguments;
use ruff_python_ast::BoolOp;
use ruff_python_ast::CmpOp;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprAttribute;
use ruff_python_ast::ExprBoolOp;
use ruff_python_ast::ExprCall;
use ruff_python_ast::ExprCompare;
use ruff_python_ast::ExprNamed;
use ruff_python_ast::ExprUnaryOp;
use ruff_python_ast::Identifier;
use ruff_python_ast::UnaryOp;
use ruff_python_ast::name::Name;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use starlark_map::small_map::SmallMap;
use starlark_map::smallmap;
use vec1::Vec1;

use crate::assert_words;
use crate::ruff::ast::Ast;
use crate::types::types::Type;
use crate::util::prelude::SliceExt;

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
pub struct AttributeChain(pub Box<Vec1<Name>>);

impl AttributeChain {
    pub fn new(chain: Vec1<Name>) -> Self {
        Self(Box::new(chain))
    }

    pub fn names(&self) -> &Vec1<Name> {
        match self {
            Self(box chain) => chain,
        }
    }
}

#[derive(Clone, Debug)]
pub enum NarrowOp {
    Atomic(Option<AttributeChain>, AtomicNarrowOp),
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
enum NarrowingSubject {
    Name(Name),
    Attribute(Name, AttributeChain),
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

    pub fn is(name: &Name, v: Expr, range: TextRange) -> Self {
        Self(smallmap! { name.clone() => (NarrowOp::Atomic(None, AtomicNarrowOp::Is(v)), range) })
    }

    pub fn eq(name: &Name, v: Expr, range: TextRange) -> Self {
        Self(smallmap! { name.clone() => (NarrowOp::Atomic(None, AtomicNarrowOp::Eq(v)), range) })
    }

    pub fn isinstance(name: &Name, v: Expr, range: TextRange) -> Self {
        Self(
            smallmap! { name.clone() => (NarrowOp::Atomic(None, AtomicNarrowOp::IsInstance(v)), range) },
        )
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
            let (name, attr) = match subject {
                NarrowingSubject::Name(name) => (name, None),
                NarrowingSubject::Attribute(name, attr) => (name, Some(attr)),
            };
            if let Some((existing, _)) = narrow_ops.0.get_mut(&name) {
                existing.and(NarrowOp::Atomic(attr, op.clone()));
            } else {
                narrow_ops
                    .0
                    .insert(name, (NarrowOp::Atomic(attr, op.clone()), range));
            }
        }
        narrow_ops
    }

    pub fn from_expr(test: Option<&Expr>) -> Self {
        match test {
            Some(Expr::Compare(ExprCompare {
                range: _,
                left,
                ops: cmp_ops,
                comparators,
            })) => {
                let mut ops = cmp_ops
                    .iter()
                    .zip(comparators)
                    .filter_map(|(cmp_op, right)| {
                        let range = right.range();
                        let op = match cmp_op {
                            CmpOp::Is => AtomicNarrowOp::Is(right.clone()),
                            CmpOp::IsNot => AtomicNarrowOp::IsNot(right.clone()),
                            CmpOp::Eq => AtomicNarrowOp::Eq(right.clone()),
                            CmpOp::NotEq => AtomicNarrowOp::NotEq(right.clone()),
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
                let mut narrow_ops = Self::from_expr(exprs.next());
                for next_val in exprs {
                    extend(&mut narrow_ops, Self::from_expr(Some(next_val)))
                }
                narrow_ops
            }
            Some(Expr::UnaryOp(ExprUnaryOp {
                range: _,
                op: UnaryOp::Not,
                operand: box e,
            })) => Self::from_expr(Some(e)).negate(),
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

pub fn identifier_and_chain_for_attribute(
    expr: &ExprAttribute,
) -> Option<(Identifier, AttributeChain)> {
    fn f(
        expr: &ExprAttribute,
        mut rev_attr_chain: Vec<Name>,
    ) -> Option<(Identifier, AttributeChain)> {
        match &*expr.value {
            Expr::Name(name) => {
                let mut final_chain = Vec1::from_vec_push(rev_attr_chain, expr.attr.id.clone());
                final_chain.reverse();
                Some((
                    Ast::expr_name_identifier(name.clone()),
                    AttributeChain::new(final_chain),
                ))
            }
            Expr::Attribute(x) => {
                rev_attr_chain.push(expr.attr.id.clone());
                f(x, rev_attr_chain)
            }
            _ => None,
        }
    }
    f(expr, Vec::new())
}

fn subject_for_attribute(expr: &ExprAttribute) -> Option<NarrowingSubject> {
    identifier_and_chain_for_attribute(expr)
        .map(|(identifier, attr)| NarrowingSubject::Attribute(identifier.id, attr))
}

fn expr_to_subjects(expr: &Expr) -> Vec<NarrowingSubject> {
    fn f(expr: &Expr, res: &mut Vec<NarrowingSubject>) {
        match expr {
            Expr::Name(name) => res.push(NarrowingSubject::Name(name.id.clone())),
            Expr::Attribute(x) => res.extend(subject_for_attribute(x)),
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
