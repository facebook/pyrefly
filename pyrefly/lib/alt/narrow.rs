/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use num_traits::ToPrimitive;
use pyrefly_util::prelude::SliceExt;
use ruff_python_ast::Arguments;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprNumberLiteral;
use ruff_python_ast::ExprStringLiteral;
use ruff_python_ast::Int;
use ruff_python_ast::Number;
use ruff_python_ast::StringLiteral;
use ruff_python_ast::StringLiteralFlags;
use ruff_python_ast::StringLiteralValue;
use ruff_python_ast::name::Name;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;

use crate::alt::answers::LookupAnswer;
use crate::alt::answers_solver::AnswersSolver;
use crate::alt::attr::Narrowable;
use crate::alt::callable::CallArg;
use crate::alt::callable::CallKeyword;
use crate::binding::narrow::AtomicNarrowOp;
use crate::binding::narrow::NarrowOp;
use crate::error::collector::ErrorCollector;
use crate::types::callable::FunctionKind;
use crate::types::class::ClassType;
use crate::types::facet::FacetChain;
use crate::types::facet::FacetKind;
use crate::types::literal::Lit;
use crate::types::tuple::Tuple;
use crate::types::type_info::TypeInfo;
use crate::types::types::CalleeKind;
use crate::types::types::Type;

/// Beyond this size, don't try and narrow an enum.
///
/// If we have over 100 fields, the odds of the negative-type being useful is vanishingly small.
/// But the cost to create such a type (and then probably knock individual elements out of it)
/// is very high.
const NARROW_ENUM_LIMIT: usize = 100;

impl<'a, Ans: LookupAnswer> AnswersSolver<'a, Ans> {
    // Get the union of all members of an enum, minus the specified member
    fn subtract_enum_member(&self, cls: &ClassType, name: &Name) -> Type {
        if cls.class_object().fields().len() > NARROW_ENUM_LIMIT {
            return Type::ClassType(cls.clone());
        }
        let e = self.get_enum_from_class_type(cls).unwrap();
        // Enums derived from enum.Flag cannot be treated as a union of their members
        if e.is_flag {
            return Type::ClassType(cls.clone());
        }
        self.unions(
            self.get_enum_members(cls.class_object())
                .into_iter()
                .filter_map(|f| {
                    if let Lit::Enum(lit_enum) = &f
                        && &lit_enum.member == name
                    {
                        None
                    } else {
                        Some(Type::Literal(f))
                    }
                })
                .collect::<Vec<_>>(),
        )
    }

    /// Get our best approximation of ty & right.
    ///
    /// If the intersection is empty - which does not necessarily indicate
    /// an actual empty set because of multiple inheritance - use `fallback`
    fn intersect_with_fallback(
        &self,
        left: &Type,
        right: &Type,
        fallback: impl Fn() -> Type,
    ) -> Type {
        self.distribute_over_union(left, |l| {
            self.distribute_over_union(right, |r| {
                if self.is_subset_eq(r, l) {
                    r.clone()
                } else if self.is_subset_eq(l, r) {
                    l.clone()
                } else {
                    fallback()
                }
            })
        })
    }

    fn intersect(&self, left: &Type, right: &Type) -> Type {
        self.intersect_with_fallback(left, right, Type::never)
    }

    fn intersects(&self, ts: &[Type]) -> Type {
        match ts {
            [] => Type::ClassType(self.stdlib.object().clone()),
            [ty] => ty.clone(),
            [ty0, ty1] => self.intersect(ty0, ty1),
            [ty0, ts @ ..] => self.intersect(ty0, &self.intersects(ts)),
        }
    }

    fn subtract(&self, left: &Type, right: &Type) -> Type {
        self.distribute_over_union(left, |left| {
            // Special is_any check because `Any <: int` as a special case, but would mess up this.
            if !left.is_any() && self.is_subset_eq(left, right) {
                Type::never()
            } else {
                left.clone()
            }
        })
    }

    fn resolve_narrowing_call(
        &self,
        func: &Expr,
        args: &Arguments,
        errors: &ErrorCollector,
    ) -> Option<AtomicNarrowOp> {
        let func_ty = self.expr_infer(func, errors);
        if args.args.len() > 1 {
            let second_arg = &args.args[1];
            let op = match func_ty.callee_kind() {
                Some(CalleeKind::Function(FunctionKind::IsInstance)) => {
                    Some(AtomicNarrowOp::IsInstance(second_arg.clone()))
                }
                Some(CalleeKind::Function(FunctionKind::IsSubclass)) => {
                    Some(AtomicNarrowOp::IsSubclass(second_arg.clone()))
                }
                _ => None,
            };
            if op.is_some() {
                return op;
            }
        }
        if func_ty.is_typeis() {
            Some(AtomicNarrowOp::TypeIs(func_ty.clone(), args.clone()))
        } else if func_ty.is_typeguard() {
            Some(AtomicNarrowOp::TypeGuard(func_ty.clone(), args.clone()))
        } else {
            None
        }
    }

    fn narrow_isinstance(&self, left: &Type, right: &Type) -> Type {
        if let Some(ts) = right.as_decomposed_tuple_or_union(self.stdlib) {
            self.unions(ts.iter().map(|t| self.narrow_isinstance(left, t)).collect())
        } else if let Some(right) = self.unwrap_class_object_silently(right) {
            self.intersect_with_fallback(left, &right, || right.clone())
        } else {
            left.clone()
        }
    }

    fn narrow_is_not_instance(&self, left: &Type, right: &Type) -> Type {
        if let Some(ts) = right.as_decomposed_tuple_or_union(self.stdlib) {
            self.intersects(&ts.map(|t| self.narrow_is_not_instance(left, t)))
        } else if let Some(right) = self.unwrap_class_object_silently(right) {
            self.subtract(left, &right)
        } else {
            left.clone()
        }
    }

    fn narrow_issubclass(&self, left: &Type, right: &Type, range: TextRange) -> Type {
        if let Some(ts) = right.as_decomposed_tuple_or_union(self.stdlib) {
            self.unions(
                ts.iter()
                    .map(|t| self.narrow_issubclass(left, t, range))
                    .collect(),
            )
        } else if let Some(left) = self.untype_opt(left.clone(), range)
            && let Some(right) = self.unwrap_class_object_silently(right)
        {
            Type::type_form(self.intersect(&left, &right))
        } else {
            left.clone()
        }
    }

    fn narrow_is_not_subclass(&self, left: &Type, right: &Type, range: TextRange) -> Type {
        if let Some(ts) = right.as_decomposed_tuple_or_union(self.stdlib) {
            self.intersects(&ts.map(|t| self.narrow_is_not_subclass(left, t, range)))
        } else if let Some(left) = self.untype_opt(left.clone(), range)
            && let Some(right) = self.unwrap_class_object_silently(right)
        {
            Type::type_form(self.subtract(&left, &right))
        } else {
            left.clone()
        }
    }

    pub fn atomic_narrow(
        &self,
        ty: &Type,
        op: &AtomicNarrowOp,
        range: TextRange,
        errors: &ErrorCollector,
    ) -> Type {
        match op {
            AtomicNarrowOp::Placeholder => ty.clone(),
            AtomicNarrowOp::LenEq(v) => {
                let right = self.expr_infer(v, errors);
                let Type::Literal(Lit::Int(lit)) = &right else {
                    return ty.clone();
                };
                let Some(len) = lit.as_i64().and_then(|i| i.to_usize()) else {
                    return ty.clone();
                };
                self.distribute_over_union(ty, |ty| match ty {
                    Type::Tuple(Tuple::Concrete(elts)) if elts.len() != len => Type::never(),
                    Type::Tuple(Tuple::Unpacked(box (prefix, _, suffix)))
                        if prefix.len() + suffix.len() > len =>
                    {
                        Type::never()
                    }
                    Type::Tuple(Tuple::Unpacked(box (prefix, _, suffix)))
                        if prefix.len() + suffix.len() == len =>
                    {
                        Type::tuple(prefix.iter().cloned().chain(suffix.clone()).collect())
                    }
                    Type::Tuple(Tuple::Unpacked(box (
                        prefix,
                        Type::Tuple(Tuple::Unbounded(middle)),
                        suffix,
                    ))) if prefix.len() + suffix.len() < len => {
                        let middle_elements =
                            vec![(**middle).clone(); len - prefix.len() - suffix.len()];
                        Type::tuple(
                            prefix
                                .iter()
                                .cloned()
                                .chain(middle_elements)
                                .chain(suffix.clone())
                                .collect(),
                        )
                    }
                    Type::Tuple(Tuple::Unbounded(elements)) => {
                        Type::tuple(vec![(**elements).clone(); len])
                    }
                    Type::ClassType(class)
                        if let Some(elements) = self.named_tuple_element_types(class)
                            && elements.len() != len =>
                    {
                        Type::never()
                    }
                    _ => ty.clone(),
                })
            }
            AtomicNarrowOp::LenNotEq(v) => {
                let right = self.expr_infer(v, errors);
                let Type::Literal(Lit::Int(lit)) = &right else {
                    return ty.clone();
                };
                let Some(len) = lit.as_i64().and_then(|i| i.to_usize()) else {
                    return ty.clone();
                };
                self.distribute_over_union(ty, |ty| match ty {
                    Type::Tuple(Tuple::Concrete(elts)) if elts.len() == len => Type::never(),
                    Type::ClassType(class)
                        if let Some(elements) = self.named_tuple_element_types(class)
                            && elements.len() == len =>
                    {
                        Type::never()
                    }
                    _ => ty.clone(),
                })
            }
            AtomicNarrowOp::In(v) => {
                let exprs = match v {
                    Expr::List(list) => Some(list.elts.clone()),
                    Expr::Tuple(tuple) => Some(tuple.elts.clone()),
                    Expr::Set(set) => Some(set.elts.clone()),
                    _ => None,
                };
                let Some(exprs) = exprs else {
                    return ty.clone();
                };
                let mut literal_types = Vec::new();
                for expr in exprs {
                    let expr_ty = self.expr_infer(&expr, errors);
                    if matches!(expr_ty, Type::Literal(_) | Type::None) {
                        literal_types.push(expr_ty);
                    } else {
                        return ty.clone();
                    }
                }
                self.intersect(ty, &self.unions(literal_types))
            }
            AtomicNarrowOp::NotIn(v) => {
                let exprs = match v {
                    Expr::List(list) => Some(list.elts.clone()),
                    Expr::Tuple(tuple) => Some(tuple.elts.clone()),
                    Expr::Set(set) => Some(set.elts.clone()),
                    _ => None,
                };
                let Some(exprs) = exprs else {
                    return ty.clone();
                };
                let mut literal_types = Vec::new();
                for expr in exprs {
                    let expr_ty = self.expr_infer(&expr, errors);
                    if matches!(expr_ty, Type::Literal(_) | Type::None) {
                        literal_types.push(expr_ty);
                    } else {
                        return ty.clone();
                    }
                }
                self.distribute_over_union(ty, |t| {
                    let mut result = t.clone();
                    for right in &literal_types {
                        match (t, right) {
                            (_, _) if t == right => {
                                result = Type::never();
                            }
                            (Type::ClassType(cls), Type::Literal(Lit::Bool(b)))
                                if cls.is_builtin("bool") =>
                            {
                                result = Type::Literal(Lit::Bool(!b));
                            }
                            (Type::ClassType(left_cls), Type::Literal(Lit::Enum(right)))
                                if left_cls == &right.class =>
                            {
                                result = self.subtract_enum_member(left_cls, &right.member);
                            }
                            _ => {}
                        }
                    }
                    result
                })
            }
            AtomicNarrowOp::Is(v) => {
                let right = self.expr_infer(v, errors);
                // Get our best approximation of ty & right.
                self.intersect(ty, &right)
            }
            AtomicNarrowOp::IsNot(v) => {
                let right = self.expr_infer(v, errors);
                // Get our best approximation of ty - right.
                self.distribute_over_union(ty, |t| {
                    // Only certain literal types can be compared by identity.
                    match (t, &right) {
                        (
                            _,
                            Type::None | Type::Literal(Lit::Bool(_)) | Type::Literal(Lit::Enum(_)),
                        ) if *t == right => Type::never(),
                        (Type::ClassType(cls), Type::Literal(Lit::Bool(b)))
                            if cls.is_builtin("bool") =>
                        {
                            Type::Literal(Lit::Bool(!b))
                        }
                        (Type::ClassType(left_cls), Type::Literal(Lit::Enum(right)))
                            if left_cls == &right.class =>
                        {
                            self.subtract_enum_member(left_cls, &right.member)
                        }
                        _ => t.clone(),
                    }
                })
            }
            AtomicNarrowOp::IsInstance(v) => {
                let right = self.expr_infer(v, errors);
                self.narrow_isinstance(ty, &right)
            }
            AtomicNarrowOp::IsNotInstance(v) => {
                let right = self.expr_infer(v, errors);
                self.narrow_is_not_instance(ty, &right)
            }
            AtomicNarrowOp::IsSubclass(v) => {
                let right = self.expr_infer(v, errors);
                self.narrow_issubclass(ty, &right, v.range())
            }
            AtomicNarrowOp::IsNotSubclass(v) => {
                let right = self.expr_infer(v, errors);
                self.narrow_is_not_subclass(ty, &right, v.range())
            }
            AtomicNarrowOp::TypeGuard(t, arguments) => {
                if let Some(call_target) = self.as_call_target(t.clone()) {
                    let args = arguments.args.map(CallArg::expr_maybe_starred);
                    let kws = arguments.keywords.map(CallKeyword::new);
                    let ret = self.call_infer(call_target, &args, &kws, range, errors, None, None);
                    if let Type::TypeGuard(t) = ret {
                        return *t;
                    }
                }
                ty.clone()
            }
            AtomicNarrowOp::NotTypeGuard(_, _) => ty.clone(),
            AtomicNarrowOp::TypeIs(t, arguments) => {
                if let Some(call_target) = self.as_call_target(t.clone()) {
                    let args = arguments.args.map(CallArg::expr_maybe_starred);
                    let kws = arguments.keywords.map(CallKeyword::new);
                    let ret = self.call_infer(call_target, &args, &kws, range, errors, None, None);
                    if let Type::TypeIs(t) = ret {
                        return self.intersect(ty, &t);
                    }
                }
                ty.clone()
            }
            AtomicNarrowOp::NotTypeIs(t, arguments) => {
                if let Some(call_target) = self.as_call_target(t.clone()) {
                    let args = arguments.args.map(CallArg::expr_maybe_starred);
                    let kws = arguments.keywords.map(CallKeyword::new);
                    let ret = self.call_infer(call_target, &args, &kws, range, errors, None, None);
                    if let Type::TypeIs(t) = ret {
                        return self.subtract(ty, &t);
                    }
                }
                ty.clone()
            }
            AtomicNarrowOp::IsTruthy | AtomicNarrowOp::IsFalsy => {
                self.distribute_over_union(ty, |t| {
                    let boolval = matches!(op, AtomicNarrowOp::IsTruthy);
                    if t.as_bool() == Some(!boolval) {
                        Type::never()
                    } else if matches!(t, Type::ClassType(cls) if cls.is_builtin("bool")) {
                        Type::Literal(Lit::Bool(boolval))
                    } else {
                        t.clone()
                    }
                })
            }
            AtomicNarrowOp::Eq(v) => {
                let right = self.expr_infer(v, errors);
                if matches!(right, Type::Literal(_) | Type::None) {
                    self.intersect(ty, &right)
                } else {
                    ty.clone()
                }
            }
            AtomicNarrowOp::NotEq(v) => {
                let right = self.expr_infer(v, errors);
                if matches!(right, Type::Literal(_) | Type::None) {
                    self.distribute_over_union(ty, |t| match (t, &right) {
                        (_, _) if *t == right => Type::never(),
                        (Type::ClassType(cls), Type::Literal(Lit::Bool(b)))
                            if cls.is_builtin("bool") =>
                        {
                            Type::Literal(Lit::Bool(!b))
                        }
                        (Type::ClassType(left_cls), Type::Literal(Lit::Enum(right)))
                            if left_cls == &right.class =>
                        {
                            self.subtract_enum_member(left_cls, &right.member)
                        }
                        _ => t.clone(),
                    })
                } else {
                    ty.clone()
                }
            }
            AtomicNarrowOp::Call(func, args) | AtomicNarrowOp::NotCall(func, args) => {
                if let Some(resolved_op) = self.resolve_narrowing_call(func, args, errors) {
                    if matches!(op, AtomicNarrowOp::Call(..)) {
                        self.atomic_narrow(ty, &resolved_op, range, errors)
                    } else {
                        self.atomic_narrow(ty, &resolved_op.negate(), range, errors)
                    }
                } else {
                    ty.clone()
                }
            }
        }
    }

    fn get_facet_chain_type(
        &self,
        base: &TypeInfo,
        facet_chain: &FacetChain,
        range: TextRange,
    ) -> Type {
        // We don't want to throw any attribute access or indexing errors when narrowing - the same code is traversed
        // separately for type checking, and there might be error context then we don't have here.
        let ignore_errors = self.error_swallower();
        let (first_facet, remaining_facets) = facet_chain.facets().clone().split_off_first();
        match self.narrowable_for_facet_chain(
            base,
            &first_facet,
            &remaining_facets,
            range,
            &ignore_errors,
        ) {
            Narrowable::Simple(ty) => ty,
            Narrowable::PropertyOrDescriptor(ty) => {
                // TODO(stroxler): Implement plumbing to warn on downstream reads.
                ty
            }
            Narrowable::UnionPropertyOrDescriptor(ty) => {
                // TODO(stroxler): Implement plumbing to warn on downstream reads.
                ty
            }
        }
    }

    fn narrowable_for_facet_chain(
        &self,
        base: &TypeInfo,
        first_facet: &FacetKind,
        remaining_facets: &[FacetKind],
        range: TextRange,
        errors: &ErrorCollector,
    ) -> Narrowable {
        match first_facet {
            FacetKind::Attribute(first_attr_name) => match remaining_facets.split_first() {
                None => match base.type_at_facet(first_facet) {
                    Some(ty) => Narrowable::Simple(ty.clone()),
                    None => self.narrowable_for_attr(base.ty(), first_attr_name, range, errors),
                },
                Some((next_name, remaining_facets)) => {
                    let base = self.attr_infer(base, first_attr_name, range, errors, None);
                    self.narrowable_for_facet_chain(
                        &base,
                        next_name,
                        remaining_facets,
                        range,
                        errors,
                    )
                }
            },
            FacetKind::Index(idx) => {
                // We synthesize a slice expression for the subscript here
                // The range doesn't matter, since narrowing logic swallows type errors
                let synthesized_slice = Expr::NumberLiteral(ExprNumberLiteral {
                    range,
                    value: Number::Int(Int::from(*idx as u64)),
                });
                match remaining_facets.split_first() {
                    None => match base.type_at_facet(first_facet) {
                        Some(ty) => Narrowable::Simple(ty.clone()),
                        None => Narrowable::Simple(self.subscript_infer_for_type(
                            base.ty(),
                            &synthesized_slice,
                            range,
                            errors,
                        )),
                    },
                    Some((next_name, remaining_facets)) => {
                        let base_ty = self.subscript_infer(base, &synthesized_slice, range, errors);
                        self.narrowable_for_facet_chain(
                            &base_ty,
                            next_name,
                            remaining_facets,
                            range,
                            errors,
                        )
                    }
                }
            }
            FacetKind::Key(key) => {
                // We synthesize a slice expression for the subscript here
                // The range doesn't matter, since narrowing logic swallows type errors
                let synthesized_slice = Expr::StringLiteral(ExprStringLiteral {
                    range,
                    value: StringLiteralValue::single(StringLiteral {
                        range,
                        value: key.clone().into_boxed_str(),
                        flags: StringLiteralFlags::empty(),
                    }),
                });
                match remaining_facets.split_first() {
                    None => match base.type_at_facet(first_facet) {
                        Some(ty) => Narrowable::Simple(ty.clone()),
                        None => Narrowable::Simple(self.subscript_infer_for_type(
                            base.ty(),
                            &synthesized_slice,
                            range,
                            errors,
                        )),
                    },
                    Some((next_name, remaining_facets)) => {
                        let base_ty = self.subscript_infer(base, &synthesized_slice, range, errors);
                        self.narrowable_for_facet_chain(
                            &base_ty,
                            next_name,
                            remaining_facets,
                            range,
                            errors,
                        )
                    }
                }
            }
        }
    }

    pub fn narrow(
        &self,
        type_info: &TypeInfo,
        op: &NarrowOp,
        range: TextRange,
        errors: &ErrorCollector,
    ) -> TypeInfo {
        match op {
            NarrowOp::Atomic(None, op) => {
                type_info
                    .clone()
                    .with_ty(self.atomic_narrow(type_info.ty(), op, range, errors))
            }
            NarrowOp::Atomic(Some(facet_chain), op) => {
                let ty = self.atomic_narrow(
                    &self.get_facet_chain_type(type_info, facet_chain, range),
                    op,
                    range,
                    errors,
                );
                type_info.with_narrow(facet_chain.facets(), ty)
            }
            NarrowOp::And(ops) => {
                let mut ops_iter = ops.iter();
                if let Some(first_op) = ops_iter.next() {
                    let mut ret = self.narrow(type_info, first_op, range, errors);
                    for next_op in ops_iter {
                        ret = self.narrow(&ret, next_op, range, errors);
                    }
                    ret
                } else {
                    type_info.clone()
                }
            }
            NarrowOp::Or(ops) => TypeInfo::join(
                ops.map(|op| self.narrow(type_info, op, range, errors)),
                &|tys| self.unions(tys),
            ),
        }
    }
}
