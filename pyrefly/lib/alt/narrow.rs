/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use ruff_python_ast::Arguments;
use ruff_python_ast::Expr;
use ruff_python_ast::name::Name;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;

use crate::alt::answers::AnswersSolver;
use crate::alt::answers::LookupAnswer;
use crate::alt::attr::Narrowable;
use crate::alt::callable::CallArg;
use crate::binding::narrow::AtomicNarrowOp;
use crate::binding::narrow::AttributeChain;
use crate::binding::narrow::NarrowOp;
use crate::error::collector::ErrorCollector;
use crate::types::callable::FunctionKind;
use crate::types::class::ClassType;
use crate::types::literal::Lit;
use crate::types::type_info::TypeInfo;
use crate::types::types::CalleeKind;
use crate::types::types::Type;
use crate::util::prelude::SliceExt;

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
                    if let Lit::Enum(box (_, member_name, _)) = &f
                        && *member_name == *name
                    {
                        None
                    } else {
                        Some(Type::Literal(f))
                    }
                })
                .collect::<Vec<_>>(),
        )
    }

    fn intersect(&self, left: &Type, right: &Type) -> Type {
        // Get our best approximation of ty & right.
        self.distribute_over_union(left, |t| {
            if self.is_subset_eq(right, t) {
                right.clone()
            } else if self.is_subset_eq(t, right) {
                t.clone()
            } else {
                Type::never()
            }
        })
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
        if let Some(ts) = right.as_decomposed_tuple_or_union() {
            self.unions(ts.iter().map(|t| self.narrow_isinstance(left, t)).collect())
        } else if let Some(right) = self.unwrap_class_object_silently(right) {
            self.intersect(left, &right)
        } else {
            left.clone()
        }
    }

    fn narrow_is_not_instance(&self, left: &Type, right: &Type) -> Type {
        if let Some(ts) = right.as_decomposed_tuple_or_union() {
            self.intersects(&ts.map(|t| self.narrow_is_not_instance(left, t)))
        } else if let Some(right) = self.unwrap_class_object_silently(right) {
            self.subtract(left, &right)
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
                        (
                            Type::ClassType(left_cls),
                            Type::Literal(Lit::Enum(box (right_cls, name, _))),
                        ) if *left_cls == *right_cls => self.subtract_enum_member(left_cls, name),
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
                if let Some(left) = self.untype_opt(ty.clone(), v.range())
                    && let Some(right) = self.unwrap_class_object_silently(&right)
                {
                    Type::type_form(self.intersect(&left, &right))
                } else {
                    ty.clone()
                }
            }
            AtomicNarrowOp::IsNotSubclass(v) => {
                let right = self.expr_infer(v, errors);
                if let Some(left) = self.untype_opt(ty.clone(), v.range())
                    && let Some(right) = self.unwrap_class_object_silently(&right)
                {
                    Type::type_form(self.subtract(&left, &right))
                } else {
                    ty.clone()
                }
            }
            AtomicNarrowOp::TypeGuard(t, arguments) => {
                if let Some(call_target) = self.as_call_target(t.clone()) {
                    let args = arguments.args.map(|arg| match arg {
                        Expr::Starred(x) => CallArg::Star(&x.value, x.range),
                        _ => CallArg::Expr(arg),
                    });
                    let ret = self.call_infer(
                        call_target,
                        &args,
                        &arguments.keywords,
                        range,
                        errors,
                        None,
                    );
                    if let Type::TypeGuard(box t) = ret {
                        return t.clone();
                    }
                }
                ty.clone()
            }
            AtomicNarrowOp::NotTypeGuard(_, _) => ty.clone(),
            AtomicNarrowOp::TypeIs(t, arguments) => {
                if let Some(call_target) = self.as_call_target(t.clone()) {
                    let args = arguments.args.map(|arg| match arg {
                        Expr::Starred(x) => CallArg::Star(&x.value, x.range),
                        _ => CallArg::Expr(arg),
                    });
                    let ret = self.call_infer(
                        call_target,
                        &args,
                        &arguments.keywords,
                        range,
                        errors,
                        None,
                    );
                    if let Type::TypeIs(box t) = ret {
                        return self.intersect(ty, &t);
                    }
                }
                ty.clone()
            }
            AtomicNarrowOp::NotTypeIs(t, arguments) => {
                if let Some(call_target) = self.as_call_target(t.clone()) {
                    let args = arguments.args.map(|arg| match arg {
                        Expr::Starred(x) => CallArg::Star(&x.value, x.range),
                        _ => CallArg::Expr(arg),
                    });
                    let ret = self.call_infer(
                        call_target,
                        &args,
                        &arguments.keywords,
                        range,
                        errors,
                        None,
                    );
                    if let Type::TypeIs(box t) = ret {
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
                        (
                            Type::ClassType(left_cls),
                            Type::Literal(Lit::Enum(box (right_cls, name, _))),
                        ) if *left_cls == *right_cls => self.subtract_enum_member(left_cls, name),
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

    fn get_attribute_type(&self, base: &TypeInfo, attr: &AttributeChain, range: TextRange) -> Type {
        // We don't want to throw any attribute access errors when narrowing - the same code is traversed
        // separately for type checking, and there might be error context then we don't have here.
        let ignore_errors = self.error_swallower();
        let AttributeChain(box names) = attr.clone();
        let (first_name, remaining_names) = names.split_off_first();
        match self.narrowable_for_attr_chain(
            base,
            &first_name,
            &remaining_names,
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

    fn narrowable_for_attr_chain(
        &self,
        base: &TypeInfo,
        first_name: &Name,
        remaining_names: &[Name],
        range: TextRange,
        errors: &ErrorCollector,
    ) -> Narrowable {
        match remaining_names.split_first() {
            None => match base.type_at_name(first_name) {
                Some(ty) => Narrowable::Simple(ty.clone()),
                None => self.narrowable_for_attr(base.ty(), first_name, range, errors),
            },
            Some((next_name, remaining_names)) => {
                let base = self.attr_infer(base, first_name, range, errors, None);
                self.narrowable_for_attr_chain(&base, next_name, remaining_names, range, errors)
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
            NarrowOp::Atomic(Some(attr), op) => {
                let ty = self.atomic_narrow(
                    &self.get_attribute_type(type_info, attr, range),
                    op,
                    range,
                    errors,
                );
                type_info.with_narrow(attr.names(), ty)
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
