/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use pyrefly_graph::index::Idx;
use pyrefly_python::ast::Ast;
use pyrefly_python::dunder;
use pyrefly_types::dimension::SizeExpr;
use pyrefly_types::dimension::canonicalize;
use pyrefly_types::lit_int::LitInt;
use pyrefly_types::literal::LitStyle;
use pyrefly_types::quantified::Quantified;
use pyrefly_types::shaped_array::ShapedArrayType;
use pyrefly_types::shaped_array::broadcast_shapes;
use pyrefly_types::simplify::intersect;
use pyrefly_types::type_var::Restriction;
use pyrefly_util::prelude::VecExt;
use ruff_python_ast::CmpOp;
use ruff_python_ast::ExprBinOp;
use ruff_python_ast::ExprCompare;
use ruff_python_ast::ExprUnaryOp;
use ruff_python_ast::Operator;
use ruff_python_ast::StmtAugAssign;
use ruff_python_ast::UnaryOp;
use ruff_python_ast::name::Name;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;

use crate::alt::answers::LookupAnswer;
use crate::alt::answers_solver::AnswersSolver;
use crate::alt::call::CallStyle;
use crate::alt::callable::CallArg;
use crate::alt::expr::MAX_TUPLE_LENGTH;
use crate::alt::unwrap::HintRef;
use crate::binding::binding::KeyAnnotation;
use crate::config::error_kind::ErrorKind;
use crate::error::collector::ErrorCollector;
use crate::error::context::ErrorContext;
use crate::error::context::TypeCheckContext;
use crate::error::context::TypeCheckKind;
use crate::types::literal::Lit;
use crate::types::tuple::Tuple;
use crate::types::types::Type;

#[derive(Clone, Copy, PartialEq, Eq)]
enum EqualityCompatibilityGroup {
    Numeric,
    BytesLike,
    SetLike,
    Str,
}

impl<'a, Ans: LookupAnswer> AnswersSolver<'a, Ans> {
    fn callable_dunder_helper(
        &self,
        method_type: Type,
        range: TextRange,
        callee_errors: &ErrorCollector,
        call_errors: &ErrorCollector,
        context: &dyn Fn() -> ErrorContext,
        opname: &Name,
        call_arg_type: &Type,
    ) -> Type {
        self.record_resolved_trace(range, &method_type);
        let callable = self.as_call_target_or_error(
            method_type,
            CallStyle::Method(opname),
            range,
            callee_errors,
            Some(context),
        );
        self.call_infer(
            callable,
            &[CallArg::ty(call_arg_type, range)],
            &[],
            range,
            call_errors,
            Some(context),
            None,
            None,
        )
    }

    /// Try to handle binary operations on symbolic integer types (Dim and SizeExpr).
    /// Returns Some(result_type) if the operation was handled, None otherwise.
    fn try_symint_binop(&self, op: Operator, lhs: &Type, rhs: &Type) -> Option<Type> {
        // Only handle if tensor shapes feature is enabled
        if !self.solver().tensor_shapes {
            return None;
        }

        // Only handle arithmetic operations that make sense for dimensions
        if !matches!(
            op,
            Operator::Add | Operator::Sub | Operator::Mult | Operator::FloorDiv
        ) {
            return None;
        }

        // Literal integers are allowed as the non-shape side of dimension arithmetic, but
        // ordinary literal arithmetic should keep the normal integer operator behavior.
        let is_shape_operand = |ty: &Type| match ty {
            Type::Dim(_) | Type::Size(_) => true,
            _ => false,
        };
        if !is_shape_operand(lhs) && !is_shape_operand(rhs) {
            return None;
        }

        // Extract the dimension type from Dim, Size, or an integer literal paired with one.
        let to_dim_type = |ty: &Type| -> Option<Type> {
            match ty {
                Type::Dim(inner_ty) => {
                    // Dim wraps a dimension type (could be SizeExpr, Quantified, etc.)
                    Some((**inner_ty).clone())
                }
                Type::Literal(f) if let Lit::Int(n) = &f.value => {
                    // Convert literal to SizeExpr
                    n.as_i64()
                        .map(|val| self.heap.mk_size(SizeExpr::Literal(val)))
                }
                Type::Size(_) => {
                    // SizeExpr is already a dimension type - pass through
                    Some(ty.clone())
                }
                _ => None,
            }
        };

        let (l_type, r_type) = (to_dim_type(lhs)?, to_dim_type(rhs)?);

        // Perform the operation on the dimension types
        let result_ty = match op {
            Operator::Add => canonicalize(self.heap.mk_size(SizeExpr::add(l_type, r_type))),
            Operator::Sub => canonicalize(self.heap.mk_size(SizeExpr::sub(l_type, r_type))),
            Operator::Mult => canonicalize(self.heap.mk_size(SizeExpr::mul(l_type, r_type))),
            Operator::FloorDiv => {
                canonicalize(self.heap.mk_size(SizeExpr::floor_div(l_type, r_type)))
            }
            _ => unreachable!(),
        };

        // If either operand is Dim, return Dim-wrapped result. Otherwise
        // (e.g., Dim-bounded type parameters), return the dimension expression.
        if matches!(lhs, Type::Dim(_)) || matches!(rhs, Type::Dim(_)) {
            Some(self.heap.mk_dim(result_ty))
        } else {
            Some(result_ty)
        }
    }

    fn try_binop_calls(
        &self,
        calls: &[(&Name, &Type, &Type)],
        range: TextRange,
        errors: &ErrorCollector,
        context: &dyn Fn() -> ErrorContext,
    ) -> Type {
        let mut first_call = None;
        for (dunder, target, arg) in calls {
            let method_type_dunder = self.type_of_magic_dunder_attr(
                target,
                dunder,
                range,
                errors,
                Some(&context),
                "Expr::binop_infer",
                // Magic method lookup for operators should ignore __getattr__/__getattribute__.
                false,
            );
            let Some(method_type_dunder) = method_type_dunder else {
                continue;
            };
            let callee_errors = self.error_collector();
            let call_errors = self.error_collector();
            let ret = self.callable_dunder_helper(
                method_type_dunder,
                range,
                &callee_errors,
                &call_errors,
                &context,
                dunder,
                arg,
            );
            if call_errors.is_empty() {
                errors.extend(callee_errors);
                return ret;
            } else if first_call.is_none() {
                first_call = Some((callee_errors, call_errors, ret));
            }
        }
        if let Some((callee_errors, call_errors, ret)) = first_call {
            errors.extend(callee_errors);
            errors.extend(call_errors);
            ret
        } else {
            let dunders = calls
                .iter()
                .map(|(dunder, _, _)| format!("`{dunder}`"))
                .collect::<Vec<_>>()
                .join(" or ");
            self.error_with_context(
                errors,
                range,
                ErrorKind::UnsupportedOperation,
                format!("Cannot find {dunders}"),
                Some(&context),
            )
        }
    }

    fn tuple_concat(&self, l: &Tuple, r: &Tuple) -> Type {
        match (l, r) {
            (Tuple::Concrete(l), Tuple::Concrete(r)) => {
                let mut elements = l.clone();
                elements.extend(r.clone());
                self.heap.mk_concrete_tuple(elements)
            }
            (Tuple::Unbounded(l), Tuple::Unbounded(r)) => self
                .heap
                .mk_unbounded_tuple(self.union((**l).clone(), (**r).clone())),
            (Tuple::Concrete(l), r @ Tuple::Unbounded(_)) => {
                self.heap
                    .mk_unpacked_tuple(l.clone(), self.heap.mk_tuple(r.clone()), Vec::new())
            }
            (l @ Tuple::Unbounded(_), Tuple::Concrete(r)) => {
                self.heap
                    .mk_unpacked_tuple(Vec::new(), self.heap.mk_tuple(l.clone()), r.clone())
            }
            (Tuple::Unpacked(l), Tuple::Concrete(r)) => {
                let (l_prefix, l_middle, l_suffix) = &**l;
                let mut new_suffix = l_suffix.clone();
                new_suffix.extend(r.clone());
                self.heap
                    .mk_unpacked_tuple(l_prefix.clone(), l_middle.clone(), new_suffix)
            }
            (Tuple::Concrete(l), Tuple::Unpacked(r)) => {
                let (r_prefix, r_middle, r_suffix) = &**r;
                let mut new_prefix = l.clone();
                new_prefix.extend(r_prefix.clone());
                self.heap
                    .mk_unpacked_tuple(new_prefix, r_middle.clone(), r_suffix.clone())
            }
            (Tuple::Unbounded(l), Tuple::Unpacked(r)) => {
                let (r_prefix, r_middle, r_suffix) = &**r;
                let mut middle = r_prefix.clone();
                middle.push((**l).clone());
                middle.push(
                    self.unwrap_iterable(r_middle)
                        .unwrap_or_else(|| self.heap.mk_any_implicit()),
                );
                self.heap.mk_unpacked_tuple(
                    Vec::new(),
                    self.heap.mk_unbounded_tuple(self.unions(middle)),
                    r_suffix.clone(),
                )
            }
            (Tuple::Unpacked(l), Tuple::Unbounded(r)) => {
                let (l_prefix, l_middle, l_suffix) = &**l;
                let mut middle = l_suffix.clone();
                middle.push((**r).clone());
                middle.push(
                    self.unwrap_iterable(l_middle)
                        .unwrap_or_else(|| self.heap.mk_any_implicit()),
                );
                self.heap.mk_unpacked_tuple(
                    l_prefix.clone(),
                    self.heap.mk_unbounded_tuple(self.unions(middle)),
                    Vec::new(),
                )
            }
            (Tuple::Unpacked(l), Tuple::Unpacked(r)) => {
                let (l_prefix, l_middle, l_suffix) = &**l;
                let (r_prefix, r_middle, r_suffix) = &**r;
                let mut middle = l_suffix.clone();
                middle.extend(r_prefix.clone());
                middle.push(
                    self.unwrap_iterable(l_middle)
                        .unwrap_or_else(|| self.heap.mk_any_implicit()),
                );
                middle.push(
                    self.unwrap_iterable(r_middle)
                        .unwrap_or_else(|| self.heap.mk_any_implicit()),
                );
                self.heap.mk_unpacked_tuple(
                    l_prefix.clone(),
                    self.heap.mk_unbounded_tuple(self.unions(middle)),
                    r_suffix.clone(),
                )
            }
        }
    }

    fn try_tuple_repeat(&self, lhs: &Type, rhs: &Type) -> Option<Type> {
        let (elements, repeats) = match (lhs, rhs) {
            (Type::Tuple(Tuple::Concrete(elts)), Type::Literal(f))
                if let Lit::Int(n) = &f.value =>
            {
                (elts, n.as_i64()?)
            }
            (Type::Literal(f), Type::Tuple(Tuple::Concrete(elts)))
                if let Lit::Int(n) = &f.value =>
            {
                (elts, n.as_i64()?)
            }
            _ => return None,
        };
        if repeats <= 0 {
            return Some(self.heap.mk_concrete_tuple(Vec::new()));
        }
        if repeats == 1 {
            return Some(self.heap.mk_concrete_tuple(elements.clone()));
        }
        let repeats = usize::try_from(repeats).ok()?;
        let total_len = elements.len().checked_mul(repeats)?;
        if total_len > MAX_TUPLE_LENGTH {
            return None;
        }
        let mut repeated = Vec::with_capacity(total_len);
        for _ in 0..repeats {
            repeated.extend(elements.iter().cloned());
        }
        Some(self.heap.mk_concrete_tuple(repeated))
    }

    pub fn binop_infer(
        &self,
        x: &ExprBinOp,
        hint: Option<HintRef>,
        errors: &ErrorCollector,
    ) -> Type {
        let lhs;
        let rhs;
        if Ast::is_list_literal_or_comprehension(&x.left) && x.op == Operator::Mult {
            // If the expression is of the form [X] * Y where Y is a number, pass down the contextual
            // type hint when evaluating [X]
            rhs = self.expr_infer(&x.right, errors);
            if self.is_subset_eq(&rhs, &self.heap.mk_class_type(self.stdlib.int().clone())) {
                lhs = self.expr_infer_with_hint(&x.left, hint, errors);
            } else {
                lhs = self.expr_infer(&x.left, errors);
            }
        } else if x.op == Operator::Add
            && Ast::is_list_literal_or_comprehension(&x.left)
            && Ast::is_list_literal_or_comprehension(&x.right)
        {
            // If both operands are list literals, pass the contextual hint down
            lhs = self.expr_infer_with_hint(&x.left, hint, errors);
            rhs = self.expr_infer_with_hint(&x.right, hint, errors);
        } else {
            lhs = self.expr_infer(&x.left, errors);
            rhs = self.expr_infer(&x.right, errors);
        }

        // Optimisation: If we have `Union[a, b] | Union[c, d]`, instead of unioning
        // (a | c) | (a | d) | (b | c) | (b | d), we can just do one union.
        if x.op == Operator::BitOr
            && !lhs.is_any()
            && !rhs.is_any()
            && let Some(l) = self.untype_opt(lhs.clone(), x.left.range(), errors)
            && let Some(r) = self.untype_opt(rhs.clone(), x.right.range(), errors)
        {
            return self.heap.mk_type_of(self.union(l, r));
        }

        if matches!(x.op, Operator::Div | Operator::FloorDiv | Operator::Mod)
            && Self::is_literal_zero(&rhs)
        {
            self.error(
                errors,
                x.range,
                ErrorKind::DivisionByZero,
                format!(
                    "Cannot divide by zero: `{}` with a literal zero divisor",
                    x.op.as_str()
                ),
            );
        }

        self.binop_types(x, &lhs, &rhs, errors)
    }

    /// Check if a type is exactly `Literal[0]` (either implicit or explicit).
    fn is_literal_zero(ty: &Type) -> bool {
        matches!(ty, Type::Literal(f) if f.value == Lit::Int(LitInt::new(0)))
    }

    /// Performs an operation `f` on a quantified `q`, which may be narrowed to a concrete type.
    fn on_quantified(
        &self,
        q: &Quantified,
        narrowed_type: Option<&Type>,
        f: &dyn Fn(&Type) -> Type,
    ) -> Type {
        if let Restriction::Constraints(constraints) = &q.restriction {
            let cur_constraints = if let Some(narrowed_type) = narrowed_type {
                vec![narrowed_type]
            } else {
                constraints.iter().collect()
            };
            self.unions(cur_constraints.into_map(|constraint| {
                let res = f(constraint);
                if res == *constraint {
                    // If f returned the constraint unchanged, preserve the quantified.
                    intersect(
                        vec![q.clone().to_type(self.heap), res.clone()],
                        res,
                        self.heap,
                    )
                } else {
                    res
                }
            }))
        } else if let Some(narrowed_type) = narrowed_type {
            f(narrowed_type)
        } else {
            f(&q.upper_bound(self.stdlib, self.heap))
        }
    }

    fn on_quantifieds(
        &self,
        left: &Type,
        right: &Type,
        f: &dyn Fn(&Type, &Type) -> Type,
    ) -> Option<Type> {
        match (left.as_quantified(), right.as_quantified()) {
            (Some((left_q, left_narrow)), Some((right_q, right_narrow)))
                if matches!(left_q.restriction(), Restriction::Constraints(_))
                    && left_q == right_q =>
            {
                Some(
                    self.on_quantified(left_q, left_narrow.or(right_narrow), &|constraint| {
                        f(constraint, constraint)
                    }),
                )
            }
            // We skip non-union bounds to avoid accidentally erasing `Self` typevars.
            (Some((left_q, left_narrow)), _)
                if matches!(
                    left_q.restriction(),
                    Restriction::Constraints(_) | Restriction::Bound(Type::Union(_))
                ) =>
            {
                Some(
                    self.on_quantified(left_q, left_narrow, &|left_restriction| {
                        f(left_restriction, right)
                    }),
                )
            }
            (_, Some((right_q, right_narrow)))
                if matches!(
                    right_q.restriction(),
                    Restriction::Constraints(_) | Restriction::Bound(Type::Union(_))
                ) =>
            {
                Some(
                    self.on_quantified(right_q, right_narrow, &|right_restriction| {
                        f(left, right_restriction)
                    }),
                )
            }
            _ => None,
        }
    }

    fn binop_types(&self, x: &ExprBinOp, lhs: &Type, rhs: &Type, errors: &ErrorCollector) -> Type {
        let left_range = x.left.range();
        let right_range = x.right.range();
        let binop_call = |op: Operator, lhs: &Type, rhs: &Type, range: TextRange| -> Type {
            let context = || {
                ErrorContext::BinaryOp(
                    op.as_str().to_owned(),
                    self.for_display(lhs.clone()),
                    self.for_display(rhs.clone()),
                    left_range,
                    right_range,
                )
            };
            // Reflected operator implementation: This deviates from the runtime semantics by calling the reflected dunder if the regular dunder call errors.
            // At runtime, the reflected dunder is called only if the regular dunder method doesn't exist or if it returns NotImplemented.
            // This deviation is necessary, given that the typeshed stubs don't record when NotImplemented is returned
            let calls_to_try = [
                (&Name::new_static(op.dunder()), lhs, rhs),
                (&Name::new_static(op.reflected_dunder()), rhs, lhs),
            ];
            self.try_binop_calls(&calls_to_try, range, errors, &context)
        };
        self.distribute_over_union(lhs, |lhs| {
            self.distribute_over_union(rhs, |rhs| {
                // If an Any appears on the RHS, do not refine the return type based on the LHS.
                // Without loss of generality, consider e1 + e2 where e1 has type int and e2 has type Any.
                // Then e1 + e2 should have a return type of Any since e2's __radd__  signature could be
                // inconsistent with the signature of e1 __add__.
                //
                // Exception: when one operand is a shaped Tensor, fall through
                // to dunder dispatch. Tensor's arithmetic dunders accept any
                // numeric type and return Self, so the shape is preserved
                // regardless of the other operand's type. Without this, e.g.
                // Tensor[B, 1] / (2**n - 1.0) loses shape because 2**n is Any.
                if (lhs.is_any() || rhs.is_any())
                    && !matches!(lhs, Type::ShapedArray(_))
                    && !matches!(rhs, Type::ShapedArray(_))
                {
                    if let Type::Any(style) = &rhs {
                        return style.propagate();
                    } else if let Type::Any(style) = &lhs {
                        return style.propagate();
                    }
                }
                if x.op == Operator::BitOr
                    && let Some(l) = self.untype_opt(lhs.clone(), x.left.range(), errors)
                    && let Some(r) = self.untype_opt(rhs.clone(), x.right.range(), errors)
                {
                    self.heap.mk_type_of(self.union(l, r))
                } else if x.op == Operator::Add
                    && let Some(lhs_style) = lhs.lit_string_style()
                    && let Some(rhs_style) = rhs.lit_string_style()
                {
                    self.heap.mk_literal_string(match (lhs_style, rhs_style) {
                        (LitStyle::Explicit, LitStyle::Explicit) => LitStyle::Explicit,
                        _ => LitStyle::Implicit,
                    })
                } else if x.op == Operator::Add
                    && let Type::Tuple(l) = lhs
                    && let Type::Tuple(r) = rhs
                {
                    self.tuple_concat(l, r)
                } else if x.op == Operator::Mult
                    && let Some(result) = self.try_tuple_repeat(lhs, rhs)
                {
                    result
                } else if matches!(
                    x.op,
                    Operator::Add
                        | Operator::Sub
                        | Operator::Mult
                        | Operator::Div
                        | Operator::Mod
                        | Operator::Pow
                        | Operator::FloorDiv
                ) && let Type::ShapedArray(l_shaped_array) = lhs
                    && let Type::ShapedArray(r_shaped_array) = rhs
                {
                    // Tensor element-wise operations with broadcasting
                    self.broadcast_shaped_array_binop(
                        l_shaped_array,
                        r_shaped_array,
                        x.range,
                        errors,
                    )
                } else if let Some(result) = self.try_symint_binop(x.op, lhs, rhs) {
                    result
                } else if x.op == Operator::Pow
                    && self.is_subset_eq(lhs, &self.heap.mk_class_type(self.stdlib.int().clone()))
                    && self.is_subset_eq(rhs, &self.heap.mk_class_type(self.stdlib.int().clone()))
                {
                    match rhs {
                        // Special case int ** int
                        // if the exponent is 0, return Literal[1] (x ** 0 = 1)
                        // if the exponent is a positive int, return int
                        // if the exponent is a negative int, return float
                        // if the exponent is unknown, call the `__pow__` method like normal
                        Type::Literal(f) if let Lit::Int(n) = &f.value => {
                            if *n == LitInt::new(0) {
                                LitInt::new(1).to_implicit_type()
                            } else if *n < LitInt::new(0) {
                                self.heap.mk_class_type(self.stdlib.float().clone())
                            } else {
                                self.heap.mk_class_type(self.stdlib.int().clone())
                            }
                        }
                        _ => {
                            let context = || {
                                ErrorContext::BinaryOp(
                                    x.op.as_str().to_owned(),
                                    self.for_display(lhs.clone()),
                                    self.for_display(rhs.clone()),
                                    left_range,
                                    right_range,
                                )
                            };
                            let calls = [
                                (&Name::new_static(x.op.dunder()), lhs, rhs),
                                (&Name::new_static(x.op.reflected_dunder()), rhs, lhs),
                            ];
                            self.try_binop_calls(&calls, x.range, errors, &context)
                        }
                    }
                } else {
                    self.on_quantifieds(lhs, rhs, &|left, right| {
                        self.binop_types(x, left, right, errors)
                    })
                    .unwrap_or_else(|| binop_call(x.op, lhs, rhs, x.range))
                }
            })
        })
    }

    pub fn augassign_infer(
        &self,
        ann: Option<Idx<KeyAnnotation>>,
        x: &StmtAugAssign,
        errors: &ErrorCollector,
    ) -> Type {
        let target_range = x.target.range();
        let value_range = x.value.range();
        let binop_call = |op: Operator, lhs: &Type, rhs: &Type, range: TextRange| -> Type {
            let context = || {
                ErrorContext::InplaceBinaryOp(
                    op.as_str().to_owned(),
                    self.for_display(lhs.clone()),
                    self.for_display(rhs.clone()),
                    target_range,
                    value_range,
                )
            };
            let calls_to_try = [
                (&Name::new_static(op.in_place_dunder()), lhs, rhs),
                (&Name::new_static(op.dunder()), lhs, rhs),
                (&Name::new_static(op.reflected_dunder()), rhs, lhs),
            ];
            self.try_binop_calls(&calls_to_try, range, errors, &context)
        };
        let base = self.expr_infer(&x.target, errors);
        let rhs = self.expr_infer(&x.value, errors);
        if matches!(x.op, Operator::Div | Operator::FloorDiv | Operator::Mod)
            && Self::is_literal_zero(&rhs)
        {
            self.error(
                errors,
                x.range,
                ErrorKind::DivisionByZero,
                format!(
                    "Cannot divide by zero: `{}=` with a literal zero divisor",
                    x.op.as_str()
                ),
            );
        }
        let tcc: &dyn Fn() -> TypeCheckContext =
            &|| TypeCheckContext::of_kind(TypeCheckKind::AugmentedAssignment);
        let result = self.distribute_over_union(&base, |lhs| {
            self.distribute_over_union(&rhs, |rhs| {
                if let Type::Any(style) = &lhs {
                    style.propagate()
                } else if let Type::Any(style) = &rhs {
                    style.propagate()
                } else if x.op == Operator::Add
                    && let Some(lhs_style) = lhs.lit_string_style()
                    && let Some(rhs_style) = rhs.lit_string_style()
                {
                    self.heap.mk_literal_string(match (lhs_style, rhs_style) {
                        (LitStyle::Explicit, LitStyle::Explicit) => LitStyle::Explicit,
                        _ => LitStyle::Implicit,
                    })
                } else if x.op == Operator::Add
                    && let Type::Tuple(ref l) = base
                    && let Type::Tuple(r) = rhs
                {
                    self.tuple_concat(l, r)
                } else if let Some(result) = self.try_symint_binop(x.op, lhs, rhs) {
                    result
                } else {
                    binop_call(x.op, lhs, rhs, x.range)
                }
            })
        });
        // If we're assigning to something with an annotation, make sure the produced value is assignable to it
        if let Some(ann) = ann.map(|k| self.get_idx(k)) {
            self.check_final_reassignment(&ann, x.range(), errors);
            if let Some(ann_ty) = ann.ty(self.heap, self.stdlib) {
                if result.is_any() {
                    // Any provides no useful narrowing information, so preserve
                    // the declared type rather than letting Any leak through.
                    return ann_ty;
                }
                return self.check_and_return_type(result, &ann_ty, x.range(), errors, tcc);
            }
        }
        result
    }

    pub fn compare_infer(&self, x: &ExprCompare, errors: &ErrorCollector) -> Type {
        // For chained comparisons like `a < b < c`, Python evaluates as `(a < b) and (b < c)`.
        // We need to track the current left operand as we iterate through the chain.
        let mut current_left = self.expr_infer(&x.left, errors);
        let mut current_left_range = x.left.range();
        let mut results = Vec::new();
        for (op, comparator) in x.ops.iter().zip(x.comparators.iter()) {
            let right = self.expr_infer(comparator, errors);

            // Check for unnecessary identity comparisons (is/is not) BEFORE distribute_over_union
            // to avoid false positives with union types.
            self.check_unnecessary_comparison(
                &current_left,
                &right,
                *op,
                comparator.range(),
                errors,
            );
            self.check_incompatible_comparison(
                &current_left,
                &right,
                *op,
                comparator.range(),
                errors,
            );

            let right_range = comparator.range();
            let result = self.compare_types(
                x,
                *op,
                &current_left,
                &right,
                current_left_range,
                right_range,
                errors,
            );
            results.push(result);
            // For next comparison, the current right becomes the new left
            current_left = right;
            current_left_range = comparator.range();
        }
        self.unions(results)
    }

    fn compare_types(
        &self,
        x: &ExprCompare,
        op: CmpOp,
        left: &Type,
        right: &Type,
        current_left_range: TextRange,
        current_right_range: TextRange,
        errors: &ErrorCollector,
    ) -> Type {
        self.distribute_over_union(left, |left| {
            self.distribute_over_union(right, |right| {
                match (left, right) {
                    // If either operand is Any, the comparison result is Any.
                    // This mirrors the same check in binop_infer.
                    (Type::Any(style), _) => style.propagate(),
                    (_, Type::Any(style)) => style.propagate(),
                    // If the RHS of a containment check isn't a quantified, it may contain a
                    // nested quantified that on_quantifieds would fail to detect.
                    _ if (!matches!(op, CmpOp::In | CmpOp::NotIn)
                        || right.as_quantified().is_some())
                        && let Some(ret_if_quantified) =
                            self.on_quantifieds(left, right, &|left, right| {
                                self.compare_types(
                                    x,
                                    op,
                                    left,
                                    right,
                                    current_left_range,
                                    current_right_range,
                                    errors,
                                )
                            }) =>
                    {
                        ret_if_quantified
                    }
                    _ => {
                        let context = || {
                            ErrorContext::BinaryOp(
                                op.as_str().to_owned(),
                                self.for_display(left.clone()),
                                self.for_display(right.clone()),
                                current_left_range,
                                current_right_range,
                            )
                        };
                        match op {
                            CmpOp::Is | CmpOp::IsNot => {
                                // These comparisons never error.
                                self.heap.mk_class_type(self.stdlib.bool().clone())
                            }
                            CmpOp::In | CmpOp::NotIn => {
                                // See https://docs.python.org/3/reference/expressions.html#membership-test-operations.
                                // `x in y` first tries `y.__contains__(x)`, then checks if `x` matches an element
                                // obtained by iterating over `y`.
                                if let Some(ret) = self.call_magic_dunder_method(
                                    right,
                                    &dunder::CONTAINS,
                                    x.range,
                                    &[CallArg::ty(left, current_left_range)],
                                    &[],
                                    errors,
                                    Some(&context),
                                ) {
                                    self.check_dunder_bool_is_callable(&ret, x.range, errors);
                                    self.heap.mk_class_type(self.stdlib.bool().clone())
                                } else {
                                    let iteration_errors = self.error_collector();
                                    let iterables = self.iterate(
                                        right,
                                        x.range,
                                        &iteration_errors,
                                        Some(&context),
                                    );
                                    if iteration_errors.is_empty() {
                                        // Make sure `x` matches the produced type.
                                        self.check_type(
                                            left,
                                            &self.get_produced_type(iterables),
                                            x.range,
                                            errors,
                                            &|| {
                                                TypeCheckContext::of_kind(TypeCheckKind::Container)
                                                    .with_context(Some(context()))
                                            },
                                        );
                                    } else {
                                        // Iterating `y` failed.
                                        errors.extend(iteration_errors);
                                    }
                                    self.heap.mk_class_type(self.stdlib.bool().clone())
                                }
                            }
                            _ => {
                                // We've handled the other cases above, so we know we have a rich comparison op.
                                let calls_to_try = [
                                    (&dunder::rich_comparison_dunder(op).unwrap(), left, right),
                                    (&dunder::rich_comparison_fallback(op).unwrap(), right, left),
                                ];
                                let ret =
                                    self.try_binop_calls(&calls_to_try, x.range, errors, &context);
                                if ret.is_error() {
                                    self.heap.mk_class_type(self.stdlib.bool().clone())
                                } else {
                                    ret
                                }
                            }
                        }
                    }
                }
            })
        })
    }

    pub fn unop_infer(&self, x: &ExprUnaryOp, errors: &ErrorCollector) -> Type {
        let t = self.expr_infer(&x.operand, errors);
        let unop = |t: &Type, f: &dyn Fn(&Lit) -> Option<Type>, method: &Name| {
            let operand_range = x.operand.range();
            let context = || {
                ErrorContext::UnaryOp(
                    x.op.as_str().to_owned(),
                    self.for_display(t.clone()),
                    operand_range,
                )
            };
            match t {
                Type::Literal(lit) if let Some(ret) = f(&lit.value) => ret,
                Type::ClassType(_)
                | Type::SelfType(_)
                | Type::Quantified(_)
                | Type::ShapedArray(_)
                | Type::NNModule(_) => {
                    self.call_method_or_error(t, method, x.range, &[], &[], errors, Some(&context))
                }
                Type::Literal(lit) if let Lit::Enum(lit_enum) = &lit.value => self
                    .call_method_or_error(
                        &self.heap.mk_class_type(lit_enum.class.clone()),
                        method,
                        x.range,
                        &[],
                        &[],
                        errors,
                        Some(&context),
                    ),
                Type::Any(style) => style.propagate(),
                _ => self.error(
                    errors,
                    x.range,
                    ErrorKind::UnsupportedOperation,
                    context().format(),
                ),
            }
        };
        self.distribute_over_union(&t, |t| match x.op {
            UnaryOp::USub => {
                // Special handling for Dim: model -N as Sub(0, N)
                if let Type::Dim(inner_ty) = t {
                    let zero = self.heap.mk_size(SizeExpr::Literal(0));
                    let result_ty =
                        canonicalize(self.heap.mk_size(SizeExpr::sub(zero, (**inner_ty).clone())));
                    return self.heap.mk_dim(result_ty);
                }
                let f = |lit: &Lit| lit.negate();
                unop(t, &f, &dunder::NEG)
            }
            UnaryOp::UAdd => {
                let f = |lit: &Lit| lit.positive();
                unop(t, &f, &dunder::POS)
            }
            UnaryOp::Not => {
                self.check_dunder_bool_is_callable(t, x.range, errors);
                match t.as_bool() {
                    None => self.heap.mk_class_type(self.stdlib.bool().clone()),
                    Some(b) => Lit::Bool(!b).to_implicit_type(),
                }
            }
            UnaryOp::Invert => {
                let f = |lit: &Lit| lit.invert();
                unop(t, &f, &dunder::INVERT)
            }
        })
    }

    /// Checks for unnecessary identity comparisons.
    ///
    /// Only emits warnings for identity comparisons (`is` or `is not`) between literals
    /// whose comparison result is statically known.
    /// Returns early without warnings for other comparison operators.
    fn check_unnecessary_comparison(
        &self,
        left: &Type,
        right: &Type,
        op: CmpOp,
        range: TextRange,
        errors: &ErrorCollector,
    ) {
        // Only check identity comparisons
        if !matches!(op, CmpOp::Is | CmpOp::IsNot) {
            return;
        }

        let is_op = matches!(op, CmpOp::Is);
        let is_bool_literal = |lit: &Lit| matches!(lit, Lit::Bool(_));
        let emit_literal_warning = |left_str: &str, right_str: &str, result: &str| {
            self.error(
                errors,
                range,
                ErrorKind::UnnecessaryComparison,
                format!(
                    "Identity comparison `{} {} {}` is always {}",
                    left_str,
                    if is_op { "is" } else { "is not" },
                    right_str,
                    result
                ),
            );
        };
        let emit_instance_is_class_warning = |instance_str: &str, class_str: &str, is_op: bool| {
            errors
                .error_builder(
                    range,
                    ErrorKind::UnnecessaryComparison,
                    format!(
                        "Identity comparison between an instance of `{}` and class `{}` is always {}",
                        instance_str,
                        class_str,
                        if is_op { "False" } else { "True" }
                    ),
                )
                .with_detail(format!(
                    "Did you mean to do `{}isinstance(..., {})`?",
                    if is_op { "" } else { "not " },
                    class_str,
                ))
                .emit();
        };

        match (left, right) {
            // If both are literals/None, check for predictable results
            (Type::Literal(l1), Type::Literal(l2)) => {
                if l1 != l2 {
                    emit_literal_warning(
                        &l1.value.to_string(),
                        &l2.value.to_string(),
                        if is_op { "False" } else { "True" },
                    );
                } else if is_bool_literal(&l1.value) {
                    emit_literal_warning(
                        &l1.value.to_string(),
                        &l2.value.to_string(),
                        if is_op { "True" } else { "False" },
                    );
                }
            }
            (Type::Literal(l), Type::None) => {
                emit_literal_warning(
                    &l.value.to_string(),
                    "None",
                    if is_op { "False" } else { "True" },
                );
            }
            (Type::None, Type::Literal(l)) => {
                emit_literal_warning(
                    "None",
                    &l.value.to_string(),
                    if is_op { "False" } else { "True" },
                );
            }

            // ClassDef vs ClassType - disjoint unless ClassType is `type`, `object`,
            // or another metaclass (subclass of type)
            (Type::ClassDef(cdef), ctype @ Type::ClassType(cls))
            | (ctype @ Type::ClassType(cls), Type::ClassDef(cdef)) => {
                // A class object is an instance of `type` (or a metaclass), so it's only
                // compatible with ClassType if that ClassType is `type`, `object`, or a metaclass
                let is_metaclass_or_object = cls.is_builtin("object")
                    || self.has_superclass(
                        cls.class_object(),
                        self.stdlib.builtins_type().class_object(),
                    );
                if !is_metaclass_or_object {
                    emit_instance_is_class_warning(&ctype.to_string(), cdef.name().as_str(), is_op);
                }
            }

            // All other combinations: no warning
            _ => {}
        }
    }

    /// Handle element-wise binary operations on tensors with broadcasting.
    /// broadcast_shapes handles all shape variants: Concrete shapes are broadcast
    /// precisely, Unpacked shapes match suffix then middles then prefixes, and
    /// mixed Concrete+Unpacked aligns against the suffix.
    fn broadcast_shaped_array_binop(
        &self,
        left: &ShapedArrayType,
        right: &ShapedArrayType,
        range: TextRange,
        errors: &ErrorCollector,
    ) -> Type {
        match broadcast_shapes(&left.shape, &right.shape) {
            Ok(result_shape) => self.shaped_array_with_shape(left, result_shape).to_type(),
            Err(err) => {
                self.error(
                    errors,
                    range,
                    ErrorKind::UnsupportedOperation,
                    format!("Cannot broadcast tensor shapes: {}", err),
                );
                Type::any_error()
            }
        }
    }
    /// Checks for incompatible equality comparisons between types that cannot overlap.
    fn check_incompatible_comparison(
        &self,
        left: &Type,
        right: &Type,
        op: CmpOp,
        range: TextRange,
        errors: &ErrorCollector,
    ) {
        if !matches!(op, CmpOp::Eq | CmpOp::NotEq) {
            return;
        }
        if left.is_any() || right.is_any() || left.is_never() || right.is_never() {
            return;
        }
        let Some(left_group) = self.equality_compatibility_group(left) else {
            return;
        };
        let Some(right_group) = self.equality_compatibility_group(right) else {
            return;
        };
        if left_group == right_group {
            return;
        }
        let left_display = self.for_display(left.clone());
        let right_display = self.for_display(right.clone());
        self.error(
            errors,
            range,
            ErrorKind::IncompatibleComparison,
            format!(
                "Comparison `{}` between incompatible types `{}` and `{}`",
                op.as_str(),
                left_display,
                right_display
            ),
        );
    }

    fn equality_compatibility_group(&self, ty: &Type) -> Option<EqualityCompatibilityGroup> {
        let class = match ty {
            Type::ClassType(cls) => cls.class_object(),
            Type::Literal(lit) => lit.value.general_class_type(self.stdlib).class_object(),
            Type::LiteralString(_) => self.stdlib.str().class_object(),
            _ => return None,
        };
        match (class.qname().module_name().as_str(), class.name().as_str()) {
            ("builtins", "bool" | "int" | "float" | "complex") | ("decimal", "Decimal") => {
                Some(EqualityCompatibilityGroup::Numeric)
            }
            ("builtins", "bytes" | "bytearray" | "memoryview") => {
                Some(EqualityCompatibilityGroup::BytesLike)
            }
            ("builtins", "set" | "frozenset") => Some(EqualityCompatibilityGroup::SetLike),
            ("builtins", "str") => Some(EqualityCompatibilityGroup::Str),
            _ => None,
        }
    }
}
