/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use pyrefly_graph::index::Idx;
use pyrefly_python::dunder;
use ruff_python_ast::CmpOp;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprBinOp;
use ruff_python_ast::ExprCompare;
use ruff_python_ast::ExprUnaryOp;
use ruff_python_ast::Operator;
use ruff_python_ast::StmtAugAssign;
use ruff_python_ast::UnaryOp;
use ruff_python_ast::name::Name;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use vec1::vec1;

use crate::alt::answers::LookupAnswer;
use crate::alt::answers_solver::AnswersSolver;
use crate::alt::call::CallStyle;
use crate::alt::callable::CallArg;
use crate::alt::unwrap::HintRef;
use crate::binding::binding::KeyAnnotation;
use crate::config::error_kind::ErrorKind;
use crate::error::collector::ErrorCollector;
use crate::error::context::ErrorContext;
use crate::error::context::ErrorInfo;
use crate::error::context::TypeCheckContext;
use crate::error::context::TypeCheckKind;
use crate::types::literal::Lit;
use crate::types::tuple::Tuple;
use crate::types::types::Type;

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
        self.record_resolved_trace(range, method_type.clone());
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
                true,
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
            self.error(
                errors,
                range,
                ErrorInfo::Context(&context),
                format!("Cannot find {dunders}"),
            )
        }
    }

    fn tuple_concat(&self, l: &Tuple, r: &Tuple) -> Type {
        match (l, r) {
            (Tuple::Concrete(l), Tuple::Concrete(r)) => {
                let mut elements = l.clone();
                elements.extend(r.clone());
                Type::concrete_tuple(elements)
            }
            (Tuple::Unbounded(l), Tuple::Unbounded(r)) => {
                Type::unbounded_tuple(self.union((**l).clone(), (**r).clone()))
            }
            (Tuple::Concrete(l), r @ Tuple::Unbounded(_)) => {
                Type::unpacked_tuple(l.clone(), Type::Tuple(r.clone()), Vec::new())
            }
            (l @ Tuple::Unbounded(_), Tuple::Concrete(r)) => {
                Type::unpacked_tuple(Vec::new(), Type::Tuple(l.clone()), r.clone())
            }
            (Tuple::Unpacked(box (l_prefix, l_middle, l_suffix)), Tuple::Concrete(r)) => {
                let mut new_suffix = l_suffix.clone();
                new_suffix.extend(r.clone());
                Type::unpacked_tuple(l_prefix.clone(), l_middle.clone(), new_suffix)
            }
            (Tuple::Concrete(l), Tuple::Unpacked(box (r_prefix, r_middle, r_suffix))) => {
                let mut new_prefix = l.clone();
                new_prefix.extend(r_prefix.clone());
                Type::unpacked_tuple(new_prefix, r_middle.clone(), r_suffix.clone())
            }
            (Tuple::Unbounded(l), Tuple::Unpacked(box (r_prefix, r_middle, r_suffix))) => {
                let mut middle = r_prefix.clone();
                middle.push((**l).clone());
                middle.push(
                    self.unwrap_iterable(r_middle)
                        .unwrap_or(Type::any_implicit()),
                );
                Type::unpacked_tuple(
                    Vec::new(),
                    Type::unbounded_tuple(self.unions(middle)),
                    r_suffix.clone(),
                )
            }
            (Tuple::Unpacked(box (l_prefix, l_middle, l_suffix)), Tuple::Unbounded(r)) => {
                let mut middle = l_suffix.clone();
                middle.push((**r).clone());
                middle.push(
                    self.unwrap_iterable(l_middle)
                        .unwrap_or(Type::any_implicit()),
                );
                Type::unpacked_tuple(
                    l_prefix.clone(),
                    Type::unbounded_tuple(self.unions(middle)),
                    Vec::new(),
                )
            }
            (
                Tuple::Unpacked(box (l_prefix, l_middle, l_suffix)),
                Tuple::Unpacked(box (r_prefix, r_middle, r_suffix)),
            ) => {
                let mut middle = l_suffix.clone();
                middle.extend(r_prefix.clone());
                middle.push(
                    self.unwrap_iterable(l_middle)
                        .unwrap_or(Type::any_implicit()),
                );
                middle.push(
                    self.unwrap_iterable(r_middle)
                        .unwrap_or(Type::any_implicit()),
                );
                Type::unpacked_tuple(
                    l_prefix.clone(),
                    Type::unbounded_tuple(self.unions(middle)),
                    r_suffix.clone(),
                )
            }
        }
    }

    pub fn binop_infer(
        &self,
        x: &ExprBinOp,
        hint: Option<HintRef>,
        errors: &ErrorCollector,
    ) -> Type {
        let binop_call = |op: Operator, lhs: &Type, rhs: &Type, range: TextRange| -> Type {
            let context = || {
                ErrorContext::BinaryOp(
                    op.as_str().to_owned(),
                    self.for_display(lhs.clone()),
                    self.for_display(rhs.clone()),
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
        // If the expression is of the form [X] * Y where Y is a number, pass down the contextual
        // type hint when evaluating [X]
        let lhs;
        let rhs;
        if matches!(&*x.left, Expr::List(_)) && x.op == Operator::Mult {
            rhs = self.expr_infer(&x.right, errors);
            if self.is_subset_eq(&rhs, &self.stdlib.int().clone().to_type()) {
                lhs = self.expr_infer_with_hint(&x.left, hint, errors);
            } else {
                lhs = self.expr_infer(&x.left, errors);
            }
        } else {
            lhs = self.expr_infer(&x.left, errors);
            rhs = self.expr_infer(&x.right, errors);
        }

        // Optimisation: If we have `Union[a, b] | Union[c, d]`, instead of unioning
        // (a | c) | (a | d) | (b | c) | (b | d), we can just do one union.
        if x.op == Operator::BitOr
            && let Some(l) = self.untype_opt(lhs.clone(), x.left.range(), errors)
            && let Some(r) = self.untype_opt(rhs.clone(), x.right.range(), errors)
        {
            return Type::type_form(self.union(l, r));
        }

        self.distribute_over_union(&lhs, |lhs| {
            self.distribute_over_union(&rhs, |rhs| {
                // If an Any appears on the RHS, do not refine the return type based on the LHS.
                // Without loss of generality, consider e1 + e2 where e1 has type int and e2 has type Any.
                // Then e1 + e2 should have a return type of Any since e2's __radd__  signature could be
                // inconsistent with the signature of e1 __add__.
                if let Type::Any(style) = &rhs {
                    style.propagate()
                } else if let Type::Any(style) = &lhs {
                    style.propagate()
                } else if x.op == Operator::BitOr
                    && let Some(l) = self.untype_opt(lhs.clone(), x.left.range(), errors)
                    && let Some(r) = self.untype_opt(rhs.clone(), x.right.range(), errors)
                {
                    Type::type_form(self.union(l, r))
                } else if x.op == Operator::Add
                    && ((*lhs == Type::LiteralString && rhs.is_literal_string())
                        || (*rhs == Type::LiteralString && lhs.is_literal_string()))
                {
                    Type::LiteralString
                } else if x.op == Operator::Add
                    && let Type::Tuple(l) = lhs
                    && let Type::Tuple(r) = rhs
                {
                    self.tuple_concat(l, r)
                } else {
                    binop_call(x.op, lhs, rhs, x.range)
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
        let binop_call = |op: Operator, lhs: &Type, rhs: &Type, range: TextRange| -> Type {
            let context = || {
                ErrorContext::InplaceBinaryOp(
                    op.as_str().to_owned(),
                    self.for_display(lhs.clone()),
                    self.for_display(rhs.clone()),
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
        let tcc: &dyn Fn() -> TypeCheckContext =
            &|| TypeCheckContext::of_kind(TypeCheckKind::AugmentedAssignment);
        let result = self.distribute_over_union(&base, |lhs| {
            self.distribute_over_union(&rhs, |rhs| {
                if let Type::Any(style) = &base {
                    style.propagate()
                } else if x.op == Operator::Add
                    && base.is_literal_string()
                    && rhs.is_literal_string()
                {
                    Type::LiteralString
                } else if x.op == Operator::Add
                    && let Type::Tuple(ref l) = base
                    && let Type::Tuple(r) = rhs
                {
                    self.tuple_concat(l, r)
                } else {
                    binop_call(x.op, lhs, rhs, x.range)
                }
            })
        });
        // If we're assigning to something with an annotation, make sure the produced value is assignable to it
        if let Some(ann) = ann.map(|k| self.get_idx(k)) {
            if ann.annotation.is_final() {
                self.error(
                    errors,
                    x.range(),
                    ErrorInfo::Kind(ErrorKind::BadAssignment),
                    format!("Cannot assign to {} because it is marked final", ann.target),
                );
            }
            if let Some(ann_ty) = ann.ty(self.stdlib) {
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

            let result = self.distribute_over_union(&current_left, |left| {
                self.distribute_over_union(&right, |right| {
                    let context = || {
                        ErrorContext::BinaryOp(
                            op.as_str().to_owned(),
                            self.for_display(left.clone()),
                            self.for_display(right.clone()),
                        )
                    };
                    match op {
                        CmpOp::Is | CmpOp::IsNot => {
                            // These comparisons never error.
                            self.stdlib.bool().clone().to_type()
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
                                // Comparison method called.
                                ret
                            } else {
                                let iteration_errors = self.error_collector();
                                let iterables =
                                    self.iterate(right, x.range, &iteration_errors, Some(&context));
                                if iteration_errors.is_empty() {
                                    // Make sure `x` matches the produced type.
                                    self.check_type(
                                        left,
                                        &self.get_produced_type(iterables),
                                        x.range,
                                        errors,
                                        &|| TypeCheckContext {
                                            kind: TypeCheckKind::Container,
                                            context: Some(context()),
                                        },
                                    );
                                } else {
                                    // Iterating `y` failed.
                                    errors.extend(iteration_errors);
                                }
                                self.stdlib.bool().clone().to_type()
                            }
                        }
                        _ => {
                            // We've handled the other cases above, so we know we have a rich comparison op.
                            let calls_to_try = [
                                (&dunder::rich_comparison_dunder(*op).unwrap(), left, right),
                                (&dunder::rich_comparison_fallback(*op).unwrap(), right, left),
                            ];
                            let ret =
                                self.try_binop_calls(&calls_to_try, x.range, errors, &context);
                            if ret.is_error() {
                                self.stdlib.bool().clone().to_type()
                            } else {
                                ret
                            }
                        }
                    }
                })
            });
            results.push(result);
            // For next comparison, the current right becomes the new left
            current_left = right;
            current_left_range = comparator.range();
        }
        self.unions(results)
    }

    pub fn unop_infer(&self, x: &ExprUnaryOp, errors: &ErrorCollector) -> Type {
        let t = self.expr_infer(&x.operand, errors);
        let unop = |t: &Type, f: &dyn Fn(&Lit) -> Option<Type>, method: &Name| {
            let context =
                || ErrorContext::UnaryOp(x.op.as_str().to_owned(), self.for_display(t.clone()));
            match t {
                Type::Literal(lit) if let Some(ret) = f(lit) => ret,
                Type::ClassType(_) | Type::SelfType(_) => {
                    self.call_method_or_error(t, method, x.range, &[], &[], errors, Some(&context))
                }
                Type::Literal(Lit::Enum(lit_enum)) => self.call_method_or_error(
                    &lit_enum.class.clone().to_type(),
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
                    ErrorInfo::Kind(ErrorKind::UnsupportedOperation),
                    context().format(),
                ),
            }
        };
        self.distribute_over_union(&t, |t| match x.op {
            UnaryOp::USub => {
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
                    None => self.stdlib.bool().clone().to_type(),
                    Some(b) => Type::Literal(Lit::Bool(!b)),
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
                ErrorInfo::Kind(ErrorKind::UnnecessaryComparison),
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
            errors.add(
                range,
                ErrorInfo::Kind(ErrorKind::UnnecessaryComparison),
                vec1![
                    format!(
                        "Identity comparison between an instance of `{}` and class `{}` is always {}",
                        instance_str,
                        class_str,
                        if is_op { "False" } else { "True" }
                    ),
                    format!(
                        "Did you mean to do `{}isinstance(..., {})`?",
                        if is_op { "" } else { "not " },
                        class_str,
                    )
                ],
            );
        };

        match (left, right) {
            // If both are literals/None, check for predictable results
            (Type::Literal(l1), Type::Literal(l2)) => {
                if l1 != l2 {
                    emit_literal_warning(
                        &l1.to_string(),
                        &l2.to_string(),
                        if is_op { "False" } else { "True" },
                    );
                } else if is_bool_literal(l1) {
                    emit_literal_warning(
                        &l1.to_string(),
                        &l2.to_string(),
                        if is_op { "True" } else { "False" },
                    );
                }
            }
            (Type::Literal(l), Type::None) => {
                emit_literal_warning(&l.to_string(), "None", if is_op { "False" } else { "True" });
            }
            (Type::None, Type::Literal(l)) => {
                emit_literal_warning("None", &l.to_string(), if is_op { "False" } else { "True" });
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
}
