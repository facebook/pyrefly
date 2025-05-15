/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

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

use crate::alt::answers::AnswersSolver;
use crate::alt::answers::LookupAnswer;
use crate::alt::call::CallStyle;
use crate::alt::callable::CallArg;
use crate::alt::solve::Iterable;
use crate::binding::binding::KeyAnnotation;
use crate::dunder;
use crate::error::collector::ErrorCollector;
use crate::error::context::ErrorContext;
use crate::error::context::TypeCheckContext;
use crate::error::context::TypeCheckKind;
use crate::error::kind::ErrorKind;
use crate::graph::index::Idx;
use crate::types::literal::Lit;
use crate::types::tuple::Tuple;
use crate::types::types::Type;

impl<'a, Ans: LookupAnswer> AnswersSolver<'a, Ans> {
    fn callable_dunder_helper(
        &self,
        method_type: Type,
        range: TextRange,
        errors: &ErrorCollector,
        context: &dyn Fn() -> ErrorContext,
        opname: &Name,
        call_arg_type: &Type,
    ) -> Type {
        let callable = self.as_call_target_or_error(
            method_type,
            CallStyle::Method(opname),
            range,
            errors,
            Some(context),
        );
        self.call_infer(
            callable,
            &[CallArg::Type(call_arg_type, range)],
            &[],
            range,
            errors,
            Some(context),
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
            );
            let Some(method_type_dunder) = method_type_dunder else {
                continue;
            };
            let dunder_errors = self.error_collector();
            let ret = self.callable_dunder_helper(
                method_type_dunder,
                range,
                &dunder_errors,
                &context,
                dunder,
                arg,
            );
            if dunder_errors.is_empty() {
                return ret;
            } else if first_call.is_none() {
                first_call = Some((dunder_errors, ret));
            }
        }
        if let Some((dunder_errors, ret)) = first_call {
            errors.extend(dunder_errors);
            ret
        } else {
            let dunders = calls
                .iter()
                .map(|(dunder, _, _)| format!("`{}`", dunder))
                .collect::<Vec<_>>()
                .join(" or ");
            self.error(
                errors,
                range,
                ErrorKind::MissingAttribute,
                Some(&context),
                format!("Cannot find {dunders}"),
            )
        }
    }

    fn tuple_concat(&self, l: &Tuple, r: &Tuple) -> Type {
        match (l, r) {
            (Tuple::Concrete(l), Tuple::Concrete(r)) => {
                let mut elements = l.clone();
                elements.extend(r.clone());
                Type::Tuple(Tuple::Concrete(elements))
            }
            (Tuple::Unbounded(box l), Tuple::Unbounded(box r)) => {
                Type::Tuple(Tuple::Unbounded(Box::new(self.union(l.clone(), r.clone()))))
            }
            (Tuple::Concrete(l), r @ Tuple::Unbounded(_)) => Type::Tuple(Tuple::Unpacked(
                Box::new((l.clone(), Type::Tuple(r.clone()), Vec::new())),
            )),
            (l @ Tuple::Unbounded(_), Tuple::Concrete(r)) => Type::Tuple(Tuple::Unpacked(
                Box::new((Vec::new(), Type::Tuple(l.clone()), r.clone())),
            )),
            (Tuple::Unpacked(box (l_prefix, l_middle, l_suffix)), Tuple::Concrete(r)) => {
                let mut new_suffix = l_suffix.clone();
                new_suffix.extend(r.clone());
                Type::Tuple(Tuple::Unpacked(Box::new((
                    l_prefix.clone(),
                    l_middle.clone(),
                    new_suffix,
                ))))
            }
            (Tuple::Concrete(l), Tuple::Unpacked(box (r_prefix, r_middle, r_suffix))) => {
                let mut new_prefix = l.clone();
                new_prefix.extend(r_prefix.clone());
                Type::Tuple(Tuple::Unpacked(Box::new((
                    new_prefix,
                    r_middle.clone(),
                    r_suffix.clone(),
                ))))
            }
            (Tuple::Unbounded(box l), Tuple::Unpacked(box (r_prefix, r_middle, r_suffix))) => {
                let mut middle = r_prefix.clone();
                middle.push(l.clone());
                middle.push(
                    self.unwrap_iterable(r_middle)
                        .unwrap_or(Type::any_implicit()),
                );
                Type::Tuple(Tuple::Unpacked(Box::new((
                    Vec::new(),
                    Type::Tuple(Tuple::unbounded(self.unions(middle))),
                    r_suffix.clone(),
                ))))
            }
            (Tuple::Unpacked(box (l_prefix, l_middle, l_suffix)), Tuple::Unbounded(box r)) => {
                let mut middle = l_suffix.clone();
                middle.push(r.clone());
                middle.push(
                    self.unwrap_iterable(l_middle)
                        .unwrap_or(Type::any_implicit()),
                );
                Type::Tuple(Tuple::Unpacked(Box::new((
                    l_prefix.clone(),
                    Type::Tuple(Tuple::unbounded(self.unions(middle))),
                    Vec::new(),
                ))))
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
                Type::Tuple(Tuple::Unpacked(Box::new((
                    l_prefix.clone(),
                    Type::Tuple(Tuple::unbounded(self.unions(middle))),
                    r_suffix.clone(),
                ))))
            }
        }
    }

    pub fn binop_infer(&self, x: &ExprBinOp, errors: &ErrorCollector) -> Type {
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
        let lhs = self.expr_infer(&x.left, errors);
        let rhs = self.expr_infer(&x.right, errors);
        self.distribute_over_union(&lhs, |lhs| {
            self.distribute_over_union(&rhs, |rhs| {
                if let Type::Any(style) = &lhs {
                    style.propagate()
                } else if x.op == Operator::BitOr
                    && let Some(l) = self.untype_opt(lhs.clone(), x.left.range())
                    && let Some(r) = self.untype_opt(rhs.clone(), x.right.range())
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
                    ErrorKind::BadAssignment,
                    None,
                    format!("Cannot assign to {} because it is marked final", ann.target),
                );
            }
            if let Some(ann_ty) = ann.ty(self.stdlib) {
                return self.check_and_return_type(&ann_ty, result, x.range(), errors, tcc);
            }
        }
        result
    }

    pub fn compare_infer(&self, x: &ExprCompare, errors: &ErrorCollector) -> Type {
        let left = self.expr_infer(&x.left, errors);
        let comparisons = x.ops.iter().zip(x.comparators.iter());
        self.unions(
            comparisons
                .map(|(op, comparator)| {
                    let right = self.expr_infer(comparator, errors);
                    self.distribute_over_union(&left, |left| {
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
                                        &[CallArg::Type(left, x.left.range())],
                                        &[],
                                        errors,
                                        Some(&context),
                                    ) {
                                        // Comparison method called.
                                        ret
                                    } else {
                                        let iteration_errors = self.error_collector();
                                        let iterables =
                                            self.iterate(right, x.range, &iteration_errors);
                                        if !iteration_errors.is_empty()
                                            || !iterables.iter().any(|iterable| match iterable {
                                                Iterable::OfType(ty) => self.is_subset_eq(left, ty),
                                                Iterable::FixedLen(ts) => {
                                                    ts.iter().any(|t| self.is_subset_eq(left, t))
                                                }
                                            })
                                        {
                                            // Iterating `y` failed, or `x` does not match any of the produced types.
                                            self.error(
                                                errors,
                                                x.range,
                                                ErrorKind::UnsupportedOperand,
                                                None,
                                                context().format(),
                                            );
                                        }
                                        self.stdlib.bool().clone().to_type()
                                    }
                                }
                                _ => {
                                    // We've handled the other cases above, so we know we have a rich comparison op.
                                    let calls_to_try = [
                                        (
                                            &dunder::rich_comparison_dunder(*op).unwrap(),
                                            left,
                                            right,
                                        ),
                                        (
                                            &dunder::rich_comparison_fallback(*op).unwrap(),
                                            right,
                                            left,
                                        ),
                                    ];
                                    let ret = self.try_binop_calls(
                                        &calls_to_try,
                                        x.range,
                                        errors,
                                        &context,
                                    );
                                    if ret.is_error() {
                                        self.stdlib.bool().clone().to_type()
                                    } else {
                                        ret
                                    }
                                }
                            }
                        })
                    })
                })
                .collect(),
        )
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
                Type::Literal(Lit::Enum(box (cls, ..))) => self.call_method_or_error(
                    &cls.clone().to_type(),
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
                    ErrorKind::UnsupportedOperand,
                    None,
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
            UnaryOp::Not => match t.as_bool() {
                None => self.stdlib.bool().clone().to_type(),
                Some(b) => Type::Literal(Lit::Bool(!b)),
            },
            UnaryOp::Invert => {
                let f = |lit: &Lit| lit.invert();
                unop(t, &f, &dunder::INVERT)
            }
        })
    }
}
