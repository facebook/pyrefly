/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use pyrefly_types::function::FunctionKind;
use pyrefly_types::quantified::QuantifiedKind;
use pyrefly_types::type_level_dsl::TypeLevelDslCall;
use pyrefly_types::type_var::Restriction;
use pyrefly_types::types::CalleeKind;
use pyrefly_types::types::Type;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprCall;
use ruff_text_size::Ranged;

use crate::alt::answers::LookupAnswer;
use crate::alt::answers_solver::AnswersSolver;
use crate::alt::solve::TypeFormContext;
use crate::config::error_kind::ErrorKind;
use crate::error::collector::ErrorCollector;

impl<'a, Ans: LookupAnswer> AnswersSolver<'a, Ans> {
    pub(crate) fn parse_type_level_dsl_call(
        &self,
        x: &Expr,
        errors: &ErrorCollector,
    ) -> Option<Type> {
        let Expr::Call(call) = x else {
            return None;
        };
        let probe_errors = self.error_swallower();
        let prepared = self.prepare_expr_call(call, &probe_errors);
        let callee = prepared.callee()?;
        self.parse_type_level_dsl_call_with_callee(call, callee, errors)
    }

    pub(crate) fn parse_type_level_dsl_call_with_callee(
        &self,
        call: &ExprCall,
        callee: &Type,
        errors: &ErrorCollector,
    ) -> Option<Type> {
        if !Self::is_native_broadcast_callee(callee) {
            return None;
        }
        if !call.arguments.keywords.is_empty() {
            return Some(self.error(
                errors,
                call.range,
                ErrorKind::InvalidAnnotation,
                "`broadcast` does not accept keyword arguments".to_owned(),
            ));
        }
        if call.arguments.args.len() != 2 {
            return Some(self.error(
                errors,
                call.range,
                ErrorKind::InvalidAnnotation,
                format!(
                    "Expected 2 arguments for `broadcast`, got {}",
                    call.arguments.args.len()
                ),
            ));
        }
        let args = call
            .arguments
            .args
            .iter()
            .map(|arg| {
                let ty = self
                    .parse_type_level_dsl_call(arg, errors)
                    .unwrap_or_else(|| {
                        self.expr_untype(arg, TypeFormContext::TypeArgument, errors)
                    });
                if !self.is_int_tuple_dsl_argument(&ty) {
                    self.error(
                        errors,
                        arg.range(),
                        ErrorKind::InvalidAnnotation,
                        format!(
                            "Expected an `IntTuple` argument to `broadcast`, got `{}`",
                            self.for_display(ty.clone())
                        ),
                    );
                }
                ty
            })
            .collect();
        Some(Type::TypeLevelDslCall(Box::new(
            TypeLevelDslCall::broadcast(args),
        )))
    }

    fn is_native_broadcast_callee(callee: &Type) -> bool {
        matches!(
            callee.callee_kind(),
            Some(CalleeKind::Function(FunctionKind::Def(id)))
                if id.has_toplevel_qname("shape_extensions", "broadcast")
        )
    }

    fn is_int_tuple_dsl_argument(&self, ty: &Type) -> bool {
        let restriction = match ty {
            Type::Any(_) | Type::IntTuple(_) | Type::TypeLevelDslCall(_) => return true,
            Type::Quantified(q) if q.kind == QuantifiedKind::TypeVar => &q.restriction,
            Type::TypeVar(type_var) => type_var.restriction(),
            _ => return false,
        };
        match restriction {
            Restriction::Bound(bound) => matches!(bound, Type::IntTuple(_)),
            Restriction::Constraints(constraints) => {
                !constraints.is_empty()
                    && constraints
                        .iter()
                        .all(|constraint| matches!(constraint, Type::IntTuple(_)))
            }
            Restriction::Unrestricted => false,
        }
    }
}
