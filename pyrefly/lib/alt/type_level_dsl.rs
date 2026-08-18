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
use ruff_python_ast::ExprCall;
use ruff_text_size::Ranged;

use crate::alt::answers::LookupAnswer;
use crate::alt::answers_solver::AnswersSolver;
use crate::alt::solve::TypeFormContext;
use crate::config::error_kind::ErrorKind;
use crate::error::collector::ErrorCollector;

impl TypeFormContext<'_> {
    pub(crate) fn allows_type_level_dsl_call(self) -> bool {
        match self {
            Self::ReturnAnnotation => true,
            Self::TypeArgument(parent) | Self::UnionMember(parent) => {
                parent.allows_type_level_dsl_call()
            }
            _ => false,
        }
    }
}

impl<'a, Ans: LookupAnswer> AnswersSolver<'a, Ans> {
    pub(crate) fn parse_type_level_dsl_call(
        &self,
        call: &ExprCall,
        callee: &Type,
        type_form_context: TypeFormContext<'_>,
        errors: &ErrorCollector,
    ) -> Type {
        if !Self::is_native_broadcast_callee(callee) {
            return self.error(
                errors,
                call.func.range(),
                ErrorKind::InvalidAnnotation,
                format!(
                    "Expected a type-level DSL function, got `{}`",
                    self.for_display(callee.clone())
                ),
            );
        }
        if !call.arguments.keywords.is_empty() {
            return self.error(
                errors,
                call.range,
                ErrorKind::InvalidAnnotation,
                "`broadcast` does not accept keyword arguments".to_owned(),
            );
        }
        if call.arguments.args.len() != 2 {
            return self.error(
                errors,
                call.range,
                ErrorKind::InvalidAnnotation,
                format!(
                    "Expected 2 arguments for `broadcast`, got {}",
                    call.arguments.args.len()
                ),
            );
        }

        let argument_context = TypeFormContext::TypeArgument(&type_form_context);
        let args: Vec<_> = call
            .arguments
            .args
            .iter()
            .map(|arg| {
                let ty = self.expr_untype(arg, argument_context, errors);
                if ty.is_error() {
                    ty
                } else if !self.is_int_tuple_dsl_argument(&ty) {
                    self.error(
                        errors,
                        arg.range(),
                        ErrorKind::InvalidAnnotation,
                        format!(
                            "Expected an `IntTuple` argument to `broadcast`, got `{}`",
                            self.for_display(ty.clone())
                        ),
                    )
                } else {
                    ty
                }
            })
            .collect();
        if args.iter().any(Type::is_error) {
            return Type::any_error();
        }
        Type::TypeLevelDslCall(Box::new(TypeLevelDslCall::broadcast(args)))
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
