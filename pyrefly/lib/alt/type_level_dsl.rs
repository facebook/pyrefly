/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::slice;
use std::sync::Arc;

use pyrefly_types::callable::Param;
use pyrefly_types::function::FunctionKind;
use pyrefly_types::quantified::QuantifiedKind;
use pyrefly_types::type_level_dsl::TypeLevelDslCall;
use pyrefly_types::type_level_dsl::TypeShapeDslDomain;
use pyrefly_types::type_level_dsl::ValidatedTypeShapeDslFunction;
use pyrefly_types::type_var::Restriction;
use pyrefly_types::types::CalleeKind;
use pyrefly_types::types::Type;
use ruff_python_ast::ExprCall;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;

use crate::alt::answers::LookupAnswer;
use crate::alt::answers_solver::AnswersSolver;
use crate::alt::solve::TypeFormContext;
use crate::config::error_kind::ErrorKind;
use crate::error::collector::ErrorCollector;

impl TypeFormContext<'_> {
    pub(crate) fn allows_type_level_dsl_call(self) -> bool {
        match self {
            Self::ReturnAnnotation => true,
            Self::TypeArgument(parent) | Self::TupleElement(parent) | Self::UnionMember(parent) => {
                parent.allows_type_level_dsl_call()
            }
            _ => false,
        }
    }
}

impl<'a, Ans: LookupAnswer> AnswersSolver<'a, Ans> {
    /// Validates resolved DSL annotations, emitting diagnostics and metadata only on success.
    pub(super) fn validate_type_shape_dsl_declaration(
        &self,
        dsl: &Arc<ValidatedTypeShapeDslFunction>,
        params: &[Param],
        return_type: &Type,
        function_kind: &FunctionKind,
        function_range: TextRange,
        errors: &ErrorCollector,
    ) -> Option<FunctionKind> {
        let parameter_domain = if dsl.has_parameter_annotation() {
            params
                .first()
                .and_then(|param| type_shape_dsl_domain(param.as_type()))
        } else {
            None
        };
        let return_domain = if dsl.has_return_annotation() {
            type_shape_dsl_domain(return_type)
        } else {
            None
        };
        match (parameter_domain, return_domain) {
            (Some(parameter), Some(result)) if parameter == result => {
                if let FunctionKind::Def(func_id) = function_kind {
                    Some(FunctionKind::TypeShapeDsl(
                        func_id.clone(),
                        parameter,
                        dsl.clone(),
                    ))
                } else {
                    self.error(
                        errors,
                        function_range,
                        ErrorKind::InvalidArgument,
                        "`@type_shape_dsl_function` must be applied to an ordinary function definition"
                            .to_owned(),
                    );
                    None
                }
            }
            (None, _) => {
                self.error(
                    errors,
                    dsl.parameter_annotation_range(),
                    ErrorKind::InvalidArgument,
                    format!(
                        "`@type_shape_dsl_function` parameter `{}` must be annotated as `Int` or `IntTuple`",
                        dsl.parameter_name()
                    ),
                );
                None
            }
            (_, None) => {
                self.error(
                    errors,
                    dsl.return_annotation_range(),
                    ErrorKind::InvalidArgument,
                    "`@type_shape_dsl_function` return must be annotated as `Int` or `IntTuple`"
                        .to_owned(),
                );
                None
            }
            (Some(_), Some(_)) => {
                self.error(
                    errors,
                    dsl.return_annotation_range(),
                    ErrorKind::InvalidArgument,
                    "`@type_shape_dsl_function` parameter and return annotations must use the same domain"
                        .to_owned(),
                );
                None
            }
        }
    }

    pub(crate) fn parse_type_level_dsl_call(
        &self,
        call: &ExprCall,
        callee: &Type,
        type_form_context: TypeFormContext<'_>,
        errors: &ErrorCollector,
    ) -> Type {
        match callee.callee_kind() {
            Some(CalleeKind::Function(FunctionKind::TypeShapeDsl(_, domain, function))) => self
                .parse_user_defined_type_level_dsl_call(
                    call,
                    function,
                    domain,
                    type_form_context,
                    errors,
                ),
            Some(CalleeKind::Function(FunctionKind::Def(id)))
                if id.has_toplevel_qname("shape_extensions", "broadcast") =>
            {
                self.parse_broadcast_type_level_dsl_call(call, type_form_context, errors)
            }
            _ => self.error(
                errors,
                call.func.range(),
                ErrorKind::InvalidAnnotation,
                format!(
                    "Expected a type-level DSL function, got `{}`",
                    self.for_display(callee.clone())
                ),
            ),
        }
    }

    fn parse_user_defined_type_level_dsl_call(
        &self,
        call: &ExprCall,
        function: Arc<ValidatedTypeShapeDslFunction>,
        domain: TypeShapeDslDomain,
        type_form_context: TypeFormContext<'_>,
        errors: &ErrorCollector,
    ) -> Type {
        let name = function.name().as_str();
        if !call.arguments.keywords.is_empty() {
            return self.error(
                errors,
                call.range,
                ErrorKind::InvalidAnnotation,
                format!("`{name}` does not accept keyword arguments"),
            );
        }
        if call.arguments.args.len() != 1 {
            return self.error(
                errors,
                call.range,
                ErrorKind::InvalidAnnotation,
                format!(
                    "Expected 1 argument for `{name}`, got {}",
                    call.arguments.args.len()
                ),
            );
        }

        let arg_expr = &call.arguments.args[0];
        let argument_context = TypeFormContext::TypeArgument(&type_form_context);
        let arg = match domain {
            TypeShapeDslDomain::Int => {
                let dimension_errors = self.error_collector();
                let parsed_dimension = self
                    .parse_dimension_list(
                        slice::from_ref(arg_expr),
                        argument_context,
                        &dimension_errors,
                    )
                    .and_then(|dims| dims.into_iter().next())
                    .filter(|ty| !ty.is_error());
                if let Some(ty) = parsed_dimension {
                    errors.extend(dimension_errors);
                    ty
                } else {
                    let ordinary_errors = self.error_collector();
                    let ty = self.expr_untype(arg_expr, argument_context, &ordinary_errors);
                    if ty.is_error() {
                        errors.extend(dimension_errors);
                        Type::any_error()
                    } else {
                        errors.extend(ordinary_errors);
                        ty
                    }
                }
            }
            TypeShapeDslDomain::IntTuple => self.expr_untype(arg_expr, argument_context, errors),
        };
        if arg.is_error() {
            return arg;
        }
        if !self.is_type_shape_dsl_argument(&arg, domain) {
            return self.error(
                errors,
                arg_expr.range(),
                ErrorKind::InvalidAnnotation,
                format!(
                    "Expected an `{domain:?}` argument to `{name}`, got `{}`",
                    self.for_display(arg.clone())
                ),
            );
        }
        Type::TypeLevelDslCall(Box::new(TypeLevelDslCall::user_defined(
            function, domain, arg,
        )))
    }

    fn parse_broadcast_type_level_dsl_call(
        &self,
        call: &ExprCall,
        type_form_context: TypeFormContext<'_>,
        errors: &ErrorCollector,
    ) -> Type {
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

    fn is_int_tuple_dsl_argument(&self, ty: &Type) -> bool {
        let restriction = match ty {
            Type::Any(_) | Type::IntTuple(_) => return true,
            Type::TypeLevelDslCall(call) => {
                return call.result_domain() == TypeShapeDslDomain::IntTuple;
            }
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
    fn is_type_shape_dsl_argument(&self, ty: &Type, domain: TypeShapeDslDomain) -> bool {
        match domain {
            TypeShapeDslDomain::Int => match ty {
                Type::Any(_) => true,
                Type::Int(_) => true,
                Type::TypeLevelDslCall(call) => call.result_domain() == TypeShapeDslDomain::Int,
                Type::Quantified(q) => q.kind() == QuantifiedKind::IntVar,
                Type::TypeVar(type_var) => type_var.kind() == QuantifiedKind::IntVar,
                _ => false,
            },
            TypeShapeDslDomain::IntTuple => self.is_int_tuple_dsl_argument(ty),
        }
    }
}

fn type_shape_dsl_domain(ty: &Type) -> Option<TypeShapeDslDomain> {
    match ty {
        Type::Int(_) => Some(TypeShapeDslDomain::Int),
        Type::IntTuple(_) => Some(TypeShapeDslDomain::IntTuple),
        _ => None,
    }
}
