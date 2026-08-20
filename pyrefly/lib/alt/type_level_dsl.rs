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
use pyrefly_types::type_level_dsl::TypeShapeDslSignature;
use pyrefly_types::type_level_dsl::ValidatedTypeShapeDslFunction;
use pyrefly_types::type_var::Restriction;
use pyrefly_types::types::CalleeKind;
use pyrefly_types::types::Type;
use pyrefly_util::display::pluralize;
use ruff_python_ast::Expr;
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
        assert_eq!(
            params.len(),
            dsl.parameter_count(),
            "validated type-level DSL AST must align with resolved parameters"
        );
        let mut parameter_domains = Vec::with_capacity(params.len());
        let mut valid_parameters = true;
        for (index, parameter) in params.iter().enumerate() {
            let domain = dsl
                .has_parameter_annotation(index)
                .then(|| type_shape_dsl_domain(parameter.as_type()))
                .flatten();
            if let Some(domain) = domain {
                parameter_domains.push(domain);
            } else {
                valid_parameters = false;
                self.error(
                    errors,
                    dsl.parameter_annotation_range(index),
                    ErrorKind::InvalidArgument,
                    format!(
                        "`@type_shape_dsl_function` parameter `{}` must be annotated as `Int` or `IntTuple`",
                        dsl.parameter_name(index)
                    ),
                );
            }
        }
        let return_domain = if dsl.has_return_annotation() {
            type_shape_dsl_domain(return_type)
        } else {
            None
        };
        if return_domain.is_none() {
            self.error(
                errors,
                dsl.return_annotation_range(),
                ErrorKind::InvalidArgument,
                "`@type_shape_dsl_function` return must be annotated as `Int` or `IntTuple`"
                    .to_owned(),
            );
        }
        if valid_parameters && let Some(result) = return_domain {
            let returned_parameter = dsl.returned_parameter_index();
            if parameter_domains[returned_parameter] != result {
                self.error(
                    errors,
                    dsl.return_annotation_range(),
                    ErrorKind::InvalidArgument,
                    format!(
                        "`@type_shape_dsl_function` return annotation must match returned parameter `{}`",
                        dsl.parameter_name(returned_parameter)
                    ),
                );
            } else if let FunctionKind::Def(func_id) = function_kind {
                return Some(FunctionKind::TypeShapeDsl(
                    func_id.clone(),
                    Arc::new(TypeShapeDslSignature::new(parameter_domains, result)),
                    dsl.clone(),
                ));
            } else {
                self.error(
                    errors,
                    function_range,
                    ErrorKind::InvalidArgument,
                    "`@type_shape_dsl_function` must be applied to an ordinary function definition"
                        .to_owned(),
                );
            }
        }
        None
    }

    pub(crate) fn parse_type_level_dsl_call(
        &self,
        call: &ExprCall,
        callee: &Type,
        type_form_context: TypeFormContext<'_>,
        errors: &ErrorCollector,
    ) -> Type {
        match callee.callee_kind() {
            Some(CalleeKind::Function(FunctionKind::TypeShapeDsl(_, signature, function))) => self
                .parse_user_defined_type_level_dsl_call(
                    call,
                    function,
                    signature,
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
        signature: Arc<TypeShapeDslSignature>,
        type_form_context: TypeFormContext<'_>,
        errors: &ErrorCollector,
    ) -> Type {
        let name = function.name().as_str();
        if let Some(keyword) = call
            .arguments
            .keywords
            .iter()
            .find(|keyword| keyword.arg.is_none())
        {
            return self.error(
                errors,
                keyword.range(),
                ErrorKind::InvalidAnnotation,
                format!("`{name}` does not accept starred keyword arguments"),
            );
        }
        if !call.arguments.keywords.is_empty() {
            return self.error(
                errors,
                call.range,
                ErrorKind::InvalidAnnotation,
                format!("`{name}` does not accept keyword arguments"),
            );
        }
        if let Some(arg) = call
            .arguments
            .args
            .iter()
            .find(|arg| matches!(arg, Expr::Starred(_)))
        {
            return self.error(
                errors,
                arg.range(),
                ErrorKind::InvalidAnnotation,
                format!("`{name}` does not accept starred arguments"),
            );
        }
        let parameter_domains = signature.parameter_domains();
        if call.arguments.args.len() != parameter_domains.len() {
            return self.error(
                errors,
                call.range,
                ErrorKind::InvalidAnnotation,
                format!(
                    "Expected {} {} for `{name}`, got {}",
                    parameter_domains.len(),
                    pluralize(parameter_domains.len(), "argument"),
                    call.arguments.args.len()
                ),
            );
        }

        let argument_context = TypeFormContext::TypeArgument(&type_form_context);
        let mut args = Vec::with_capacity(call.arguments.args.len());
        for (index, (arg_expr, domain)) in call
            .arguments
            .args
            .iter()
            .zip(parameter_domains)
            .enumerate()
        {
            let arg =
                self.parse_type_shape_dsl_argument(arg_expr, *domain, argument_context, errors);
            if arg.is_error() {
                return arg;
            }
            if !self.is_type_shape_dsl_argument(&arg, *domain) {
                return self.error(
                    errors,
                    arg_expr.range(),
                    ErrorKind::InvalidAnnotation,
                    format!(
                        "Expected an `{domain:?}` argument for parameter `{}` (position {}) of `{name}`, got `{}`",
                        function.parameter_name(index),
                        index + 1,
                        self.for_display(arg.clone())
                    ),
                );
            }
            args.push(arg);
        }
        Type::TypeLevelDslCall(Box::new(TypeLevelDslCall::user_defined(
            function, signature, args,
        )))
    }

    fn parse_type_shape_dsl_argument(
        &self,
        arg: &Expr,
        domain: TypeShapeDslDomain,
        type_form_context: TypeFormContext<'_>,
        errors: &ErrorCollector,
    ) -> Type {
        match domain {
            TypeShapeDslDomain::Int => {
                let dimension_errors = self.error_collector();
                let parsed_dimension = self
                    .parse_dimension_list(
                        slice::from_ref(arg),
                        type_form_context,
                        &dimension_errors,
                    )
                    .and_then(|dims| dims.into_iter().next())
                    .filter(|ty| !ty.is_error());
                if let Some(ty) = parsed_dimension {
                    errors.extend(dimension_errors);
                    ty
                } else {
                    let ordinary_errors = self.error_collector();
                    let ty = self.expr_untype(arg, type_form_context, &ordinary_errors);
                    if ty.is_error() {
                        errors.extend(dimension_errors);
                        Type::any_error()
                    } else {
                        errors.extend(ordinary_errors);
                        ty
                    }
                }
            }
            TypeShapeDslDomain::IntTuple => self.expr_untype(arg, type_form_context, errors),
        }
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
