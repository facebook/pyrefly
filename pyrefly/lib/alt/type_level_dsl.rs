/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::slice;
use std::sync::Arc;

use pyrefly_types::callable::Param;
use pyrefly_types::function::FuncDefId;
use pyrefly_types::function::FunctionKind;
use pyrefly_types::quantified::QuantifiedKind;
use pyrefly_types::type_level_dsl::ParsedTypeShapeDslFunction;
use pyrefly_types::type_level_dsl::ResolvedTypeShapeDslFunction;
use pyrefly_types::type_level_dsl::TypeLevelDslCall;
use pyrefly_types::type_level_dsl::TypeShapeDslConditionKind;
use pyrefly_types::type_level_dsl::TypeShapeDslDomain;
use pyrefly_types::type_level_dsl::TypeShapeDslExpressionKind;
use pyrefly_types::type_level_dsl::TypeShapeDslFlagIntComparisonOp;
use pyrefly_types::type_level_dsl::TypeShapeDslFlagValueKind;
use pyrefly_types::type_level_dsl::TypeShapeDslInputDomain;
use pyrefly_types::type_level_dsl::TypeShapeDslIntrinsic;
use pyrefly_types::type_level_dsl::TypeShapeDslProgramError;
use pyrefly_types::type_level_dsl::TypeShapeDslReturnKind;
use pyrefly_types::type_level_dsl::ValidatedTypeShapeDslFunction;
use pyrefly_types::type_var::FlagDomain;
use pyrefly_types::type_var::FlagMember;
use pyrefly_types::type_var::Restriction;
use pyrefly_types::types::CalleeKind;
use pyrefly_types::types::Type;
use pyrefly_util::display::pluralize;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprCall;
use ruff_python_ast::UnaryOp;
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

/// The widest `Flag` domain the DSL's Flag operations can inspect.
fn type_shape_dsl_flag_domain() -> FlagDomain {
    FlagDomain::of(FlagMember::Int)
        .join(FlagDomain::of(FlagMember::Tuple))
        .join(FlagDomain::of(FlagMember::NoneType))
}

impl<'a, Ans: LookupAnswer> AnswersSolver<'a, Ans> {
    /// Resolves the shape-specific helper calls retained during syntax validation.
    ///
    /// `FunctionKind` is the narrow boundary with Pyrefly's ordinary function model. Resolving
    /// callees here gives imports and aliases normal name-resolution semantics while keeping the
    /// helper graph and evaluator entirely in the shape DSL representation.
    fn resolve_type_shape_dsl_function(
        &self,
        func_id: &Arc<FuncDefId>,
        definition: Arc<ValidatedTypeShapeDslFunction>,
        parameter_domains: Vec<TypeShapeDslInputDomain>,
        result_domain: TypeShapeDslDomain,
        function_range: TextRange,
        errors: &ErrorCollector,
    ) -> Option<ResolvedTypeShapeDslFunction> {
        let mut helpers = Vec::new();
        let mut valid = true;
        let swallowed_errors = self.error_swallower();
        for helper_call in definition.helper_calls() {
            let callee = self.expr_infer(helper_call.callee(), &swallowed_errors);
            match callee.callee_kind() {
                Some(CalleeKind::Function(FunctionKind::TypeShapeDsl(helper_id, helper))) => {
                    let argument_domains = helper_call.argument_domains(&parameter_domains);
                    if argument_domains.as_deref() != Some(helper.parameter_domains()) {
                        self.error(
                            errors,
                            helper_call.callee().range(),
                            ErrorKind::InvalidArgument,
                            format!(
                                "DSL helper argument domains must exactly match `{}`",
                                helper.name()
                            ),
                        );
                        valid = false;
                        continue;
                    }
                    if helper.result_domain() != result_domain {
                        self.error(
                            errors,
                            helper_call.callee().range(),
                            ErrorKind::InvalidArgument,
                            "DSL helper result domain must match the caller result domain"
                                .to_owned(),
                        );
                        valid = false;
                        continue;
                    }
                    if func_id.as_ref() == helper_id.as_ref()
                        || helper.contains_function(func_id.as_ref())
                    {
                        self.error(
                            errors,
                            helper_call.callee().range(),
                            ErrorKind::InvalidArgument,
                            TypeShapeDslProgramError::Cycle.message().to_owned(),
                        );
                        valid = false;
                        continue;
                    }
                    helpers.push((helper_id.clone(), helper.clone()));
                }
                Some(CalleeKind::Function(FunctionKind::Def(callee_id)))
                    if func_id.as_ref() == callee_id.as_ref() =>
                {
                    self.error(
                        errors,
                        helper_call.callee().range(),
                        ErrorKind::InvalidArgument,
                        TypeShapeDslProgramError::Cycle.message().to_owned(),
                    );
                    valid = false;
                }
                _ => {
                    self.error(
                        errors,
                        helper_call.callee().range(),
                        ErrorKind::InvalidArgument,
                        "@type_shape_dsl_function return value must be a bare parameter name or validated DSL helper call; DSL helper callee must be a validated `@type_shape_dsl_function`".to_owned(),
                    );
                    valid = false;
                }
            }
        }
        if !valid {
            return None;
        }
        match ResolvedTypeShapeDslFunction::try_new(
            func_id.clone(),
            definition,
            parameter_domains,
            result_domain,
            helpers,
        ) {
            Ok(function) => Some(function),
            Err(error) => {
                self.error(
                    errors,
                    function_range,
                    ErrorKind::InvalidArgument,
                    error.message().to_owned(),
                );
                None
            }
        }
    }

    /// Validates resolved DSL annotations, emitting diagnostics and metadata only on success.
    pub(super) fn validate_type_shape_dsl_declaration(
        &self,
        dsl: &Arc<ParsedTypeShapeDslFunction>,
        params: &[Param],
        return_type: &Type,
        function_kind: &FunctionKind,
        function_range: TextRange,
        errors: &ErrorCollector,
    ) -> Option<FunctionKind> {
        let validated = match dsl.validate(|expr| self.resolve_type_shape_dsl_intrinsic(expr)) {
            Ok(validated) => Arc::new(validated),
            Err(error) => {
                self.error(
                    errors,
                    error.range,
                    ErrorKind::InvalidArgument,
                    format!("@type_shape_dsl_function {}", error.message),
                );
                return None;
            }
        };
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
                .then(|| type_shape_dsl_input_domain(parameter.as_type()))
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
                        "`@type_shape_dsl_function` parameter `{}` must be annotated as `Int`, `IntTuple`, `int`, `bool`, or `str`",
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
            let flag_value = dsl.has_return_annotation()
                && type_shape_dsl_input_domain(return_type)
                    .is_some_and(|domain| matches!(domain, TypeShapeDslInputDomain::Flag(_)));
            self.error(
                errors,
                dsl.return_annotation_range(),
                ErrorKind::InvalidArgument,
                if flag_value {
                    "`@type_shape_dsl_function` Flag values are input-only; return must be annotated as `Int` or `IntTuple`"
                        .to_owned()
                } else {
                    "`@type_shape_dsl_function` return must be annotated as `Int` or `IntTuple`"
                        .to_owned()
                },
            );
        }
        if valid_parameters && let Some(result) = return_domain {
            let mut valid_body = true;
            for condition in validated.conditions() {
                let invalid_domain = match condition.kind() {
                    TypeShapeDslConditionKind::SlotCompare {
                        left_parameters,
                        right_parameters,
                        op,
                        ..
                    } => {
                        let mut domains = left_parameters
                            .iter()
                            .chain(right_parameters)
                            .map(|parameter| parameter_domains[*parameter]);
                        let Some(first) = domains.next() else {
                            unreachable!("slot comparison must retain parameter origins")
                        };
                        let same_domain = domains.all(|domain| domain == first);
                        if same_domain
                            && first
                                == TypeShapeDslInputDomain::Value(TypeShapeDslDomain::Int)
                        {
                            (!matches!(
                                op,
                                TypeShapeDslFlagIntComparisonOp::Equal
                                    | TypeShapeDslFlagIntComparisonOp::LessThan
                            ))
                            .then_some("`@type_shape_dsl_function` `Int` comparisons support only `==` and `<`")
                        } else if same_domain
                            && first
                                == TypeShapeDslInputDomain::Flag(FlagDomain::of(FlagMember::Int))
                        {
                            None
                        } else {
                            Some("`@type_shape_dsl_function` comparison operands must both be annotated as `Int` or both be `Flag[int]`")
                        }
                    }
                    TypeShapeDslConditionKind::Any { .. }
                    | TypeShapeDslConditionKind::GeneratorElementSelfCompare(_) => None,
                    TypeShapeDslConditionKind::IsConcreteInt {
                        parameter_origins,
                        ..
                    } => parameter_origins.as_deref().and_then(|parameters| {
                        (!parameters.iter().all(|parameter| {
                            parameter_domains[*parameter]
                                == TypeShapeDslInputDomain::Value(TypeShapeDslDomain::Int)
                        }))
                            .then_some("`@type_shape_dsl_function` condition operands must be annotated as `Int`")
                    }),
                    TypeShapeDslConditionKind::IsIntValue {
                        parameter_origins, ..
                    }
                    | TypeShapeDslConditionKind::IsNone {
                        parameter_origins, ..
                    } => parameter_origins.as_deref().and_then(|parameters| {
                        (!parameters.iter().all(|parameter| matches!(
                            parameter_domains[*parameter],
                            TypeShapeDslInputDomain::Flag(domain)
                                if domain.is_subset_of(type_shape_dsl_flag_domain())
                        )))
                        .then_some("`@type_shape_dsl_function` control-flow narrowing requires a Flag[int | tuple[int, ...] | None] value")
                    }),
                    TypeShapeDslConditionKind::FlagIntCompare(_)
                    | TypeShapeDslConditionKind::Membership { .. } => None,
                };
                if let Some(message) = invalid_domain {
                    self.error(
                        errors,
                        condition.range(),
                        ErrorKind::InvalidArgument,
                        message.to_owned(),
                    );
                    valid_body = false;
                }
            }
            for expression in validated.expressions() {
                let invalid_domain = match expression.kind() {
                    TypeShapeDslExpressionKind::DimensionSlot {
                        parameter_origins: Some(parameters),
                        ..
                    } => (!parameters.iter().all(|parameter| {
                        parameter_domains[*parameter]
                            == TypeShapeDslInputDomain::Value(TypeShapeDslDomain::Int)
                    }))
                    .then_some("`@type_shape_dsl_function` IntTuple elements must be annotated as `Int`"),
                    TypeShapeDslExpressionKind::IntTupleIndex {
                        parameter_origins: Some(shapes),
                        ..
                    } => (!shapes.iter().all(|shape| {
                        parameter_domains[*shape]
                            == TypeShapeDslInputDomain::Value(TypeShapeDslDomain::IntTuple)
                    }))
                    .then_some("`@type_shape_dsl_function` len and indexing require an `IntTuple` parameter"),
                    TypeShapeDslExpressionKind::IntTupleLength {
                        parameter_origins: Some(shapes),
                        ..
                    } => {
                        if shapes.iter().all(|shape| {
                            parameter_domains[*shape]
                                == TypeShapeDslInputDomain::Value(TypeShapeDslDomain::IntTuple)
                        }) {
                            None
                        } else if shapes.iter().any(|shape| {
                            matches!(
                                parameter_domains[*shape],
                                TypeShapeDslInputDomain::Flag(_)
                            )
                        }) {
                            Some("`@type_shape_dsl_function` `len` of a Flag value requires control-flow narrowing to a sequence")
                        } else {
                            Some("`@type_shape_dsl_function` len and indexing require an `IntTuple` parameter")
                        }
                    }
                    TypeShapeDslExpressionKind::GeneratorSourceSlot {
                        parameter_origins: Some(parameters),
                        narrowed,
                        ..
                    } => {
                        let valid = parameters.iter().all(|parameter| {
                            matches!(
                                parameter_domains[*parameter],
                                TypeShapeDslInputDomain::Value(TypeShapeDslDomain::IntTuple)
                            ) || match parameter_domains[*parameter] {
                                TypeShapeDslInputDomain::Flag(domain) if *narrowed => {
                                    domain.is_subset_of(type_shape_dsl_flag_domain())
                                }
                                TypeShapeDslInputDomain::Flag(domain) => {
                                    domain == FlagDomain::of(FlagMember::Tuple)
                                }
                                _ => false,
                            }
                        });
                        (!valid).then_some(
                            "`@type_shape_dsl_function` generator source must be an `IntTuple` or Flag sequence",
                        )
                    }
                    TypeShapeDslExpressionKind::IntTupleConstructor => {
                        (result != TypeShapeDslDomain::IntTuple).then_some(
                            "`@type_shape_dsl_function` dsl.IntTuple requires an `IntTuple` result",
                        )
                    }
                    TypeShapeDslExpressionKind::FlagValueSlot {
                        parameter_origins: Some(parameters),
                        required,
                        narrowed,
                        ..
                    } => {
                        let valid = parameters.iter().all(|parameter| {
                            match parameter_domains[*parameter] {
                                TypeShapeDslInputDomain::Flag(domain) if *narrowed => {
                                    domain.is_subset_of(type_shape_dsl_flag_domain())
                                }
                                TypeShapeDslInputDomain::Flag(domain) => match required {
                                    TypeShapeDslFlagValueKind::Int => {
                                        domain == FlagDomain::of(FlagMember::Int)
                                    }
                                    TypeShapeDslFlagValueKind::Sequence => {
                                        domain == FlagDomain::of(FlagMember::Tuple)
                                    }
                                },
                                _ => false,
                            }
                        });
                        (!valid).then_some("`@type_shape_dsl_function` Flag operation requires a compatible Flag parameter")
                    }
                    TypeShapeDslExpressionKind::DimensionLiteral(_)
                    | TypeShapeDslExpressionKind::DimensionSlot { .. }
                    | TypeShapeDslExpressionKind::IntTupleIndex { .. }
                    | TypeShapeDslExpressionKind::IntTupleLength { .. }
                    | TypeShapeDslExpressionKind::GeneratorSourceSlot { .. }
                    | TypeShapeDslExpressionKind::GeneratorElementAsDimension(_)
                    | TypeShapeDslExpressionKind::GeneratorElementAsFlagInt(_)
                    | TypeShapeDslExpressionKind::Slot(_)
                    | TypeShapeDslExpressionKind::FlagValueSlot { .. }
                    | TypeShapeDslExpressionKind::FlagIntLiteral(_)
                    | TypeShapeDslExpressionKind::FlagNone
                    | TypeShapeDslExpressionKind::FlagTuple
                    | TypeShapeDslExpressionKind::FlagRange
                    | TypeShapeDslExpressionKind::FlagSequenceLength
                    | TypeShapeDslExpressionKind::FlagSequenceCount
                    | TypeShapeDslExpressionKind::FlagIntArithmetic(_)
                    | TypeShapeDslExpressionKind::DimensionTuple
                    | TypeShapeDslExpressionKind::Conditional
                    | TypeShapeDslExpressionKind::DimensionGenerator { .. }
                    | TypeShapeDslExpressionKind::FlagGenerator { .. } => None,
                };
                if let Some(message) = invalid_domain {
                    self.error(
                        errors,
                        expression.range(),
                        ErrorKind::InvalidArgument,
                        message.to_owned(),
                    );
                    valid_body = false;
                }
            }
            for return_ in validated.returns() {
                match return_.kind() {
                    TypeShapeDslReturnKind::Local { domain, .. } if *domain != result => {
                        self.error(errors, return_.range(), ErrorKind::InvalidArgument, "`@type_shape_dsl_function` local return domain must match the declared result".to_owned());
                        valid_body = false;
                    }
                    TypeShapeDslReturnKind::AliasedParameter { parameters, .. }
                        if !parameters.iter().all(|parameter| {
                            parameter_domains[*parameter] == TypeShapeDslInputDomain::Value(result)
                        }) =>
                    {
                        self.error(errors, return_.range(), ErrorKind::InvalidArgument, "`@type_shape_dsl_function` local alias return domain must match the declared result".to_owned());
                        valid_body = false;
                    }
                    TypeShapeDslReturnKind::Parameter(index)
                        if parameter_domains[*index] != TypeShapeDslInputDomain::Value(result) =>
                    {
                        let flag_value =
                            matches!(parameter_domains[*index], TypeShapeDslInputDomain::Flag(_));
                        self.error(
                            errors,
                            return_.range(),
                            ErrorKind::InvalidArgument,
                            if flag_value {
                                format!(
                                    "`@type_shape_dsl_function` Flag parameter `{}` is input-only and cannot be returned",
                                    dsl.parameter_name(*index)
                                )
                            } else {
                                format!(
                                    "`@type_shape_dsl_function` return annotation must match returned parameter `{}`",
                                    dsl.parameter_name(*index)
                                )
                            },
                        );
                        valid_body = false;
                    }
                    TypeShapeDslReturnKind::Broadcast {
                        left_parameters,
                        right_parameters,
                        ..
                    } if result != TypeShapeDslDomain::IntTuple
                        || !left_parameters.iter().all(|parameter| {
                            parameter_domains[*parameter]
                                == TypeShapeDslInputDomain::Value(TypeShapeDslDomain::IntTuple)
                        })
                        || !right_parameters.iter().all(|parameter| {
                            parameter_domains[*parameter]
                                == TypeShapeDslInputDomain::Value(TypeShapeDslDomain::IntTuple)
                        }) =>
                    {
                        self.error(
                            errors,
                            return_.range(),
                            ErrorKind::InvalidArgument,
                            "`@type_shape_dsl_function` broadcast return requires two `IntTuple` parameters and an `IntTuple` result"
                                .to_owned(),
                        );
                        valid_body = false;
                    }
                    TypeShapeDslReturnKind::IntFlagArithmetic { left, right, .. }
                        if result != TypeShapeDslDomain::Int
                            || parameter_domains[*left]
                                != TypeShapeDslInputDomain::Value(TypeShapeDslDomain::Int)
                            || parameter_domains[*right]
                                != TypeShapeDslInputDomain::Flag(FlagDomain::of(
                                    FlagMember::Int,
                                )) =>
                    {
                        self.error(
                            errors,
                            return_.range(),
                            ErrorKind::InvalidArgument,
                            "`@type_shape_dsl_function` arithmetic return requires `Int +/- Flag[int]` and an `Int` result"
                                .to_owned(),
                        );
                        valid_body = false;
                    }
                    TypeShapeDslReturnKind::Gradual(domain) if *domain != result => {
                        self.error(
                            errors,
                            return_.range(),
                            ErrorKind::InvalidArgument,
                            format!(
                                "`@type_shape_dsl_function` declares return domain `{}`, but `shape_extensions.dsl.{}.gradual()` returns `{}`",
                                result.as_str(),
                                domain.as_str(),
                                domain.as_str(),
                            ),
                        );
                        valid_body = false;
                    }
                    TypeShapeDslReturnKind::Parameter(_)
                    | TypeShapeDslReturnKind::Local { .. }
                    | TypeShapeDslReturnKind::AliasedParameter { .. }
                    | TypeShapeDslReturnKind::Broadcast { .. }
                    | TypeShapeDslReturnKind::IntFlagArithmetic { .. }
                    | TypeShapeDslReturnKind::Expression
                    | TypeShapeDslReturnKind::Invalid
                    | TypeShapeDslReturnKind::HelperCall(_)
                    | TypeShapeDslReturnKind::Gradual(_) => {}
                }
            }
            if valid_body && let FunctionKind::Def(func_id) = function_kind {
                if let Some(function) = self.resolve_type_shape_dsl_function(
                    func_id,
                    validated,
                    parameter_domains,
                    result,
                    function_range,
                    errors,
                ) {
                    return Some(FunctionKind::TypeShapeDsl(
                        func_id.clone(),
                        Arc::new(function),
                    ));
                }
            } else if valid_body {
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

    /// Recognize a DSL intrinsic by resolved callable identity, so imports, aliases, and
    /// reexports work while unrelated same-spelling functions do not.
    fn resolve_type_shape_dsl_intrinsic(&self, expr: &Expr) -> Option<TypeShapeDslIntrinsic> {
        let callee = self.expr_infer(expr, &self.error_swallower());
        if let Type::ClassDef(class) = &callee
            && class.qname().module_name().as_str() == "shape_extensions.dsl"
            && class.qname().id().as_str() == "IntTuple"
        {
            return Some(TypeShapeDslIntrinsic::IntTuple);
        }
        if let Type::ClassDef(class) = &callee
            && class.qname().module_name().as_str() == "builtins"
            && class.qname().id().as_str() == "range"
        {
            return Some(TypeShapeDslIntrinsic::Range);
        }
        if let Type::ClassDef(class) = &callee
            && class.qname().module_name().as_str() == "builtins"
            && class.qname().id().as_str() == "tuple"
        {
            return Some(TypeShapeDslIntrinsic::Tuple);
        }
        let Some(CalleeKind::Function(function_kind)) = callee.callee_kind() else {
            return None;
        };
        if function_kind == FunctionKind::Len {
            return Some(TypeShapeDslIntrinsic::Len);
        }
        let FunctionKind::Def(id) = function_kind else {
            return None;
        };
        if id.has_toplevel_qname("shape_extensions", "broadcast") {
            return Some(TypeShapeDslIntrinsic::Broadcast);
        }
        if id.has_toplevel_qname("builtins", "any") {
            return Some(TypeShapeDslIntrinsic::Any);
        }
        if id.has_toplevel_qname("shape_extensions.dsl", "is_concrete_int") {
            return Some(TypeShapeDslIntrinsic::IsConcreteInt);
        }
        if id.has_toplevel_qname("shape_extensions.dsl", "is_int_value") {
            return Some(TypeShapeDslIntrinsic::IsIntValue);
        }
        if id.has_toplevel_qname("shape_extensions.dsl", "Invalid") {
            return Some(TypeShapeDslIntrinsic::Invalid);
        }
        if id.qname.module_name().as_str() != "shape_extensions.dsl" {
            return None;
        }
        let class = id.cls.as_ref()?;
        if id.qname.id().as_str() != "gradual" {
            return None;
        }
        if class.has_toplevel_qname("shape_extensions.dsl", "Int") {
            Some(TypeShapeDslIntrinsic::Gradual(TypeShapeDslDomain::Int))
        } else if class.has_toplevel_qname("shape_extensions.dsl", "IntTuple") {
            Some(TypeShapeDslIntrinsic::Gradual(TypeShapeDslDomain::IntTuple))
        } else {
            None
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
            Some(CalleeKind::Function(FunctionKind::TypeShapeDsl(_, function))) => self
                .parse_user_defined_type_level_dsl_call(call, function, type_form_context, errors),
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
        function: Arc<ResolvedTypeShapeDslFunction>,
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
        let parameter_domains = function.parameter_domains();
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
                let article = match domain {
                    TypeShapeDslInputDomain::Flag(_) => "a",
                    TypeShapeDslInputDomain::Value(_) => "an",
                };
                return self.error(
                    errors,
                    arg_expr.range(),
                    ErrorKind::InvalidAnnotation,
                    format!(
                        "Expected {article} `{domain}` argument for parameter `{}` (position {}) of `{name}`, got `{}`",
                        function.parameter_name(index),
                        index + 1,
                        self.for_display(arg.clone())
                    ),
                );
            }
            args.push(arg);
        }
        Type::TypeLevelDslCall(Box::new(TypeLevelDslCall::user_defined(function, args)))
    }

    fn parse_type_shape_dsl_argument(
        &self,
        arg: &Expr,
        domain: TypeShapeDslInputDomain,
        type_form_context: TypeFormContext<'_>,
        errors: &ErrorCollector,
    ) -> Type {
        match domain {
            TypeShapeDslInputDomain::Value(TypeShapeDslDomain::Int) => {
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
            TypeShapeDslInputDomain::Value(TypeShapeDslDomain::IntTuple) => {
                self.expr_untype(arg, type_form_context, errors)
            }
            TypeShapeDslInputDomain::Flag(_) => match arg {
                Expr::NumberLiteral(_) | Expr::BooleanLiteral(_) | Expr::StringLiteral(_) => {
                    self.expr_infer(arg, errors)
                }
                Expr::UnaryOp(unary)
                    if unary.op == UnaryOp::USub
                        && matches!(unary.operand.as_ref(), Expr::NumberLiteral(_)) =>
                {
                    self.expr_infer(arg, errors)
                }
                _ => self.expr_untype(arg, type_form_context, errors),
            },
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
            Restriction::Flag(_) => false,
            Restriction::Unrestricted => false,
        }
    }
    fn is_type_shape_dsl_argument(&self, ty: &Type, domain: TypeShapeDslInputDomain) -> bool {
        match domain {
            TypeShapeDslInputDomain::Value(TypeShapeDslDomain::Int) => match ty {
                Type::Any(_) => true,
                Type::Int(_) => true,
                Type::TypeLevelDslCall(call) => call.result_domain() == TypeShapeDslDomain::Int,
                Type::Quantified(q) => q.kind() == QuantifiedKind::IntVar,
                Type::TypeVar(type_var) => type_var.kind() == QuantifiedKind::IntVar,
                _ => false,
            },
            TypeShapeDslInputDomain::Value(TypeShapeDslDomain::IntTuple) => {
                self.is_int_tuple_dsl_argument(ty)
            }
            TypeShapeDslInputDomain::Flag(domain) => domain.accepts(ty),
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

fn type_shape_dsl_input_domain(ty: &Type) -> Option<TypeShapeDslInputDomain> {
    type_shape_dsl_domain(ty)
        .map(TypeShapeDslInputDomain::Value)
        .or_else(|| FlagDomain::from_type(ty).map(TypeShapeDslInputDomain::Flag))
}
