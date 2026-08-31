/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;
use std::slice;
use std::sync::Arc;

use pyrefly_types::callable::Param;
use pyrefly_types::dimension::is_gradual_size_bound_type_var;
use pyrefly_types::dimension::is_optional_int;
use pyrefly_types::dimension::is_optional_int_bound_type_var;
use pyrefly_types::function::FuncDefId;
use pyrefly_types::function::FunctionKind;
use pyrefly_types::quantified::QuantifiedKind;
use pyrefly_types::type_level_dsl::ParsedTypeShapeDslFunction;
use pyrefly_types::type_level_dsl::ResolvedTypeShapeDslFunction;
use pyrefly_types::type_level_dsl::StructurallyValidatedTypeShapeDslFunction;
use pyrefly_types::type_level_dsl::TypeLevelDslCall;
use pyrefly_types::type_level_dsl::TypeShapeDslComparisonOp;
use pyrefly_types::type_level_dsl::TypeShapeDslComparisonOperand;
use pyrefly_types::type_level_dsl::TypeShapeDslConditionKind;
use pyrefly_types::type_level_dsl::TypeShapeDslDomain;
use pyrefly_types::type_level_dsl::TypeShapeDslExpressionKind;
use pyrefly_types::type_level_dsl::TypeShapeDslFlagValueKind;
use pyrefly_types::type_level_dsl::TypeShapeDslHelperArgumentError;
use pyrefly_types::type_level_dsl::TypeShapeDslInputDomain;
use pyrefly_types::type_level_dsl::TypeShapeDslIntrinsic;
use pyrefly_types::type_level_dsl::TypeShapeDslParameterNarrowing;
use pyrefly_types::type_level_dsl::TypeShapeDslProgramError;
use pyrefly_types::type_level_dsl::TypeShapeDslReturnKind;
use pyrefly_types::type_level_dsl::TypeShapeDslSlotReturnKind;
use pyrefly_types::type_var::FlagDomain;
use pyrefly_types::type_var::FlagMember;
use pyrefly_types::type_var::Restriction;
use pyrefly_types::types::AnyStyle;
use pyrefly_types::types::CalleeKind;
use pyrefly_types::types::Type;
use pyrefly_util::display::pluralize;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprCall;
use ruff_python_ast::Operator;
use ruff_python_ast::UnaryOp;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;

use crate::alt::answers::LookupAnswer;
use crate::alt::answers_solver::AnswersSolver;
use crate::alt::expr::DimensionExprError;
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

/// The domain accepted by scalar and sequence narrowing operations. Boolean conditions are
/// validated separately against `Flag[bool]`.
fn type_shape_dsl_narrowable_flag_domain() -> FlagDomain {
    FlagDomain::of(FlagMember::Int)
        .join(FlagDomain::of(FlagMember::IntTuple))
        .join(FlagDomain::of(FlagMember::NoneType))
}

fn type_shape_dsl_representable_flag_domain() -> FlagDomain {
    type_shape_dsl_narrowable_flag_domain()
        .join(FlagDomain::of(FlagMember::Bool))
        .join(FlagDomain::of(FlagMember::Str))
}

fn type_shape_dsl_optional_flag_domain(member: FlagMember) -> FlagDomain {
    FlagDomain::of(member).join(FlagDomain::of(FlagMember::NoneType))
}

#[derive(Clone, Copy)]
enum TypeShapeDslComparisonDomain {
    Dimension,
    FlagInt,
    FlagString,
}

fn type_shape_dsl_comparison_domain(
    left: &TypeShapeDslComparisonOperand,
    right: &TypeShapeDslComparisonOperand,
    parameter_domains: &[TypeShapeDslInputDomain],
) -> Option<TypeShapeDslComparisonDomain> {
    let operands = [left, right];
    if !operands.iter().all(|operand| {
        (operand.parameter_uses.is_none() && !operand.is_flag_operand)
            || operand
                .parameter_uses
                .as_deref()
                .is_some_and(|uses| !uses.is_empty())
            || operand.non_parameter_flag_domain.is_some()
    }) {
        unreachable!("validated comparison operands have a known value or a parameter source")
    }
    let supports_dimension = |operand: &TypeShapeDslComparisonOperand| {
        let uses = operand.parameter_uses.as_deref();
        uses.is_none() && !operand.is_flag_operand
            || uses.is_some_and(|uses| {
                uses.iter().all(|use_| {
                    parameter_domains[use_.parameter()]
                        .can_use_as(TypeShapeDslDomain::Int, use_.narrowing())
                }) && operand
                    .non_parameter_flag_domain
                    .is_none_or(|domain| domain == FlagDomain::of(FlagMember::Int))
            })
    };
    let supports_flag_int = |operand: &TypeShapeDslComparisonOperand| {
        let uses = operand.parameter_uses.as_deref();
        (operand.is_flag_operand || uses.is_some_and(|uses| !uses.is_empty()))
            && operand
                .non_parameter_flag_domain
                .is_none_or(|domain| domain == FlagDomain::of(FlagMember::Int))
            && uses.is_some_and(|uses| {
                uses.iter().all(|use_| {
                    matches!(parameter_domains[use_.parameter()], TypeShapeDslInputDomain::Flag(domain)
                        if (domain == FlagDomain::of(FlagMember::Int)
                            && use_.narrowing() == TypeShapeDslParameterNarrowing::Unnarrowed)
                            || (use_.narrowing() == TypeShapeDslParameterNarrowing::Integer
                                && domain.contains(FlagMember::Int)
                                && domain.is_subset_of(type_shape_dsl_narrowable_flag_domain())))
                })
            })
    };
    let supports_flag_string = |operand: &TypeShapeDslComparisonOperand| {
        let uses = operand.parameter_uses.as_deref();
        (operand.is_flag_operand || uses.is_some_and(|uses| !uses.is_empty()))
            && operand
                .non_parameter_flag_domain
                .is_none_or(|domain| {
            domain.contains(FlagMember::Str)
                && domain.is_subset_of(type_shape_dsl_optional_flag_domain(FlagMember::Str))
        }) && uses.is_some_and(|uses| {
            uses.iter().all(|use_| {
                matches!(parameter_domains[use_.parameter()], TypeShapeDslInputDomain::Flag(domain)
                    if domain.contains(FlagMember::Str)
                        && domain.is_subset_of(type_shape_dsl_optional_flag_domain(FlagMember::Str)))
            })
        })
    };
    if operands.into_iter().all(supports_flag_int) {
        Some(TypeShapeDslComparisonDomain::FlagInt)
    } else if operands.into_iter().all(supports_dimension) {
        Some(TypeShapeDslComparisonDomain::Dimension)
    } else if operands.into_iter().all(supports_flag_string) {
        Some(TypeShapeDslComparisonDomain::FlagString)
    } else {
        None
    }
}

#[derive(Clone, Copy)]
struct TypeShapeDslArgumentContext<'a> {
    function_name: &'a str,
    parameter_name: &'a str,
    position: usize,
}

impl<'ctx, 'answer, Ans: LookupAnswer> AnswersSolver<'ctx, 'answer, Ans> {
    /// Resolves helper calls and rebuilds metadata affected by their selected argument domains.
    ///
    /// Resolving callees through the ordinary function model gives imports and aliases normal
    /// name-resolution semantics while keeping the helper graph and evaluator entirely in the
    /// shape DSL representation.
    fn resolve_and_finalize_type_shape_dsl_helpers(
        &self,
        func_id: &Arc<FuncDefId>,
        definition: Arc<StructurallyValidatedTypeShapeDslFunction>,
        parameter_domains: &[TypeShapeDslInputDomain],
        result_domain: TypeShapeDslDomain,
        errors: &ErrorCollector,
    ) -> Option<(
        Arc<StructurallyValidatedTypeShapeDslFunction>,
        Vec<(Arc<FuncDefId>, Arc<ResolvedTypeShapeDslFunction>)>,
    )> {
        let mut helpers = Vec::new();
        let mut helper_argument_domains = Vec::new();
        let mut deferred_integer_domains = HashMap::new();
        let mut valid = true;
        let swallowed_errors = self.error_swallower();
        for helper_call in definition.helper_calls() {
            let callee = self.expr_infer(helper_call.callee(), &swallowed_errors);
            match callee.callee_kind() {
                Some(CalleeKind::Function(FunctionKind::TypeShapeDsl(helper_id, helper))) => {
                    let argument_domains = match helper_call.argument_domains(
                        parameter_domains,
                        helper.parameter_domains(),
                        &mut deferred_integer_domains,
                    ) {
                        Ok(domains) => domains,
                        Err(error) => {
                            let detail = match error {
                                TypeShapeDslHelperArgumentError::Arity => String::new(),
                                TypeShapeDslHelperArgumentError::IncompatibleDomain {
                                    argument,
                                    actual,
                                    expected,
                                } => {
                                    format!(
                                        ": argument {} has domain `{actual}`, expected `{expected}`",
                                        argument + 1,
                                    )
                                }
                            };
                            self.error(
                                errors,
                                helper_call.callee().range(),
                                ErrorKind::InvalidArgument,
                                format!(
                                    "DSL helper argument domains are incompatible with `{}`{detail}",
                                    helper.name()
                                ),
                            );
                            valid = false;
                            continue;
                        }
                    };
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
                    helper_argument_domains.push(argument_domains);
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
        let definition = if deferred_integer_domains.is_empty() {
            definition
        } else {
            match definition.finalize_helper_argument_domains(
                |expr| self.resolve_type_shape_dsl_intrinsic(expr),
                &helper_argument_domains,
            ) {
                Ok(definition) => Arc::new(definition),
                Err(error) => {
                    self.error(
                        errors,
                        error.range,
                        ErrorKind::InvalidArgument,
                        format!("@type_shape_dsl_function {}", error.message),
                    );
                    return None;
                }
            }
        };
        Some((definition, helpers))
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
                        "`@type_shape_dsl_function` parameter `{}` must be annotated as `Int`, `Int | None`, `IntTuple`, or a supported Flag value type (`int`, `bool`, `str`, `tuple[int, ...]`, `None`, or a union of these)",
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
            let (validated, helpers, valid_helpers) = match function_kind {
                FunctionKind::Def(func_id) => {
                    match self.resolve_and_finalize_type_shape_dsl_helpers(
                        func_id,
                        validated.clone(),
                        &parameter_domains,
                        result,
                        errors,
                    ) {
                        Some((validated, helpers)) => (validated, helpers, true),
                        None => (validated, Vec::new(), false),
                    }
                }
                _ => (validated, Vec::new(), true),
            };
            let mut valid_body = valid_helpers;
            for condition in validated.conditions() {
                let invalid_domain = match condition.kind() {
                    TypeShapeDslConditionKind::SlotCompare {
                        left_operand,
                        right_operand,
                        op,
                        ..
                    }
                    | TypeShapeDslConditionKind::IntegerCompare {
                        left_operand,
                        right_operand,
                        op,
                        ..
                    } => match type_shape_dsl_comparison_domain(
                        left_operand,
                        right_operand,
                        &parameter_domains,
                    ) {
                        Some(TypeShapeDslComparisonDomain::Dimension) => (!matches!(
                            op,
                            TypeShapeDslComparisonOp::Equal
                                | TypeShapeDslComparisonOp::NotEqual
                                | TypeShapeDslComparisonOp::LessThan
                        ))
                        .then_some("`@type_shape_dsl_function` `Int` comparisons support only `==`, `!=`, and `<`"),
                        Some(TypeShapeDslComparisonDomain::FlagInt) => None,
                        Some(TypeShapeDslComparisonDomain::FlagString) => (!matches!(
                            op,
                            TypeShapeDslComparisonOp::Equal
                                | TypeShapeDslComparisonOp::NotEqual
                        ))
                        .then_some("`@type_shape_dsl_function` Flag string comparisons support only `==` and `!=`"),
                        None => Some("`@type_shape_dsl_function` comparison operands must both be annotated as `Int` or both be `Flag[int]`; string equality also accepts compatible Flag string values"),
                    },
                    TypeShapeDslConditionKind::Any { .. }
                    | TypeShapeDslConditionKind::DimensionEquality { .. }
                    | TypeShapeDslConditionKind::StringEquality { .. }
                    | TypeShapeDslConditionKind::GeneratorElementSelfCompare(_) => None,
                    TypeShapeDslConditionKind::IsConcreteInt {
                        parameter_origins,
                        ..
                    } => parameter_origins.as_deref().and_then(|parameters| {
                        (!parameters.iter().all(|parameter| {
                            matches!(
                                parameter_domains[*parameter],
                                TypeShapeDslInputDomain::Value(TypeShapeDslDomain::Int)
                                    | TypeShapeDslInputDomain::OptionalInt
                            )
                        }))
                        .then_some("`@type_shape_dsl_function` `is_concrete_int` requires an `Int` or `Int | None` value")
                    }),
                    TypeShapeDslConditionKind::IsIntValue {
                        parameter_origins, ..
                    } => parameter_origins.as_deref().and_then(|parameters| {
                        (!parameters.iter().all(|parameter| matches!(
                            parameter_domains[*parameter],
                            TypeShapeDslInputDomain::Flag(domain)
                                if domain.is_subset_of(type_shape_dsl_narrowable_flag_domain())
                        )))
                        .then_some("`@type_shape_dsl_function` `is_int_value` requires a Flag[int | tuple[int, ...] | None] value")
                    }),
                    TypeShapeDslConditionKind::IsNone {
                        parameter_origins,
                        negated,
                        ..
                    } => parameter_origins.as_deref().and_then(|parameters| {
                        (!parameters.iter().all(|parameter| {
                            parameter_domains[*parameter] == TypeShapeDslInputDomain::OptionalInt
                                || matches!(
                                    parameter_domains[*parameter],
                                    TypeShapeDslInputDomain::Flag(domain)
                                        if domain.is_subset_of(type_shape_dsl_representable_flag_domain())
                                )
                        }))
                        .then_some(if *negated {
                            "`@type_shape_dsl_function` `is not None` requires an `Int | None` value or a supported Flag value"
                        } else {
                            "`@type_shape_dsl_function` `is None` requires an `Int | None` value or a supported Flag value"
                        })
                    }),
                    TypeShapeDslConditionKind::BoolSlot {
                        parameter_origins,
                        ..
                    } => parameter_origins.as_deref().and_then(|parameters| {
                        (!parameters.iter().all(|parameter| {
                            matches!(
                                parameter_domains[*parameter],
                                TypeShapeDslInputDomain::Flag(domain)
                                    if domain.contains(FlagMember::Bool)
                                        && domain.is_subset_of(type_shape_dsl_optional_flag_domain(FlagMember::Bool))
                            )
                        }))
                        .then_some("`@type_shape_dsl_function` a name used directly as a condition requires a boolean Flag value")
                    }),
                    TypeShapeDslConditionKind::FlagIntCompare(_)
                    | TypeShapeDslConditionKind::Membership { .. }
                    | TypeShapeDslConditionKind::LengthEqualLiteral { .. } => None,
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
                        parameter_uses: Some(uses),
                        ..
                    } => (!uses.iter().all(|use_| {
                        parameter_domains[use_.parameter()]
                            .can_use_as(TypeShapeDslDomain::Int, use_.narrowing())
                    }))
                    .then_some("`@type_shape_dsl_function` IntTuple elements must be annotated as `Int`, or as `Int | None` and narrowed to exclude `None`"),
                    TypeShapeDslExpressionKind::IntegerSlot {
                        parameter_uses: Some(uses),
                        ..
                    } => {
                        let valid = uses.iter().all(|use_| {
                            let narrowing = use_.narrowing();
                            parameter_domains[use_.parameter()]
                                .can_use_as(TypeShapeDslDomain::Int, narrowing)
                                || match parameter_domains[use_.parameter()] {
                                TypeShapeDslInputDomain::Flag(domain)
                                    if narrowing == TypeShapeDslParameterNarrowing::Integer =>
                                {
                                    domain.is_subset_of(type_shape_dsl_narrowable_flag_domain())
                                }
                                TypeShapeDslInputDomain::Flag(domain) => {
                                    domain == FlagDomain::of(FlagMember::Int)
                                }
                                _ => false,
                            }
                        });
                        (!valid).then_some("`@type_shape_dsl_function` dimension arithmetic operands must be annotated as `Int`, as `Int | None` narrowed to exclude `None`, or as a compatible integer Flag")
                    }
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
                    TypeShapeDslExpressionKind::IntTupleSlot {
                        parameter_origins: Some(shapes),
                        ..
                    } => {
                        (!shapes.iter().all(|shape| {
                            parameter_domains[*shape]
                                == TypeShapeDslInputDomain::Value(TypeShapeDslDomain::IntTuple)
                        }))
                        .then_some("`@type_shape_dsl_function` shape expression operands must be annotated as `IntTuple`")
                    }
                    TypeShapeDslExpressionKind::GeneratorSourceSlot {
                        parameter_uses: Some(uses),
                        ..
                    } => {
                        let valid = uses.iter().all(|use_| {
                            matches!(
                                parameter_domains[use_.parameter()],
                                TypeShapeDslInputDomain::Value(TypeShapeDslDomain::IntTuple)
                            ) || match parameter_domains[use_.parameter()] {
                                TypeShapeDslInputDomain::Flag(domain)
                                    if use_.narrowing()
                                        != TypeShapeDslParameterNarrowing::Unnarrowed =>
                                {
                                    domain.is_subset_of(type_shape_dsl_narrowable_flag_domain())
                                }
                                TypeShapeDslInputDomain::Flag(domain) => {
                                    domain.is_subset_of(FlagDomain::of(FlagMember::IntTuple))
                                }
                                _ => false,
                            }
                        });
                        (!valid).then_some(
                            "`@type_shape_dsl_function` generator source must be an `IntTuple` or Flag sequence",
                        )
                    }
                    TypeShapeDslExpressionKind::FlagValueSlot {
                        parameter_uses: Some(uses),
                        required,
                        ..
                    } => {
                        let valid = uses.iter().all(|use_| {
                            match parameter_domains[use_.parameter()] {
                                TypeShapeDslInputDomain::Flag(domain) => match required {
                                    TypeShapeDslFlagValueKind::Int
                                        if use_.narrowing()
                                            == TypeShapeDslParameterNarrowing::Integer =>
                                    {
                                        domain.is_subset_of(type_shape_dsl_narrowable_flag_domain())
                                    }
                                    TypeShapeDslFlagValueKind::Int => {
                                        domain == FlagDomain::of(FlagMember::Int)
                                    }
                                    TypeShapeDslFlagValueKind::String => {
                                        domain.contains(FlagMember::Str)
                                            && domain.is_subset_of(type_shape_dsl_optional_flag_domain(
                                                FlagMember::Str,
                                            ))
                                    }
                                    TypeShapeDslFlagValueKind::Sequence
                                        if use_.narrowing()
                                            != TypeShapeDslParameterNarrowing::Unnarrowed =>
                                    {
                                        domain.is_subset_of(type_shape_dsl_narrowable_flag_domain())
                                    }
                                    TypeShapeDslFlagValueKind::Sequence => {
                                        domain.is_subset_of(FlagDomain::of(FlagMember::IntTuple))
                                    }
                                },
                                _ => false,
                            }
                        });
                        (!valid).then_some("`@type_shape_dsl_function` Flag operation requires a compatible Flag parameter")
                    }
                    TypeShapeDslExpressionKind::DimensionLiteral(_)
                    | TypeShapeDslExpressionKind::Gradual
                    | TypeShapeDslExpressionKind::IntTupleSlot { .. }
                    | TypeShapeDslExpressionKind::IntTupleSlice
                    | TypeShapeDslExpressionKind::IntTupleConcat
                    | TypeShapeDslExpressionKind::IntTupleConstructor
                    | TypeShapeDslExpressionKind::IntTupleProduct
                    | TypeShapeDslExpressionKind::DimensionSlot { .. }
                    | TypeShapeDslExpressionKind::IntegerSlot { .. }
                    | TypeShapeDslExpressionKind::IntTupleIndex { .. }
                    | TypeShapeDslExpressionKind::IntTupleLength { .. }
                    | TypeShapeDslExpressionKind::GeneratorSourceSlot { .. }
                    | TypeShapeDslExpressionKind::GeneratorElementAsDimension(_)
                    | TypeShapeDslExpressionKind::GeneratorElementAsFlagInt(_)
                    | TypeShapeDslExpressionKind::GeneratorZip { .. }
                    | TypeShapeDslExpressionKind::Slot(_)
                    | TypeShapeDslExpressionKind::FlagValueSlot { .. }
                    | TypeShapeDslExpressionKind::FlagIntLiteral(_)
                    | TypeShapeDslExpressionKind::FlagStringLiteral
                    | TypeShapeDslExpressionKind::FlagBool(_)
                    | TypeShapeDslExpressionKind::FlagNone
                    | TypeShapeDslExpressionKind::FlagTuple
                    | TypeShapeDslExpressionKind::FlagRange
                    | TypeShapeDslExpressionKind::FlagSequenceLength
                    | TypeShapeDslExpressionKind::FlagSequenceCount
                    | TypeShapeDslExpressionKind::FlagSequenceIndex
                    | TypeShapeDslExpressionKind::FlagIntArithmetic(_)
                    | TypeShapeDslExpressionKind::DimensionArithmetic(_)
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
                    TypeShapeDslReturnKind::Slot {
                        kind: TypeShapeDslSlotReturnKind::KnownDomain { domain, .. },
                        ..
                    } if *domain != result => {
                        self.error(errors, return_.range(), ErrorKind::InvalidArgument, "`@type_shape_dsl_function` local return domain must match the declared result".to_owned());
                        valid_body = false;
                    }
                    TypeShapeDslReturnKind::Slot {
                        kind:
                            TypeShapeDslSlotReturnKind::KnownDomain {
                                domain,
                                parameter_uses: Some(uses),
                            },
                        ..
                    } if !uses.iter().all(|use_| {
                        parameter_domains[use_.parameter()].can_use_as(*domain, use_.narrowing())
                    }) =>
                    {
                        self.error(
                            errors,
                            return_.range(),
                            ErrorKind::InvalidArgument,
                            format!(
                                "`@type_shape_dsl_function` local return requires contributing parameters to use the `{}` domain",
                                domain.as_str(),
                            ),
                        );
                        valid_body = false;
                    }
                    TypeShapeDslReturnKind::Slot {
                        kind: TypeShapeDslSlotReturnKind::DirectParameter(use_),
                        ..
                    } if !parameter_domains[use_.parameter()]
                        .can_use_as(result, use_.narrowing()) =>
                    {
                        let parameter = use_.parameter();
                        self.error(
                            errors,
                            return_.range(),
                            ErrorKind::InvalidArgument,
                            match parameter_domains[parameter] {
                                TypeShapeDslInputDomain::Flag(_) => format!(
                                    "`@type_shape_dsl_function` Flag parameter `{}` is input-only and cannot be returned",
                                    dsl.parameter_name(parameter)
                                ),
                                TypeShapeDslInputDomain::OptionalInt
                                    if result == TypeShapeDslDomain::Int => format!(
                                    "`@type_shape_dsl_function` `Int | None` parameter `{}` must be narrowed to exclude `None` before it can be returned as `Int`",
                                    dsl.parameter_name(parameter)
                                ),
                                TypeShapeDslInputDomain::OptionalInt
                                | TypeShapeDslInputDomain::Value(_) => format!(
                                    "`@type_shape_dsl_function` return annotation must match returned parameter `{}`",
                                    dsl.parameter_name(parameter)
                                ),
                            },
                        );
                        valid_body = false;
                    }
                    TypeShapeDslReturnKind::Slot {
                        kind: TypeShapeDslSlotReturnKind::ParameterAlias(parameter_uses),
                        ..
                    } => {
                        if let Some(invalid_use) = parameter_uses.iter().find(|use_| {
                            !parameter_domains[use_.parameter()]
                                .can_use_as(result, use_.narrowing())
                        }) {
                            let message = match parameter_domains[invalid_use.parameter()] {
                                TypeShapeDslInputDomain::Flag(_) => format!(
                                        "`@type_shape_dsl_function` Flag parameter `{}` is input-only and cannot be returned",
                                        dsl.parameter_name(invalid_use.parameter())
                                    ),
                                TypeShapeDslInputDomain::OptionalInt
                                    if result == TypeShapeDslDomain::Int => format!(
                                        "`@type_shape_dsl_function` `Int | None` parameter `{}` must be narrowed to exclude `None` before it can be returned as `Int`",
                                        dsl.parameter_name(invalid_use.parameter())
                                    ),
                                TypeShapeDslInputDomain::OptionalInt
                                | TypeShapeDslInputDomain::Value(_) => "`@type_shape_dsl_function` local alias return domain must match the declared result".to_owned(),
                            };
                            self.error(
                                errors,
                                return_.range(),
                                ErrorKind::InvalidArgument,
                                message,
                            );
                            valid_body = false;
                        }
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
                    TypeShapeDslReturnKind::Expression(domain) if *domain != result => {
                        self.error(
                            errors,
                            return_.range(),
                            ErrorKind::InvalidArgument,
                            format!(
                                "`@type_shape_dsl_function` returned expression requires a result in the `{}` domain",
                                domain.as_str(),
                            ),
                        );
                        valid_body = false;
                    }
                    TypeShapeDslReturnKind::Slot { .. }
                    | TypeShapeDslReturnKind::Broadcast { .. }
                    | TypeShapeDslReturnKind::Expression(_)
                    | TypeShapeDslReturnKind::Invalid
                    | TypeShapeDslReturnKind::HelperCall(_)
                    | TypeShapeDslReturnKind::Gradual(_) => {}
                }
            }
            if valid_body && let FunctionKind::Def(func_id) = function_kind {
                match ResolvedTypeShapeDslFunction::try_new(
                    func_id.clone(),
                    validated,
                    parameter_domains,
                    result,
                    helpers,
                ) {
                    Ok(function) => {
                        return Some(FunctionKind::TypeShapeDsl(
                            func_id.clone(),
                            Arc::new(function),
                        ));
                    }
                    Err(error) => {
                        self.error(
                            errors,
                            function_range,
                            ErrorKind::InvalidArgument,
                            error.message().to_owned(),
                        );
                    }
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
        if let Type::ClassDef(class) = &callee
            && class.qname().module_name().as_str() == "builtins"
            && class.qname().id().as_str() == "zip"
        {
            return Some(TypeShapeDslIntrinsic::Zip);
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
        if id.has_toplevel_qname("shape_extensions.dsl", "concat") {
            return Some(TypeShapeDslIntrinsic::Concat);
        }
        if id.has_toplevel_qname("shape_extensions.dsl", "prod") {
            return Some(TypeShapeDslIntrinsic::Prod);
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
                call.range(),
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
                call.range(),
                ErrorKind::InvalidAnnotation,
                format!(
                    "Expected {} {} for `{name}`, got {}",
                    parameter_domains.len(),
                    pluralize(parameter_domains.len(), "argument"),
                    call.arguments.args.len()
                ),
            );
        }

        let type_argument_context = TypeFormContext::TypeArgument(&type_form_context);
        let mut args = Vec::with_capacity(call.arguments.args.len());
        for (index, (arg_expr, domain)) in call
            .arguments
            .args
            .iter()
            .zip(parameter_domains)
            .enumerate()
        {
            let argument_context = TypeShapeDslArgumentContext {
                function_name: name,
                parameter_name: function.parameter_name(index).as_str(),
                position: index + 1,
            };
            let arg = self.parse_type_shape_dsl_argument(
                arg_expr,
                *domain,
                type_argument_context,
                argument_context,
                errors,
            );
            if arg.is_error() {
                return arg;
            }
            if !self.is_type_shape_dsl_argument(&arg, *domain) {
                let article = match domain {
                    TypeShapeDslInputDomain::Flag(_) => "a",
                    TypeShapeDslInputDomain::Value(_) | TypeShapeDslInputDomain::OptionalInt => {
                        "an"
                    }
                };
                let displayed = self.for_display(arg.clone());
                return self.error(
                    errors,
                    arg_expr.range(),
                    ErrorKind::InvalidAnnotation,
                    format!(
                        "Expected {article} `{domain}` argument for parameter `{}` (position {}) of `{}`, got `{displayed}`",
                        argument_context.parameter_name,
                        argument_context.position,
                        argument_context.function_name,
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
        argument_context: TypeShapeDslArgumentContext<'_>,
        errors: &ErrorCollector,
    ) -> Type {
        match domain {
            TypeShapeDslInputDomain::Value(TypeShapeDslDomain::Int)
            | TypeShapeDslInputDomain::OptionalInt => {
                let raw_int_var_error = |range, ty| {
                    let displayed = self.for_display(ty);
                    let TypeShapeDslArgumentContext {
                        function_name,
                        parameter_name,
                        position,
                    } = argument_context;
                    self.error(
                        errors,
                        range,
                        ErrorKind::InvalidAnnotation,
                        format!(
                            "Expected an `{domain}` argument for parameter `{parameter_name}` (position {position}) of `{function_name}`; raw `IntVar` `{displayed}` must be wrapped as `Int[{displayed}]`"
                        ),
                    )
                };
                if matches!(domain, TypeShapeDslInputDomain::OptionalInt)
                    && let Expr::BinOp(binary) = arg
                    && binary.op == Operator::BitOr
                {
                    let dimension = if matches!(binary.left.as_ref(), Expr::NoneLiteral(_)) {
                        Some(binary.right.as_ref())
                    } else if matches!(binary.right.as_ref(), Expr::NoneLiteral(_)) {
                        Some(binary.left.as_ref())
                    } else {
                        None
                    };
                    if let Some(dimension) = dimension {
                        let discarded = self.error_collector();
                        if let Err(DimensionExprError::RawIntVar { range, ty }) = self
                            .parse_dimension_list_for_type_shape_dsl_int_argument(
                                slice::from_ref(dimension),
                                type_form_context,
                                &discarded,
                            )
                        {
                            return raw_int_var_error(range, ty);
                        }
                    }
                }
                let dimension_errors = self.error_collector();
                let parsed_dimension = match self
                    .parse_dimension_list_for_type_shape_dsl_int_argument(
                        slice::from_ref(arg),
                        type_form_context,
                        &dimension_errors,
                    ) {
                    Ok(dimensions) => dimensions.into_iter().next().filter(|ty| !ty.is_error()),
                    Err(DimensionExprError::Invalid) => None,
                    Err(DimensionExprError::InvalidExplicitIntWrapper) => {
                        errors.extend(dimension_errors);
                        return Type::any_error();
                    }
                    Err(DimensionExprError::RawIntVar { range, ty }) => {
                        errors.extend(dimension_errors);
                        return raw_int_var_error(range, ty);
                    }
                };
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
                call.range(),
                ErrorKind::InvalidAnnotation,
                "`broadcast` does not accept keyword arguments".to_owned(),
            );
        }
        if call.arguments.args.len() != 2 {
            return self.error(
                errors,
                call.range(),
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
                Type::Quantified(_) | Type::TypeVar(_) => is_gradual_size_bound_type_var(ty),
                _ => false,
            },
            TypeShapeDslInputDomain::Value(TypeShapeDslDomain::IntTuple) => {
                self.is_int_tuple_dsl_argument(ty)
            }
            TypeShapeDslInputDomain::OptionalInt => match ty {
                Type::None => true,
                Type::Union(_) => is_optional_int(ty),
                _ if is_optional_int_bound_type_var(ty) => true,
                _ => self.is_type_shape_dsl_argument(
                    ty,
                    TypeShapeDslInputDomain::Value(TypeShapeDslDomain::Int),
                ),
            },
            TypeShapeDslInputDomain::Flag(domain) => {
                if matches!(ty, Type::Any(AnyStyle::Error)) {
                    return false;
                }
                domain.accepts_with_str_subclasses(ty, |member| match member {
                    Type::ClassType(cls) | Type::SelfType(cls) => {
                        self.has_superclass(cls.class_object(), self.stdlib.str().class_object())
                    }
                    _ => false,
                })
            }
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
    if is_optional_int(ty) {
        return Some(TypeShapeDslInputDomain::OptionalInt);
    }
    type_shape_dsl_domain(ty)
        .map(TypeShapeDslInputDomain::Value)
        .or_else(|| FlagDomain::from_type(ty).map(TypeShapeDslInputDomain::Flag))
}
