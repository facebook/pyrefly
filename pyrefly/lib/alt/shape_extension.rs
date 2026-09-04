/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Shared helpers for experimental shape-extension types and restrictions.

use std::sync::Arc;

use pyrefly_types::callable::Param;
use pyrefly_types::quantified::Quantified;
use pyrefly_types::tuple::Tuple;
use pyrefly_types::type_var::FlagDomain;
use pyrefly_types::type_var::Restriction;
use pyrefly_types::types::TParams;
use pyrefly_types::types::TParamsSource;
use pyrefly_types::types::Type;
use pyrefly_types::types::Var;
use ruff_python_ast::Expr;
use ruff_python_ast::name::Name;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use starlark_map::small_set::SmallSet;

use crate::alt::answers::LookupAnswer;
use crate::alt::answers_solver::AnswersSolver;
use crate::alt::solve::TypeFormContext;
use crate::binding::binding::FunctionDefData;
use crate::binding::shape_type::TypeParameterBound;
use crate::config::error_kind::ErrorKind;
use crate::error::collector::ErrorCollector;

/// Returns whether `ty` is the normalized upper bound for an `IntTuple`-bounded `TypeVar`.
///
/// Other tuple bounds are ordinary type bounds and must not enable shape-specific parsing.
pub(crate) fn is_int_tuple_bound(ty: &Type, int_type: &Type) -> bool {
    match ty {
        Type::IntTuple(_) => true,
        Type::Tuple(Tuple::Unbounded(inner)) => inner.as_ref() == int_type,
        _ => false,
    }
}

pub(crate) fn shape_extension_vars(tparams: &TParams, vars: &[Var]) -> Option<Arc<SmallSet<Var>>> {
    assert_eq!(
        tparams.len(),
        vars.len(),
        "fresh callable variables must align with type parameters"
    );
    let vars = tparams
        .iter()
        .zip(vars)
        .filter_map(|(tparam, var)| {
            tparam
                .restriction()
                .uses_direct_value_source()
                .then_some(*var)
        })
        .collect::<SmallSet<_>>();
    (!vars.is_empty()).then(|| Arc::new(vars))
}

pub(crate) fn direct_function_parameter_sources(
    stmt: &FunctionDefData,
    params: &[Param],
    tparam: &Quantified,
) -> Vec<(usize, TextRange, bool)> {
    stmt.parameters
        .iter()
        .zip(params)
        .enumerate()
        .filter_map(|(index, (parameter, param))| {
            let single_value_parameter =
                matches!(param, Param::PosOnly(..) | Param::Pos(..) | Param::KwOnly(..));
            // Scalar sources require direct syntax. Unpacked sources use the resolved type so
            // equivalent `Unpack` spellings are treated identically.
            match (parameter.annotation(), param) {
                (Some(Expr::Name(name)), _)
                    if single_value_parameter && name.id == *tparam.name() =>
                {
                    Some((index, name.range(), false))
                }
                (Some(annotation), Param::Varargs(_, Type::Unpack(inner)))
                    if matches!(&**inner, Type::Quantified(q) if q.as_ref() == tparam) =>
                {
                    Some((index, annotation.range(), true))
                }
                _ => None,
            }
        })
        .collect()
}

impl<Ans: LookupAnswer> AnswersSolver<'_, '_, Ans> {
    pub(crate) fn validate_shape_extension_type_parameter_default(
        &self,
        name: &Name,
        default: &Type,
        range: TextRange,
        restriction: &Restriction,
        errors: &ErrorCollector,
    ) -> Option<Type> {
        self.validate_shape_flag_type_parameter_default(name, default, range, restriction, errors)
            .or_else(|| {
                self.validate_shape_index_type_parameter_default(
                    name,
                    default,
                    range,
                    restriction,
                    errors,
                )
            })
    }

    pub(crate) fn validate_shape_extension_function_parameters(
        &self,
        stmt: &FunctionDefData,
        params: &[Param],
        tparams: &TParams,
        errors: &ErrorCollector,
    ) {
        self.validate_shape_flag_function_parameters(stmt, params, tparams, errors);
        self.validate_shape_index_function_parameters(stmt, params, tparams, errors);
    }

    pub(crate) fn reject_legacy_shape_extension_bound(
        &self,
        bound: &Type,
        range: TextRange,
        errors: &ErrorCollector,
    ) -> bool {
        let kind = match bound {
            Type::ClassType(cls) if cls.has_qname("shape_extensions", "Flag") => "Flag",
            Type::ClassType(cls) if cls.has_qname("shape_extensions", "Index") => "Index",
            _ => return false,
        };
        self.error(
            errors,
            range,
            ErrorKind::InvalidTypeVar,
            format!(
                "`shape_extensions.{kind}` is supported only as a direct PEP 695 type parameter bound"
            ),
        );
        true
    }

    pub(crate) fn resolve_shape_type_parameter_bound(
        &self,
        bound: &TypeParameterBound,
        errors: &ErrorCollector,
    ) -> Restriction {
        match bound {
            TypeParameterBound::ShapeFlag {
                domain: Some(domain),
                ..
            } => {
                let domain_ty =
                    self.expr_untype(domain, TypeFormContext::TypeVarConstraint, errors);
                if domain_ty.is_error() {
                    return Restriction::Unrestricted;
                }
                match FlagDomain::from_type(&domain_ty) {
                    Some(flag_domain) => Restriction::flag(flag_domain),
                    None => {
                        self.error(
                            errors,
                            domain.range(),
                            ErrorKind::InvalidTypeVar,
                            format!(
                                "`Flag` domain must resolve to a nonempty union of `int`, `bool`, `str`, `None`, and integer tuples of one fixed arity or `tuple[int, ...]`, got `{domain_ty}`"
                            ),
                        );
                        Restriction::Unrestricted
                    }
                }
            }
            TypeParameterBound::ShapeFlag {
                domain: None,
                range,
            } => {
                self.error(
                    errors,
                    *range,
                    ErrorKind::InvalidTypeVar,
                    "`shape_extensions.Flag` requires one domain argument: `int`, `bool`, `str`, `tuple[int, ...]`, `None`, or a union of these"
                        .to_owned(),
                );
                Restriction::Unrestricted
            }
            TypeParameterBound::ShapeIndex => Restriction::index(),
            TypeParameterBound::Ordinary(bound) => {
                let bound_ty = self.expr_untype(bound, TypeFormContext::TypeVarConstraint, errors);
                let aliased_kind = match &bound_ty {
                    Type::ClassType(cls) if cls.has_qname("shape_extensions", "Flag") => {
                        Some("Flag")
                    }
                    Type::ClassType(cls) if cls.has_qname("shape_extensions", "Index") => {
                        Some("Index")
                    }
                    _ => None,
                };
                if let Some(kind) = aliased_kind {
                    // TODO: Distinguish quoted canonical bounds from aliases so quoted syntax gets
                    // its intended behavior or a quoted-form diagnostic rather than an alias error.
                    self.error(
                        errors,
                        bound.range(),
                        ErrorKind::InvalidTypeVar,
                        format!(
                            "`shape_extensions.{kind}` must be used directly rather than through a type alias"
                        ),
                    );
                    Restriction::Unrestricted
                } else {
                    Restriction::Bound(bound_ty)
                }
            }
        }
    }

    pub(crate) fn validate_shape_extension_type_parameter_scope(
        &self,
        tparams: &[Quantified],
        source: &TParamsSource,
        range: TextRange,
        errors: &ErrorCollector,
    ) {
        if matches!(source, TParamsSource::TypeAlias)
            && tparams.iter().any(|tparam| tparam.restriction().is_flag())
        {
            self.error(
                errors,
                range,
                ErrorKind::InvalidTypeVar,
                "`Flag` type parameters are not supported on type aliases".to_owned(),
            );
        }
        let index_source = match source {
            TParamsSource::Function => None,
            TParamsSource::Class => Some("classes"),
            TParamsSource::TypeAlias => Some("type aliases"),
        };
        if let Some(source_name) = index_source
            && tparams.iter().any(|tparam| tparam.restriction().is_index())
        {
            self.error(
                errors,
                range,
                ErrorKind::InvalidTypeVar,
                format!("`Index` type parameters are not supported on {source_name}"),
            );
        }
    }
}
