/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::sync::Arc;

use pyrefly_types::callable::Param;
use pyrefly_types::callable::Required;
use pyrefly_types::quantified::Quantified;
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

pub(crate) fn shape_flag_vars(tparams: &TParams, vars: &[Var]) -> Option<Arc<SmallSet<Var>>> {
    assert_eq!(
        tparams.len(),
        vars.len(),
        "fresh callable variables must align with type parameters"
    );
    let vars = tparams
        .iter()
        .zip(vars)
        .filter_map(|(tparam, var)| {
            matches!(tparam.restriction(), Restriction::Flag(_)).then_some(*var)
        })
        .collect::<SmallSet<_>>();
    (!vars.is_empty()).then(|| Arc::new(vars))
}

impl<'a, Ans: LookupAnswer> AnswersSolver<'a, Ans> {
    pub(crate) fn reject_legacy_shape_flag_bound(
        &self,
        bound: &Type,
        range: TextRange,
        errors: &ErrorCollector,
    ) -> bool {
        if matches!(bound, Type::ClassType(cls) if cls.has_qname("shape_extensions", "Flag")) {
            self.error(
                errors,
                range,
                ErrorKind::InvalidTypeVar,
                "`shape_extensions.Flag` is supported only as a direct PEP 695 type parameter bound"
                    .to_owned(),
            );
            true
        } else {
            false
        }
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
                    Some(flag_domain) => Restriction::Flag(flag_domain),
                    None => {
                        self.error(
                            errors,
                            domain.range(),
                            ErrorKind::InvalidTypeVar,
                            format!(
                                "`Flag` domain must resolve to `int`, `bool`, `str`, `tuple[int, ...]`, `None`, or a union of these, got `{domain_ty}`"
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
            TypeParameterBound::Ordinary(bound) => {
                let bound_ty = self.expr_untype(bound, TypeFormContext::TypeVarConstraint, errors);
                if matches!(&bound_ty, Type::ClassType(cls) if cls.has_qname("shape_extensions", "Flag"))
                {
                    // TODO: Distinguish quoted canonical bounds from aliases so quoted syntax gets
                    // its intended behavior or a quoted-form diagnostic rather than an alias error.
                    self.error(
                        errors,
                        bound.range(),
                        ErrorKind::InvalidTypeVar,
                        "`shape_extensions.Flag` must be used directly rather than through a type alias"
                            .to_owned(),
                    );
                    Restriction::Unrestricted
                } else {
                    Restriction::Bound(bound_ty)
                }
            }
        }
    }

    pub(crate) fn validate_shape_flag_type_parameter_scope(
        &self,
        tparams: &[Quantified],
        source: &TParamsSource,
        range: TextRange,
        errors: &ErrorCollector,
    ) {
        if !matches!(source, TParamsSource::Function)
            && tparams
                .iter()
                .any(|tparam| matches!(tparam.restriction(), Restriction::Flag(_)))
        {
            self.error(
                errors,
                range,
                ErrorKind::InvalidTypeVar,
                "`Flag` type parameters are currently supported only on functions".to_owned(),
            );
        }
    }

    pub(crate) fn validate_shape_flag_type_parameter_default(
        &self,
        name: &Name,
        default: &Type,
        range: TextRange,
        restriction: &Restriction,
        errors: &ErrorCollector,
    ) -> Option<Type> {
        let Restriction::Flag(domain) = restriction else {
            return None;
        };
        // Tuple type expressions infer as `type[...]`; defaults store the corresponding value.
        let default = match default {
            Type::Type(inner) => inner.as_ref(),
            _ => default,
        };
        if domain.accepts_literal(default) {
            Some(default.clone())
        } else {
            self.error(
                errors,
                range,
                ErrorKind::InvalidTypeVar,
                format!(
                    "Default for `Flag[{domain}]` type parameter `{name}` must be a `{domain}` literal, got `{default}`"
                ),
            );
            Some(self.heap.mk_any_error())
        }
    }

    pub(crate) fn validate_shape_flag_function_parameters(
        &self,
        stmt: &FunctionDefData,
        params: &[Param],
        tparams: &TParams,
        errors: &ErrorCollector,
    ) {
        for (tparam, domain) in tparams
            .iter()
            .filter_map(|tparam| match tparam.restriction() {
                Restriction::Flag(domain) => Some((tparam, domain)),
                _ => None,
            })
        {
            let sources = stmt
                .parameters
                .iter()
                .zip(params)
                .enumerate()
                .filter_map(|(index, (parameter, param))| {
                    let single_value_parameter = matches!(
                        param,
                        Param::PosOnly(..) | Param::Pos(..) | Param::KwOnly(..)
                    );
                    match parameter.annotation() {
                        Some(Expr::Name(name))
                            if single_value_parameter && name.id == *tparam.name() =>
                        {
                            Some((index, name.range()))
                        }
                        _ => None,
                    }
                })
                .collect::<Vec<_>>();
            if sources.len() != 1 {
                self.error(
                    errors,
                    stmt.name.range(),
                    ErrorKind::InvalidTypeVar,
                    format!(
                        "`Flag` type parameter `{}` must directly annotate exactly one function parameter, found {}",
                        tparam.name(),
                        sources.len(),
                    ),
                );
                continue;
            }
            let (source_index, source_range) = sources[0];
            let runtime_default = match &params[source_index] {
                Param::PosOnly(_, _, Required::Optional(default))
                | Param::Pos(_, _, Required::Optional(default))
                | Param::KwOnly(_, _, Required::Optional(default)) => default.as_ref(),
                _ => None,
            };
            if let Some(default) = runtime_default
                && !domain.accepts_literal(&default.ty)
            {
                self.error(
                    errors,
                    source_range,
                    ErrorKind::InvalidTypeVar,
                    format!(
                        "Default for parameter binding `Flag[{domain}]` type parameter `{}` must be a `{domain}` literal, got `{}`",
                        tparam.name(),
                        default.ty,
                    ),
                );
            }
        }
    }

    pub(crate) fn is_shape_flag_parameter_type(&self, ty: &Type) -> bool {
        matches!(ty, Type::Quantified(q) if matches!(q.restriction(), Restriction::Flag(_)))
    }
}
