/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Normalization and redundant-union simplification for `shape_extensions.Scalar`.
//!
//! The marker representation and subset logic live in `crate::solver::shape_markers`; this module
//! keeps only the `AnswersSolver` work of normalizing markers and simplifying signatures.

use pyrefly_types::callable::Callable;
use pyrefly_types::callable::Params;
use pyrefly_types::callable::PrefixParam;
use pyrefly_types::callable::Required;
use pyrefly_types::quantified::Quantified;
use pyrefly_types::type_var::Restriction;
use pyrefly_types::types::TParams;
use pyrefly_types::types::Type;
use pyrefly_util::visit::VisitMut;
use ruff_text_size::TextRange;

use crate::alt::answers::LookupAnswer;
use crate::alt::answers_solver::AnswersSolver;
use crate::config::error_kind::ErrorKind;
use crate::error::collector::ErrorCollector;
use crate::solver::shape_markers::ScalarFamily;
use crate::solver::shape_markers::ScalarNormalForm;
use crate::solver::shape_markers::expanded_scalar_normal_form;
use crate::solver::shape_markers::scalar;
use crate::solver::solver::Solver;

/// Whether `ty` contains a `Scalar` marker that normalization may replace.
fn contains_scalar(ty: &Type) -> bool {
    ty.any(|candidate| scalar(candidate).is_some())
}

/// Return the ordinary domain represented by a viable `Scalar` specialization.
///
/// `None` means that `ty` is not a `Scalar`.
pub(crate) fn scalar_upper_bound(solver: &Solver, ty: &Type) -> Option<Type> {
    match expanded_scalar_normal_form(solver, ty) {
        Some(
            ScalarNormalForm::Domain(domain)
            | ScalarNormalForm::Suspended(ScalarFamily { domain, .. }),
        ) => Some(domain),
        None => None,
    }
}

impl<'ctx, 'answer, Ans: LookupAnswer> AnswersSolver<'ctx, 'answer, Ans> {
    fn normalize_scalar_type_inner(&self, ty: &mut Type) -> bool {
        let mut changed = false;
        ty.recurse_mut(&mut |child| changed |= self.normalize_scalar_type_inner(child));

        if let Some(ScalarNormalForm::Domain(domain)) =
            expanded_scalar_normal_form(self.solver(), ty)
            && *ty != domain
        {
            *ty = domain;
            changed = true;
        }
        if changed && let Type::Union(union) = ty {
            let simplified = self.unions(union.members.clone());
            if simplified != *ty {
                *ty = simplified;
            }
        }
        changed
    }

    /// Remove `Scalar` members already covered by ordinary members of the same union.
    fn simplify_redundant_scalar_unions_in_type(&self, ty: Type) -> Type {
        ty.transform(&mut |candidate| {
            let Type::Union(union) = candidate else {
                return;
            };
            let ordinary_members = union
                .members
                .iter()
                .filter(|member| scalar(member).is_none())
                .cloned()
                .collect::<Vec<_>>();
            if ordinary_members.is_empty() {
                return;
            }
            let ordinary = self.unions(ordinary_members);
            let members = union
                .members
                .iter()
                .filter(|member| {
                    let Some(marker) = scalar(member) else {
                        return true;
                    };
                    let snapshot = self
                        .solver()
                        .snapshot_for_speculative_inference(&[&marker.domain, &ordinary]);
                    // `AnswersSolver::is_subset_eq` owns a transient `Subset`, so only solver
                    // variables need to be restored after this probe.
                    let redundant = self.is_subset_eq(&marker.domain, &ordinary);
                    self.solver().restore_vars(snapshot);
                    !redundant
                })
                .cloned()
                .collect::<Vec<_>>();
            if members.len() != union.members.len() {
                *candidate = self.unions(members);
            }
        })
    }

    /// Remove redundant `Scalar` union arms recursively and reject annotations that would strand
    /// an observable shape variable by doing so.
    pub(crate) fn simplify_redundant_scalar_unions(
        &self,
        callable: &mut Callable,
        tparams: &TParams,
        parameter_ranges: &[TextRange],
        fallback_range: TextRange,
        errors: &ErrorCollector,
    ) {
        let ret = &callable.ret;
        let (original_parameter_types, required_parameters) = match &callable.params {
            Params::List(params) => (
                params
                    .items()
                    .iter()
                    .map(|param| param.as_type().clone())
                    .collect::<Vec<_>>(),
                params
                    .items()
                    .iter()
                    .map(|param| param.is_required())
                    .collect::<Vec<_>>(),
            ),
            Params::ParamSpec(prefix, _) => (
                prefix
                    .iter()
                    .map(|param| param.ty().clone())
                    .collect::<Vec<_>>(),
                prefix
                    .iter()
                    .map(|param| {
                        matches!(
                            param,
                            PrefixParam::PosOnly(_, _, Required::Required)
                                | PrefixParam::Pos(_, _, Required::Required)
                        )
                    })
                    .collect::<Vec<_>>(),
            ),
            Params::Partial(_) | Params::Ellipsis | Params::Materialization => return,
        };
        if !original_parameter_types.iter().any(contains_scalar) {
            return;
        }
        let mut simplified_parameter_types = original_parameter_types
            .iter()
            .map(|parameter_type| {
                self.simplify_redundant_scalar_unions_in_type(parameter_type.clone())
            })
            .collect::<Vec<_>>();

        let contains_tparam = |ty: &Type, tparam: &Quantified| {
            ty.any(|ty| matches!(ty, Type::Quantified(other) if other.as_ref() == tparam))
        };
        let mut observable_tparams = tparams
            .iter()
            .filter(|tparam| contains_tparam(ret, tparam))
            .collect::<Vec<_>>();
        loop {
            let mut changed = false;
            for dependency in tparams.iter() {
                if observable_tparams.contains(&dependency) {
                    continue;
                }
                let referenced_by_observable = observable_tparams.iter().any(|observable| {
                    observable
                        .default()
                        .is_some_and(|default| contains_tparam(default, dependency))
                        || match observable.restriction() {
                            Restriction::Bound(bound) => contains_tparam(bound, dependency),
                            Restriction::Constraints(constraints) => constraints
                                .iter()
                                .any(|constraint| contains_tparam(constraint, dependency)),
                            Restriction::ShapeExtension(_) | Restriction::Unrestricted => false,
                        }
                });
                if referenced_by_observable {
                    observable_tparams.push(dependency);
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }

        for tparam in observable_tparams {
            let mentions_tparam = |ty: &Type| contains_tparam(ty, tparam);
            if simplified_parameter_types
                .iter()
                .zip(&required_parameters)
                .any(|(parameter_type, required)| *required && mentions_tparam(parameter_type))
            {
                continue;
            }
            let lost_indices = original_parameter_types
                .iter()
                .zip(&simplified_parameter_types)
                .enumerate()
                .filter_map(|(index, (original, simplified))| {
                    (mentions_tparam(original) && !mentions_tparam(simplified)).then_some(index)
                })
                .collect::<Vec<_>>();
            if let Some(&first_index) = lost_indices.first() {
                self.error(
                    errors,
                    parameter_ranges
                        .get(first_index)
                        .copied()
                        .unwrap_or(fallback_range),
                    ErrorKind::InvalidAnnotation,
                    format!(
                        "Redundant `Scalar` union arm cannot bind observable type parameter `{}`",
                        tparam.name()
                    ),
                );
                for index in lost_indices {
                    simplified_parameter_types[index] = original_parameter_types[index].clone();
                }
            }
        }
        match &mut callable.params {
            Params::List(params) => {
                for (param, parameter_type) in params
                    .items_mut()
                    .iter_mut()
                    .zip(simplified_parameter_types)
                {
                    *param.as_type_mut() = parameter_type;
                }
            }
            Params::ParamSpec(prefix, _) => {
                for (param, parameter_type) in prefix.iter_mut().zip(simplified_parameter_types) {
                    match param {
                        PrefixParam::PosOnly(_, ty, _) | PrefixParam::Pos(_, ty, _) => {
                            *ty = parameter_type
                        }
                    }
                }
            }
            Params::Partial(_) | Params::Ellipsis | Params::Materialization => {
                unreachable!("only callable parameters collected above are rewritten")
            }
        }
    }

    /// Replace resolved `Scalar` occurrences with their domain or `Never` normal form.
    ///
    /// Union members are normalized first so their containing union can simplify redundant
    /// members. Unchanged unions are left intact to preserve alias display names.
    pub(crate) fn normalize_scalar_type(&self, ty: Type) -> Type {
        if !self.solver().tensor_shapes || !contains_scalar(&ty) {
            return ty;
        }
        let mut ty = ty;
        self.normalize_scalar_type_inner(&mut ty);
        ty
    }
}
