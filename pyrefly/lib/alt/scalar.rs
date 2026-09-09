/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Solver integration for `shape_extensions.Scalar`.

use pyrefly_types::heap::TypeHeap;
use pyrefly_types::shaped_array::IntTuple;
use pyrefly_types::shaped_array::IntTupleView;
use pyrefly_types::shaped_array::tuple_carrier_to_shape;
use pyrefly_types::types::Type;
use pyrefly_util::visit::VisitMut;

use crate::alt::answers::LookupAnswer;
use crate::alt::answers_solver::AnswersSolver;
use crate::solver::solver::Solver;
use crate::solver::solver::Subset;
use crate::solver::solver::SubsetError;

/// The shape witness and scalar domain extracted from `Scalar[Shape, Domain]`.
struct ScalarFamily {
    shape: Type,
    domain: Type,
}

/// The semantic form of a `Scalar` after its shape has been normalized.
enum ScalarNormalForm {
    /// A rank-zero or gradual shape exposes the scalar domain.
    Domain(Type),
    /// The shape is not yet known, so its relationship with the domain must be preserved.
    Suspended(ScalarFamily),
}

fn scalar(ty: &Type) -> Option<ScalarFamily> {
    let Type::ClassType(cls) = ty else {
        return None;
    };
    if cls.has_qname("shape_extensions", "Scalar") {
        let [shape, domain] = cls.targs().as_slice() else {
            return None;
        };
        Some(ScalarFamily {
            shape: shape.clone(),
            domain: domain.clone(),
        })
    } else {
        None
    }
}

fn scalar_normal_form(heap: &TypeHeap, marker: ScalarFamily) -> ScalarNormalForm {
    let shape = match &marker.shape {
        Type::Any(_) => return ScalarNormalForm::Domain(marker.domain),
        Type::IntTuple(shape) => shape.normalize(),
        shape => match tuple_carrier_to_shape(shape) {
            Some(shape) => shape.normalize(),
            // Unresolved or invalid shapes retain the marker. Validation reports malformed
            // specializations, while inference may later solve suspended variables.
            None => return ScalarNormalForm::Suspended(marker),
        },
    };
    match shape.view() {
        IntTupleView::Concrete([]) => ScalarNormalForm::Domain(marker.domain),
        IntTupleView::Concrete(_) => ScalarNormalForm::Domain(heap.mk_never()),
        IntTupleView::Gradual => ScalarNormalForm::Domain(marker.domain),
        IntTupleView::Unpacked { prefix, suffix, .. }
            if !prefix.is_empty() || !suffix.is_empty() =>
        {
            ScalarNormalForm::Domain(heap.mk_never())
        }
        IntTupleView::Unpacked { .. } => ScalarNormalForm::Suspended(marker),
    }
}

/// Whether `ty` contains a `Scalar` marker that normalization may replace.
fn contains_scalar(ty: &Type) -> bool {
    ty.any(|candidate| scalar(candidate).is_some())
}

fn contains_var(ty: &Type) -> bool {
    ty.any(|candidate| matches!(candidate, Type::Var(_)))
}

fn expanded_scalar_normal_form(solver: &Solver, ty: &Type) -> Option<ScalarNormalForm> {
    let mut marker = scalar(ty)?;
    if contains_var(&marker.shape) {
        solver.expand_with_bounds(&mut marker.shape);
    }
    Some(scalar_normal_form(&solver.heap, marker))
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

impl<'solver, 'subset, Ans: LookupAnswer> Subset<'solver, 'subset, Ans> {
    fn is_subset_domain_to_suspended(
        &mut self,
        got: &Type,
        want: &ScalarFamily,
    ) -> Result<(), SubsetError> {
        self.with_speculative_subset_branch_result(&[got, &want.domain, &want.shape], |me| {
            me.is_subset_eq(got, &want.domain)?;
            me.is_subset_eq(&IntTuple::new(Vec::new()).to_shape_arg_type(), &want.shape)
        })
    }

    /// Handle a subset relation involving `Scalar`, or delegate by returning `None`.
    ///
    /// Successful speculative branches retain inferred variable constraints. Failed branches and
    /// probes restore all subset state captured by the transaction helper.
    pub(crate) fn is_subset_scalar(
        &mut self,
        got: &Type,
        want: &Type,
    ) -> Option<Result<(), SubsetError>> {
        if !self.solver.tensor_shapes {
            return None;
        }
        if got.is_any() || want.is_any() {
            return None;
        }
        let got_scalar = expanded_scalar_normal_form(self.solver, got);
        let want_scalar = expanded_scalar_normal_form(self.solver, want);
        if got.is_never() && want_scalar.is_some() {
            return Some(Ok(()));
        }
        match (got_scalar, want_scalar) {
            (None, None) => None,
            (Some(ScalarNormalForm::Domain(domain)), None) => {
                Some(self.is_subset_eq(&domain, want))
            }
            (Some(ScalarNormalForm::Suspended(got_marker)), None) => {
                if let Type::Union(union) = want {
                    for member in union
                        .members
                        .iter()
                        .filter(|member| scalar(member).is_some())
                    {
                        if self
                            .with_speculative_subset_branch_result(
                                &[got, &got_marker.shape, &got_marker.domain, member],
                                |me| me.is_subset_eq(got, member),
                            )
                            .is_ok()
                        {
                            return Some(Ok(()));
                        }
                    }
                }
                Some(self.is_subset_eq(&got_marker.domain, want))
            }
            (None, Some(ScalarNormalForm::Domain(domain))) => Some(self.is_subset_eq(got, &domain)),
            (None, Some(ScalarNormalForm::Suspended(want))) => {
                Some(self.is_subset_domain_to_suspended(got, &want))
            }
            (
                Some(ScalarNormalForm::Domain(got_domain)),
                Some(ScalarNormalForm::Domain(want_domain)),
            ) => Some(self.is_subset_eq(&got_domain, &want_domain)),
            (
                Some(ScalarNormalForm::Domain(got_domain)),
                Some(ScalarNormalForm::Suspended(want_marker)),
            ) => Some(self.is_subset_domain_to_suspended(&got_domain, &want_marker)),
            (
                Some(ScalarNormalForm::Suspended(got_marker)),
                Some(ScalarNormalForm::Domain(want_domain)),
            ) => Some(self.is_subset_eq(&got_marker.domain, &want_domain)),
            (
                Some(ScalarNormalForm::Suspended(got_marker)),
                Some(ScalarNormalForm::Suspended(want_marker)),
            ) => {
                if contains_var(&got_marker.shape) || contains_var(&want_marker.shape) {
                    return Some(self.with_speculative_subset_branch_result(
                        &[
                            &got_marker.domain,
                            &got_marker.shape,
                            &want_marker.domain,
                            &want_marker.shape,
                        ],
                        |me| {
                            me.is_subset_eq(&got_marker.domain, &want_marker.domain)?;
                            me.is_subset_eq(&got_marker.shape, &want_marker.shape)?;
                            me.is_subset_eq(&want_marker.shape, &got_marker.shape)
                        },
                    ));
                }
                Some(
                    self.probe_speculative_subset_branch_result(
                        &[&got_marker.shape, &want_marker.shape],
                        |me| {
                            me.is_subset_eq(&got_marker.shape, &want_marker.shape)?;
                            me.is_subset_eq(&want_marker.shape, &got_marker.shape)
                        },
                    )
                    .and_then(|()| self.is_subset_eq(&got_marker.domain, &want_marker.domain)),
                )
            }
        }
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

    /// Replace resolved `Scalar` occurrences with their domain or `Never` normal form.
    ///
    /// Union members are normalized before their containing union so redundant members can be
    /// simplified together.
    pub(crate) fn normalize_scalar_type(&self, ty: Type) -> Type {
        if !self.solver().tensor_shapes || !contains_scalar(&ty) {
            return ty;
        }
        let mut ty = ty;
        self.normalize_scalar_type_inner(&mut ty);
        ty
    }
}
