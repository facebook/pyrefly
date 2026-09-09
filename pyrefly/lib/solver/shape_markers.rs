/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Subset logic for the `shape_extensions` markers `Scalar` and `ArrayCoercible`.
//!
//! The generic solver remains responsible for variable state and subset traversal. This module
//! isolates the shape-specific subset decisions it needs, keeping them out of `alt/`, which
//! retains only the `AnswersSolver` work: literal projection, scalar normalization, and
//! redundant-union simplification.

use pyrefly_types::class::ClassType;
use pyrefly_types::heap::TypeHeap;
use pyrefly_types::shaped_array::IntTuple;
use pyrefly_types::shaped_array::IntTupleView;
use pyrefly_types::shaped_array::tuple_carrier_to_shape;
use pyrefly_types::type_var::Restriction;
use pyrefly_types::types::Type;

use crate::alt::answers::LookupAnswer;
use crate::solver::shape::has_int_tuple_bound;
use crate::solver::solver::Solver;
use crate::solver::solver::Subset;
use crate::solver::solver::SubsetError;
use crate::solver::type_order::TypeOrder;

/// The shape and element domain extracted from `ArrayCoercible[Shape, Domain]`.
///
/// The original two-argument class type is retained so literal projection can replace only its
/// shape argument without reconstructing the marker.
pub(crate) struct ArrayCoercible {
    /// The shape accepted or inferred for the value.
    pub(crate) shape: Type,
    /// The type accepted for every scalar leaf.
    pub(crate) domain: Type,
    class_type: ClassType,
}

/// Extract an `ArrayCoercible` marker from its class-type representation.
pub(crate) fn array_coercible(ty: &Type) -> Option<ArrayCoercible> {
    let Type::ClassType(cls) = ty else {
        return None;
    };
    if !cls.has_qname("shape_extensions", "ArrayCoercible") {
        return None;
    }
    let [shape, domain] = cls.targs().as_slice() else {
        return None;
    };
    Some(ArrayCoercible {
        shape: shape.clone(),
        domain: domain.clone(),
        class_type: cls.clone(),
    })
}

impl<Ans: LookupAnswer> TypeOrder<'_, Ans> {
    /// Whether a value may contain array structure that cannot be projected from its expression.
    ///
    /// False positives deliberately fall back to ordinary typing rather than treating an existing
    /// container as one scalar leaf. Strings and bytes are leaves despite being sequences.
    pub(crate) fn is_deferred_array_coercible_container(self, ty: &Type) -> bool {
        match ty {
            Type::Var(_) => {
                let mut expanded = ty.clone();
                self.expand_with_bounds(&mut expanded);
                // An unresolved variable may still solve to a container, so treat it as
                // deferred rather than pinning a scalar shape for it.
                matches!(expanded, Type::Var(_))
                    || self.is_deferred_array_coercible_container(&expanded)
            }
            Type::Tuple(_) | Type::ShapedArray(_) => true,
            Type::Union(union) => union
                .members
                .iter()
                .any(|member| self.is_deferred_array_coercible_container(member)),
            Type::Intersect(intersect) => {
                intersect
                    .0
                    .iter()
                    .any(|member| self.is_deferred_array_coercible_container(member))
                    || self.is_deferred_array_coercible_container(&intersect.1)
            }
            Type::Quantified(quantified) => {
                self.is_deferred_array_coercible_restriction(quantified.restriction())
            }
            Type::TypeVar(type_var) => {
                self.is_deferred_array_coercible_restriction(type_var.restriction())
            }
            _ if array_coercible(ty).is_some() => true,
            _ if scalar(ty).is_some() => false,
            Type::ClassType(cls) | Type::SelfType(cls) => {
                // `has_superclass` includes identity, so subclasses of leaves and
                // sequences are covered without separate equality checks.
                if self.has_superclass(cls.class_object(), self.stdlib().str().class_object())
                    || self.has_superclass(cls.class_object(), self.stdlib().bytes().class_object())
                {
                    return false;
                }
                self.shaped_array_shape_for_class_type(cls).is_some()
                    || cls.tparams().iter().any(has_int_tuple_bound)
                    || self.has_superclass(cls.class_object(), self.stdlib().list_object())
                    || self.has_superclass(cls.class_object(), self.stdlib().tuple_object())
                    || self.has_superclass(
                        cls.class_object(),
                        self.stdlib().sequence(Type::any_implicit()).class_object(),
                    )
            }
            _ => false,
        }
    }

    /// Whether a type-variable restriction permits array structure.
    fn is_deferred_array_coercible_restriction(self, restriction: &Restriction) -> bool {
        match restriction {
            Restriction::Bound(bound) => self.is_deferred_array_coercible_container(bound),
            Restriction::Constraints(constraints) => constraints
                .iter()
                .any(|constraint| self.is_deferred_array_coercible_container(constraint)),
            Restriction::Unrestricted => true,
            Restriction::ShapeExtension(_) => false,
        }
    }
}

impl ArrayCoercible {
    /// Replace the marker's shape argument while preserving its domain and class metadata.
    pub(crate) fn with_shape(&self, shape: IntTuple) -> Type {
        let mut class_type = self.class_type.clone();
        class_type.targs_mut().as_mut()[0] = shape.to_shape_arg_type();
        class_type.to_type()
    }
}

impl<'solver, 'subset, Ans: LookupAnswer> Subset<'solver, 'subset, Ans> {
    /// Handle a subset relation involving `ArrayCoercible`, or delegate by returning `None`.
    ///
    /// Left-hand unions are delegated so generic union distribution can check each member.
    pub(crate) fn is_subset_array_coercible(
        &mut self,
        got: &Type,
        want: &Type,
    ) -> Option<Result<(), SubsetError>> {
        if !self.solver.tensor_shapes {
            return None;
        }
        let got_marker = array_coercible(got);
        let want_marker = array_coercible(want);
        if got.is_never() && want_marker.is_some() {
            return Some(Ok(()));
        }
        if got.is_any() || want.is_any() || matches!(got, Type::Union(_)) {
            return None;
        }
        match (got_marker, want_marker) {
            (None, None) => None,
            (Some(_), None) => None,
            (Some(got), Some(want)) => Some(self.with_speculative_subset_branch_result(
                &[&got.domain, &got.shape, &want.domain, &want.shape],
                |me| {
                    me.is_subset_eq(&got.domain, &want.domain)?;
                    me.is_subset_eq(&got.shape, &want.shape)
                },
            )),
            (None, Some(_)) if self.type_order.is_deferred_array_coercible_container(got) => None,
            (None, Some(want)) => Some(self.with_speculative_subset_branch_result(
                &[got, &want.domain, &want.shape],
                |me| {
                    me.is_subset_eq(got, &want.domain)?;
                    me.is_subset_eq(&IntTuple::new(Vec::new()).to_shape_arg_type(), &want.shape)
                },
            )),
        }
    }
}

/// The shape witness and scalar domain extracted from `Scalar[Shape, Domain]`.
pub(crate) struct ScalarFamily {
    /// The shape accepted for the scalar value.
    pub(crate) shape: Type,
    /// The type accepted for the scalar value.
    pub(crate) domain: Type,
}

/// The semantic form of a `Scalar` after its shape has been normalized.
pub(crate) enum ScalarNormalForm {
    /// A rank-zero or gradual shape exposes the scalar domain.
    Domain(Type),
    /// The shape is not yet known, so its relationship with the domain must be preserved.
    Suspended(ScalarFamily),
}

/// Extract a `Scalar` marker from its class-type representation.
pub(crate) fn scalar(ty: &Type) -> Option<ScalarFamily> {
    let Type::ClassType(cls) = ty else {
        return None;
    };
    if cls.has_qname("shape_extensions", "Scalar") {
        // Qualified names are user input: a user module can shadow `Scalar` with a different
        // arity, and malformed specializations are reachable during error recovery, so a
        // length mismatch degrades to "not a Scalar" instead of panicking.
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

fn contains_var(ty: &Type) -> bool {
    ty.any(|candidate| matches!(candidate, Type::Var(_)))
}

/// Expand inference variables in a `Scalar` shape before normalizing it.
pub(crate) fn expanded_scalar_normal_form(solver: &Solver, ty: &Type) -> Option<ScalarNormalForm> {
    let mut marker = scalar(ty)?;
    if contains_var(&marker.shape) {
        solver.expand_with_bounds(&mut marker.shape);
    }
    Some(scalar_normal_form(&solver.heap, marker))
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
