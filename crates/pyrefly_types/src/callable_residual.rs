/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::sync::Arc;

use pyrefly_derive::TypeEq;
use pyrefly_derive::Visit;
use pyrefly_derive::VisitMut;
use pyrefly_util::visit::VisitMut;
use starlark_map::small_set::SmallSet;
use vec1::Vec1;

use crate::callable::Callable;
use crate::callable::FuncFlags;
use crate::callable::FuncMetadata;
use crate::callable::Function;
use crate::callable::FunctionKind;
use crate::callable::Params;
use crate::callable::PrefixParam;
use crate::heap::TypeHeap;
use crate::quantified::Quantified;
use crate::simplify::unions;
use crate::types::Forall;
use crate::types::Forallable;
use crate::types::Overload;
use crate::types::OverloadType;
use crate::types::TParams;
use crate::types::Type;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(Visit, VisitMut, TypeEq)]
pub enum CallableResidualKind {
    /// A generic residual. The `quantified` is the quantified type variable we
    /// want to use when reconstructing a `Callable` that contains a generic
    /// residual; we'll wrap it in a Forall that scopes all the residuals.
    ///
    /// If it appears anywhere else, the fallback is `quantified.as_gradual_type()`
    Generic { quantified: Quantified },
    /// Per-var overload residual with identity for cross-var correlation.
    ///
    /// Finishing normalizes branch types so an overload residual does not
    /// contain nested overload residual markers in `branches[*].ty`.
    Overload {
        identity: OverloadResidualIdentity,
        branches: Vec<OverloadBranchProjection>,
    },
}

/// Correlation key for matching overload residuals across vars during finalization.
/// The hash is derived from the got-side type at the comparison that produced the
/// residual, making it a stable function of the value rather than of solve order.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(Visit, VisitMut, TypeEq)]
pub struct OverloadResidualIdentity {
    pub witness_hash: u64,
}

/// Per-branch result for a single var in an overload residual.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(Visit, VisitMut, TypeEq)]
pub struct OverloadBranchProjection {
    pub branch_index: usize,
    pub ty: Type,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(Visit, VisitMut, TypeEq)]
pub struct CallableResidual {
    pub kind: CallableResidualKind,
}

impl CallableResidual {
    /// Canonical fallback policy for residual elimination outside callable-preserving paths.
    fn fallback_type(&self, heap: &TypeHeap) -> Type {
        match &self.kind {
            CallableResidualKind::Generic { quantified } => quantified.as_gradual_type(),
            CallableResidualKind::Overload { branches, .. } => unions(
                branches
                    .iter()
                    .map(|branch| branch.ty.clone())
                    .collect::<Vec<_>>(),
                heap,
            ),
        }
    }
}

enum OverloadResidualIdentityAnalysis {
    None,
    Single {
        identity: OverloadResidualIdentity,
        branch_indices: Vec<usize>,
    },
    Multiple,
}

struct OverloadBranchSubstitutionResult {
    substituted: bool,
    marker_remaining: bool,
}

impl Type {
    pub fn callable_residual_generic(quantified: Quantified) -> Type {
        Type::CallableResidual(Box::new(CallableResidual {
            kind: CallableResidualKind::Generic { quantified },
        }))
    }

    pub fn callable_residual_overload(
        identity: OverloadResidualIdentity,
        branches: Vec<OverloadBranchProjection>,
    ) -> Type {
        Type::CallableResidual(Box::new(CallableResidual {
            kind: CallableResidualKind::Overload { identity, branches },
        }))
    }

    /// Check if the type contains an overload callable residual marker anywhere.
    pub fn contains_overload_callable_residual(&self) -> bool {
        self.any(|inner| {
            matches!(
                inner,
                Type::CallableResidual(residual)
                    if matches!(&residual.kind, CallableResidualKind::Overload { .. })
            )
        })
    }

    /// Replace all callable residual markers with their fallback types.
    pub fn flatten_residuals(mut self, heap: &TypeHeap) -> Type {
        self.transform_mut(&mut |inner| {
            if let Type::CallableResidual(residual) = inner {
                *inner = residual.fallback_type(heap);
            }
        });
        self
    }

    /// Finishing invariant: overload residual branch types must not contain
    /// overload residual markers. This keeps boundary finalization to one
    /// overload pass followed by one generic pass.
    pub fn flatten_overload_residual_markers(&mut self, heap: &TypeHeap) {
        self.transform_mut(&mut |inner| {
            if let Type::CallableResidual(residual) = inner
                && matches!(&residual.kind, CallableResidualKind::Overload { .. })
            {
                *inner = residual.fallback_type(heap);
            }
        });
    }

    /// Analyze overload residual markers in one pass.
    /// Returns:
    /// - `None`: no overload residual markers in this type
    /// - `Single`: exactly one witness identity with the branch-index intersection
    /// - `Multiple`: more than one witness identity appears in the same type
    fn analyze_overload_residual_identity(&self) -> OverloadResidualIdentityAnalysis {
        let mut first: Option<OverloadResidualIdentity> = None;
        let mut intersection: Option<SmallSet<usize>> = None;
        let mut conflict = false;
        self.universe(&mut |inner| {
            if conflict {
                return;
            }
            if let Type::CallableResidual(residual) = inner
                && let CallableResidualKind::Overload { identity, branches } = &residual.kind
            {
                let branch_indices: SmallSet<usize> =
                    branches.iter().map(|branch| branch.branch_index).collect();
                match &first {
                    None => {
                        first = Some(identity.clone());
                        intersection = Some(branch_indices);
                    }
                    Some(existing) if existing == identity => {
                        let current = intersection.take().expect(
                            "matching overload residual identity must have intersection state",
                        );
                        intersection = Some(
                            current
                                .into_iter()
                                .filter(|idx| branch_indices.contains(idx))
                                .collect(),
                        );
                    }
                    Some(_) => conflict = true,
                }
            }
        });
        if conflict {
            OverloadResidualIdentityAnalysis::Multiple
        } else if let Some(identity) = first {
            let mut branch_indices = intersection
                .expect("matching overload residual identity must produce an intersection set")
                .into_iter()
                .collect::<Vec<_>>();
            branch_indices.sort_unstable();
            OverloadResidualIdentityAnalysis::Single {
                identity,
                branch_indices,
            }
        } else {
            OverloadResidualIdentityAnalysis::None
        }
    }

    /// Strip all overload residual markers with the given identity, replacing
    /// them with their fallback types.
    fn strip_overload_residual_identity(
        &mut self,
        identity: &OverloadResidualIdentity,
        heap: &TypeHeap,
    ) {
        self.transform_mut(&mut |inner| {
            if let Type::CallableResidual(residual) = inner
                && let CallableResidualKind::Overload {
                    identity: marker_identity,
                    ..
                } = &residual.kind
                && marker_identity == identity
            {
                *inner = residual.fallback_type(heap);
            }
        });
    }

    /// Substitute overload residual markers matching `identity` with the type
    /// from the branch at `branch_index`.
    fn substitute_overload_residual_identity_branch(
        &mut self,
        identity: &OverloadResidualIdentity,
        branch_index: usize,
    ) -> OverloadBranchSubstitutionResult {
        let mut substituted = false;
        let mut marker_remaining = false;
        self.transform_mut(&mut |inner| {
            if let Type::CallableResidual(residual) = inner
                && let CallableResidualKind::Overload {
                    identity: marker_identity,
                    branches,
                } = &residual.kind
                && marker_identity == identity
            {
                let branch = branches
                    .iter()
                    .find(|branch| branch.branch_index == branch_index)
                    .expect("selected overload branch index must exist on every matching marker");
                let branch_ty = &branch.ty;
                let branch_contains_identity = branch_ty.any(|candidate| {
                    if let Type::CallableResidual(candidate_residual) = candidate
                        && let CallableResidualKind::Overload {
                            identity: candidate_identity,
                            ..
                        } = &candidate_residual.kind
                    {
                        candidate_identity == identity
                    } else {
                        false
                    }
                });
                marker_remaining |= branch_contains_identity;
                *inner = branch_ty.clone();
                substituted = true;
            }
        });
        OverloadBranchSubstitutionResult {
            substituted,
            marker_remaining,
        }
    }

    /// Try to reconstruct an overloaded type from per-branch finalized types.
    ///
    /// Returns `None` if any branch type cannot be converted to an overload signature.
    fn try_combine_reconstructed_overload(&self, reconstructed: &[Type]) -> Option<Type> {
        let metadata = self
            .visit_toplevel_func_metadata(&|metadata| Some(metadata.clone()))
            .unwrap_or(FuncMetadata {
                kind: FunctionKind::Overload,
                flags: FuncFlags::default(),
            });
        let signatures = reconstructed
            .iter()
            .cloned()
            .map(|branch_ty| branch_ty.into_overload_signature(&metadata))
            .collect::<Option<Vec<_>>>()?;
        let signatures = Vec1::try_from_vec(signatures).ok()?;
        Some(Type::Overload(Overload {
            signatures,
            metadata: Box::new(metadata),
        }))
    }

    fn into_overload_signature(self, metadata: &FuncMetadata) -> Option<OverloadType> {
        match self {
            Type::Function(function) => Some(OverloadType::Function(*function)),
            Type::Forall(forall) => match forall.body {
                Forallable::Function(function) => Some(OverloadType::Forall(Forall {
                    tparams: forall.tparams,
                    body: function,
                })),
                Forallable::Callable(callable) => Some(OverloadType::Forall(Forall {
                    tparams: forall.tparams,
                    body: Function {
                        signature: callable,
                        metadata: metadata.clone(),
                    },
                })),
                Forallable::TypeAlias(_) => None,
            },
            Type::Callable(callable) => Some(OverloadType::Function(Function {
                signature: *callable,
                metadata: metadata.clone(),
            })),
            _ => None,
        }
    }

    /// Collect quantified type parameters for wrapping in a Forall.
    fn quantified_tparams_for_forall(&self, heap: &TypeHeap) -> Arc<TParams> {
        let callable_ty = match self {
            Type::Callable(c) => heap.mk_callable(c.params.clone(), c.ret.clone()),
            Type::Function(f) => {
                heap.mk_callable(f.signature.params.clone(), f.signature.ret.clone())
            }
            _ => self.clone(),
        };
        let mut tparams = Vec::new();
        callable_ty.for_each_quantified(&mut |q| tparams.push(q.clone()));
        tparams.sort();
        tparams.dedup();
        Arc::new(TParams::new(tparams))
    }

    /// Finalize callable residuals at a boundary with one outer traversal.
    ///
    /// Non-callable structure is traversed once. Callable/function subtrees run
    /// two phases (overload then generic) internally.
    fn finalize_callable_residuals_mut(
        &mut self,
        heap: &TypeHeap,
        callable_slot: bool,
        preserve_class_targs: bool,
    ) -> (bool, bool) {
        match self {
            Type::CallableResidual(residual) => match &residual.kind {
                CallableResidualKind::Generic { quantified } => {
                    if !callable_slot {
                        *self = residual.fallback_type(heap);
                        return (true, false);
                    }
                    *self = heap.mk_quantified(quantified.clone());
                    (true, true)
                }
                CallableResidualKind::Overload { .. } => {
                    *self = residual.fallback_type(heap);
                    let (_nested_changed, nested_consumed) = self.finalize_callable_residuals_mut(
                        heap,
                        callable_slot,
                        preserve_class_targs,
                    );
                    (true, nested_consumed)
                }
            },
            Type::Callable(callable) => {
                // NOTE: This loop is intentionally duplicated in the Type::Function
                // arm below. The phase ordering and accumulation logic are identical,
                // but extracting a shared higher-order helper adds closure/generic
                // indirection without a clear zero-cost win here.
                let mut changed = false;
                let mut consumed_residual = false;
                for phase in [
                    CallableResidualFinalizePhase::Overload,
                    CallableResidualFinalizePhase::Generic,
                ] {
                    let (phase_changed, phase_consumed) =
                        callable.finalize_residuals_in_phase_mut(heap, preserve_class_targs, phase);
                    changed |= phase_changed;
                    consumed_residual |= phase_consumed;
                }
                if consumed_residual && !callable_slot {
                    let tparams = self.quantified_tparams_for_forall(heap);
                    if let Type::Callable(c) = std::mem::replace(self, Type::None) {
                        *self = Forallable::Callable(*c).forall(tparams);
                    }
                    (true, true)
                } else {
                    (changed, consumed_residual)
                }
            }
            Type::Function(function) => {
                // Intentionally kept in lockstep with the Type::Callable arm above.
                let mut changed = false;
                let mut consumed_residual = false;
                for phase in [
                    CallableResidualFinalizePhase::Overload,
                    CallableResidualFinalizePhase::Generic,
                ] {
                    let (phase_changed, phase_consumed) =
                        function.finalize_residuals_in_phase_mut(heap, preserve_class_targs, phase);
                    changed |= phase_changed;
                    consumed_residual |= phase_consumed;
                }
                if !changed {
                    return (false, false);
                }
                if consumed_residual && !callable_slot {
                    let tparams = self.quantified_tparams_for_forall(heap);
                    if let Type::Function(f) = std::mem::replace(self, Type::None) {
                        *self = Forallable::Function(*f).forall(tparams);
                    }
                    (true, true)
                } else {
                    (true, consumed_residual)
                }
            }
            Type::ClassType(_) if preserve_class_targs && !callable_slot => (false, false),
            _ => {
                let mut changed = false;
                let mut consumed_residual = false;
                self.recurse_mut(&mut |inner| {
                    let (inner_changed, inner_consumed) = inner.finalize_callable_residuals_mut(
                        heap,
                        callable_slot,
                        preserve_class_targs,
                    );
                    changed |= inner_changed;
                    consumed_residual |= inner_consumed;
                });
                (changed, consumed_residual)
            }
        }
    }

    /// Finalize callable residuals at a substitution boundary (a return type or class field).
    ///
    /// We run overload handling first and generic handling second.
    pub fn finalize_callable_residuals_at_boundary(
        self,
        heap: &TypeHeap,
        preserve_class_targs: bool,
    ) -> Type {
        let mut active_overload_identities = Vec::new();
        self.finalize_callable_residuals_at_boundary_impl(
            heap,
            preserve_class_targs,
            &mut active_overload_identities,
        )
    }

    fn finalize_callable_residuals_at_boundary_impl(
        mut self,
        heap: &TypeHeap,
        preserve_class_targs: bool,
        active_overload_identities: &mut Vec<OverloadResidualIdentity>,
    ) -> Type {
        // Overload reconstruction is only for callable roots. For non-callable
        // roots, keep the outer structure and rely on inline residual fallback.
        let callable_root = self.is_toplevel_callable();

        // Reconstruction is only safe when all overload residual markers in
        // the type share a single witness identity. Multiple witnesses would
        // produce a cross-product explosion; strip all markers and fall
        // through to generic residual finalization instead.
        if callable_root
            && let OverloadResidualIdentityAnalysis::Single {
                identity,
                branch_indices,
            } = self.analyze_overload_residual_identity()
        {
            if active_overload_identities.contains(&identity) {
                unreachable!(
                    "detected recursive overload residual identity cycle during finalization",
                );
            }
            if branch_indices.is_empty() {
                // TODO(T235420905): This path is intended to be unreachable for coherent
                // witnesses, but we have seen CI panic signatures around this stack. Audit
                // the top-of-stack identity/branch-coherence invariant and decide whether
                // to harden this boundary or tighten upstream residual capture.
                self.strip_overload_residual_identity(&identity, heap);
            } else {
                active_overload_identities.push(identity.clone());
                let reconstructed: Vec<Type> = branch_indices
                    .into_iter()
                    .map(|branch_index| {
                        let mut branch_ty = self.clone();
                        let substitution_result = branch_ty
                            .substitute_overload_residual_identity_branch(
                                &identity,
                                branch_index,
                            );
                        if !substitution_result.substituted {
                            unreachable!(
                                "selected overload residual identity must be present during reconstruction",
                            );
                        }
                        if substitution_result.marker_remaining {
                            unreachable!(
                                "overload residual substitution did not eliminate active identity"
                            );
                        }
                        branch_ty.finalize_callable_residuals_at_boundary_impl(
                            heap,
                            preserve_class_targs,
                            active_overload_identities,
                        )
                    })
                    .collect();
                let popped = active_overload_identities
                    .pop()
                    .expect("active_overload_identities push/pop must stay balanced");
                debug_assert_eq!(popped, identity);
                if let Some(overload_ty) = self.try_combine_reconstructed_overload(&reconstructed) {
                    self = overload_ty;
                } else {
                    // This fallback only applies to callable roots. Non-callable
                    // roots bypass reconstruction and rely on inline fallback.
                    self = unions(reconstructed, heap);
                }
            }
        }

        self.finalize_callable_residuals_mut(heap, false, preserve_class_targs);
        self
    }

    fn finalize_callable_residuals_in_phase_mut(
        &mut self,
        heap: &TypeHeap,
        callable_slot: bool,
        preserve_class_targs: bool,
        phase: CallableResidualFinalizePhase,
    ) -> (bool, bool) {
        match self {
            Type::CallableResidual(residual) => match &residual.kind {
                CallableResidualKind::Generic { quantified } => {
                    if phase != CallableResidualFinalizePhase::Generic {
                        return (false, false);
                    }
                    if !callable_slot {
                        *self = residual.fallback_type(heap);
                        return (true, false);
                    }
                    *self = heap.mk_quantified(quantified.clone());
                    (true, true)
                }
                CallableResidualKind::Overload { .. } => {
                    if phase != CallableResidualFinalizePhase::Overload {
                        return (false, false);
                    }
                    *self = residual.fallback_type(heap);
                    (true, false)
                }
            },
            Type::Callable(callable) => {
                callable.finalize_residuals_in_phase_mut(heap, preserve_class_targs, phase)
            }
            Type::Function(function) => {
                function.finalize_residuals_in_phase_mut(heap, preserve_class_targs, phase)
            }
            Type::ClassType(_) if preserve_class_targs && !callable_slot => (false, false),
            _ => {
                let mut changed = false;
                let mut consumed_residual = false;
                self.recurse_mut(&mut |inner| {
                    let (inner_changed, inner_consumed) = inner
                        .finalize_callable_residuals_in_phase_mut(
                            heap,
                            callable_slot,
                            preserve_class_targs,
                            phase,
                        );
                    changed |= inner_changed;
                    consumed_residual |= inner_consumed;
                });
                (changed, consumed_residual)
            }
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum CallableResidualFinalizePhase {
    Overload,
    Generic,
}

impl Callable {
    fn finalize_residuals_in_phase_mut(
        &mut self,
        heap: &TypeHeap,
        preserve_class_targs: bool,
        phase: CallableResidualFinalizePhase,
    ) -> (bool, bool) {
        let mut changed = false;
        let mut consumed_residual = false;
        match &mut self.params {
            Params::List(params) => {
                for param in params.items_mut() {
                    let (param_changed, param_consumed) = param
                        .as_type_mut()
                        .finalize_callable_residuals_in_phase_mut(
                            heap,
                            true,
                            preserve_class_targs,
                            phase,
                        );
                    changed |= param_changed;
                    consumed_residual |= param_consumed;
                }
            }
            Params::ParamSpec(prefix, p) => {
                for prefix_param in prefix.iter_mut() {
                    let prefix_ty = match prefix_param {
                        PrefixParam::PosOnly(_, ty, _) | PrefixParam::Pos(_, ty, _) => ty,
                    };
                    let (prefix_changed, prefix_consumed) = prefix_ty
                        .finalize_callable_residuals_in_phase_mut(
                            heap,
                            true,
                            preserve_class_targs,
                            phase,
                        );
                    changed |= prefix_changed;
                    consumed_residual |= prefix_consumed;
                }
                let (paramspec_changed, paramspec_consumed) = p
                    .finalize_callable_residuals_in_phase_mut(
                        heap,
                        true,
                        preserve_class_targs,
                        phase,
                    );
                changed |= paramspec_changed;
                consumed_residual |= paramspec_consumed;
            }
            Params::Ellipsis | Params::Materialization => {}
        }
        let (ret_changed, ret_consumed) = self.ret.finalize_callable_residuals_in_phase_mut(
            heap,
            true,
            preserve_class_targs,
            phase,
        );
        changed |= ret_changed;
        consumed_residual |= ret_consumed;
        (changed, consumed_residual)
    }
}

impl Function {
    fn finalize_residuals_in_phase_mut(
        &mut self,
        heap: &TypeHeap,
        preserve_class_targs: bool,
        phase: CallableResidualFinalizePhase,
    ) -> (bool, bool) {
        if !self.signature.contains_callable_residual() {
            return (false, false);
        }
        let mut signature = self.signature.clone();
        let (changed, consumed_residual) =
            signature.finalize_residuals_in_phase_mut(heap, preserve_class_targs, phase);
        if !changed {
            return (false, false);
        }
        self.signature = signature;
        (true, consumed_residual)
    }
}
