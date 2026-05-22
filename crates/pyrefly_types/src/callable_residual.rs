/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use pyrefly_derive::TypeEq;
use pyrefly_derive::Visit;
use pyrefly_derive::VisitMut;
use starlark_map::small_set::SmallSet;

use crate::quantified::Quantified;
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

pub enum OverloadResidualIdentityAnalysis {
    None,
    Single {
        identity: OverloadResidualIdentity,
        branch_indices: Vec<usize>,
    },
    Multiple,
}

pub struct OverloadBranchSubstitutionResult {
    pub substituted: bool,
    pub marker_remaining: bool,
}

impl Type {
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

    /// Analyze overload residual markers in one pass.
    /// Returns:
    /// - `None`: no overload residual markers in this type
    /// - `Single`: exactly one witness identity with the branch-index intersection
    /// - `Multiple`: more than one witness identity appears in the same type
    pub fn analyze_overload_residual_identity(&self) -> OverloadResidualIdentityAnalysis {
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

    /// Substitute overload residual markers matching `identity` with the type
    /// from the branch at `branch_index`.
    pub fn substitute_overload_residual_identity_branch(
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
}
