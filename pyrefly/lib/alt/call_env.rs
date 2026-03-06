/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Correlated constrained-TypeVar branching for calls and operators.
//!
//! When types contain constrained `Type::Quantified` (e.g. `T: (int, float)`),
//! this module enumerates deterministic substitution environments (one per
//! cartesian-product element) and runs a caller-supplied closure under each
//! environment. Results are aggregated: if any environment succeeds the call
//! is accepted and the return type is the union of successful return types.

use pyrefly_types::quantified::Quantified;
use pyrefly_types::quantified::QuantifiedKind;
use pyrefly_util::visit::Visit;

use crate::types::type_var::Restriction;
use crate::types::types::Type;

/// Maximum number of environments to enumerate before falling back to
/// the legacy (non-correlated) behaviour.
const ENV_CAP: usize = 32;

/// A single substitution environment mapping constrained quantifieds
/// to one of their constraint types.
type Env = Vec<(Quantified, Type)>;

/// Collect all distinct constrained `Type::Quantified` that appear anywhere
/// inside the given seed types. Only `QuantifiedKind::TypeVar` with
/// `Restriction::Constraints` are collected.
fn collect_constrained_quantifieds(seeds: &[&Type]) -> Vec<(Quantified, Vec<Type>)> {
    let mut result: Vec<(Quantified, Vec<Type>)> = Vec::new();
    for seed in seeds {
        collect_from_type(seed, &mut result);
    }
    result
}

/// Recursively walk a type tree and collect constrained quantifieds.
fn collect_from_type(ty: &Type, acc: &mut Vec<(Quantified, Vec<Type>)>) {
    if let Type::Quantified(q) = ty
        && q.kind() == QuantifiedKind::TypeVar
        && let Restriction::Constraints(constraints) = q.restriction()
    {
        if !acc.iter().any(|(existing, _)| existing == q.as_ref()) {
            acc.push((q.as_ref().clone(), constraints.clone()));
        }
        // Don't recurse into the quantified's own restriction types.
        return;
    }
    ty.recurse(&mut |child: &Type| collect_from_type(child, acc));
}

/// Apply a substitution environment to a type, replacing every
/// `Type::Quantified(q)` whose identity matches an entry in `env`
/// with the corresponding concrete type.
pub fn subst_env(ty: &Type, env: &Env) -> Type {
    let mut result = ty.clone();
    result.subst_mut_fn(&mut |q| {
        env.iter()
            .find(|(eq, _)| eq == q)
            .map(|(_, replacement)| replacement.clone())
    });
    result
}

/// Enumerate the cartesian product of constraint lists for each
/// distinct constrained quantified. Returns `None` if no constrained
/// quantifieds are found or if the product exceeds `ENV_CAP`.
fn enumerate_envs(quantifieds: &[(Quantified, Vec<Type>)]) -> Option<Vec<Env>> {
    if quantifieds.is_empty() {
        return None;
    }
    let total: usize = quantifieds
        .iter()
        .map(|(_, cs)| cs.len())
        .try_fold(1usize, |acc, n| acc.checked_mul(n))
        .unwrap_or(usize::MAX);
    if total > ENV_CAP {
        return None;
    }
    let mut envs: Vec<Env> = vec![Vec::new()];
    for (q, constraints) in quantifieds {
        let mut new_envs = Vec::with_capacity(envs.len() * constraints.len());
        for env in &envs {
            for constraint in constraints {
                let mut extended = env.clone();
                extended.push((q.clone(), constraint.clone()));
                new_envs.push(extended);
            }
        }
        envs = new_envs;
    }
    Some(envs)
}

/// The outcome of running a single environment trial.
pub struct EnvTrialResult {
    /// The inferred return type from this trial.
    pub ret: Type,
    /// Whether this trial had call-level errors.
    pub has_errors: bool,
    /// Number of errors produced during this trial.
    pub error_count: usize,
}

/// Result of running all environment trials.
pub enum ConstrainedEnvResult {
    /// At least one environment succeeded (no errors).
    /// Contains the list of successful return types.
    Success(Vec<Type>),
    /// All environments failed.
    /// Contains the index of the best (fewest errors) environment.
    AllFailed(usize),
}

/// Run a closure under each constrained-TypeVar substitution environment,
/// aggregating results.
///
/// Returns `None` if the seed types contain no constrained quantifieds or
/// the environment count exceeds the cap, signalling that the caller should
/// fall back to the existing (legacy) behaviour.
///
/// The `run_trial` closure receives substituted seed types and must return
/// an `EnvTrialResult` describing the outcome.
///
/// `seed_types` are the types to scan for constrained quantifieds and to
/// substitute in each trial.
pub fn run_with_constrained_envs(
    seed_types: &[&Type],
    run_trial: &mut dyn FnMut(&[Type]) -> EnvTrialResult,
) -> Option<ConstrainedEnvResult> {
    let quantifieds = collect_constrained_quantifieds(seed_types);
    let envs = enumerate_envs(&quantifieds)?;

    let mut successful_returns: Vec<Type> = Vec::new();
    let mut best_failed_idx: usize = 0;
    let mut best_failed_error_count: usize = usize::MAX;
    let mut any_succeeded = false;

    for (idx, env) in envs.iter().enumerate() {
        let substituted: Vec<Type> = seed_types.iter().map(|ty| subst_env(ty, env)).collect();
        let trial = run_trial(&substituted);
        if !trial.has_errors {
            any_succeeded = true;
            successful_returns.push(trial.ret);
        } else if trial.error_count < best_failed_error_count {
            best_failed_error_count = trial.error_count;
            best_failed_idx = idx;
        }
    }

    if any_succeeded {
        Some(ConstrainedEnvResult::Success(successful_returns))
    } else {
        Some(ConstrainedEnvResult::AllFailed(best_failed_idx))
    }
}
