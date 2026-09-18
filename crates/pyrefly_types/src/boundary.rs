/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Helpers for finishing a type at a boundary. A boundary is usually the end of a function call.
//!
//! Two kinds of types need finishing:
//! * Free quantifieds. A generic higher-order function call might return a type that itself should
//!   generic. The unfinished type contains `Type::Quantified`s whose scopes have not yet been
//!   determined; `finalize_free_quantifieds` walks the type, determines the appropriate scopes,
//!   and binds the quantifieds in them.
//! * Overloaded return types. When a generic higher-order function is passed an overloaded
//!   callable, the higher-order function's return type might capture overloaded structure from the
//!   argument. `combine_overload_results` takes return types that have been determined from
//!   individual branches of the overloaded callable and combines them into an overloaded type.
//!
//! See also:
//! * `Subset::is_subset_eq_impl`, which records information from generic arguments and overload
//!   branches involved in an assignability check in a function call.
//! * `Solver::finish_quantified_with_pruning`, which reads this information to create unfinished
//!   quantifieds and overload tables describing solutions that need combining.
//! * `AnswersSolver::finish_return`, which calls `boundary.rs`'s finishing helpers to finish a
//!   function call's returned type.

use std::sync::Arc;

use pyrefly_util::visit::Visit;
use pyrefly_util::visit::VisitMut;
use starlark_map::small_set::SmallSet;
use vec1::Vec1;

use crate::callable::Callable;
use crate::function::FuncFlags;
use crate::function::FuncMetadata;
use crate::function::Function;
use crate::function::FunctionKind;
use crate::heap::TypeHeap;
use crate::quantified::Quantified;
use crate::simplify::unions;
use crate::types::Forall;
use crate::types::Forallable;
use crate::types::Overload;
use crate::types::OverloadType;
use crate::types::TParams;
use crate::types::Type;

impl Type {
    /// Scope each free quantified to the outermost callable whose parameter types contain it.
    /// Replace quantifieds with no possible scope with their gradual fallback.
    pub fn finalize_free_quantifieds(mut self) -> Type {
        self.finalize_free_quantifieds_mut();
        self
    }

    fn finalize_free_quantifieds_mut(&mut self) {
        fn go(ty: &mut Type, in_scope: &mut Vec<Quantified>) {
            if let Type::Quantified(q) = ty {
                if q.needs_finalization {
                    if in_scope.contains(q) {
                        // The enclosing callable chose this quantified before descending into its
                        // body. Clear the marker here to avoid another walk over that body.
                        q.needs_finalization = false
                    } else {
                        // We failed to find a callable to scope this free quantified to.
                        *ty = q.as_gradual_type();
                    }
                }
                return;
            }

            ty.transform_toplevel_callable_signatures(|callable, tparams| {
                let new_quantifieds = callable.quantifieds_to_declare(in_scope, tparams.as_deref());
                if !new_quantifieds.is_empty() {
                    let new_tparams = TParams::new(new_quantifieds);
                    if let Some(tparams) = tparams {
                        Arc::make_mut(tparams).extend(&new_tparams);
                    } else {
                        *tparams = Some(Arc::new(new_tparams));
                    }
                }
            });
            ty.recurse_with_type_parameter_scopes_mut(in_scope, &mut go);
        }
        go(self, &mut Vec::new())
    }

    /// Finalize free quantifieds (see `Type::finalize_free_quantifieds`) exposed outside a class instance.
    pub fn finalize_exposed_free_quantifieds(mut self) -> Type {
        self.finalize_exposed_free_quantifieds_mut();
        self
    }

    fn finalize_exposed_free_quantifieds_mut(&mut self) {
        match self {
            Type::Quantified(q) if q.needs_finalization => *self = q.as_gradual_type(),
            Type::ClassType(_) => {} // Skip finalizing free quantifieds in class instances.
            _ => {
                if self.is_toplevel_callable() {
                    self.finalize_free_quantifieds_mut()
                } else {
                    self.recurse_mut(&mut Type::finalize_exposed_free_quantifieds_mut)
                }
            }
        }
    }

    /// Combine per-overload results of a function call into a single type.
    pub fn combine_overload_results(results: Vec<Type>, heap: &TypeHeap) -> Option<Type> {
        let results = Vec1::try_from_vec(results).ok()?;
        let first = results.first();
        if results.iter().skip(1).all(|other| other == first) {
            return Some(first.clone());
        }
        if let Some(combined) = Self::try_combine_reconstructed_overload(&results) {
            return Some(combined);
        }
        if !Self::results_share_a_shape(&results) {
            return Some(unions(results.into_vec(), heap));
        }
        Some(Type::Overloaded(Box::new(results)))
    }

    /// Whether these types all have the same outermost shape.
    /// This is a very rough heuristic for whether a function has returned something like a wrapper
    /// class or a callback protocol that might hold an overloaded function.
    fn results_share_a_shape(results: &Vec1<Type>) -> bool {
        fn same_shape(a: &Type, b: &Type) -> bool {
            match (a, b) {
                (Type::ClassType(a), Type::ClassType(b)) => a.class_object() == b.class_object(),
                (Type::Tuple(_), Type::Tuple(_)) => true,
                (Type::Union(a), Type::Union(b)) => {
                    a.members.len() == b.members.len()
                        && a.members
                            .iter()
                            .zip(&b.members)
                            .all(|(a, b)| same_shape(a, b))
                }
                (Type::Intersect(a), Type::Intersect(b)) => {
                    a.0.len() == b.0.len()
                        && a.0.iter().zip(&b.0).all(|(a, b)| same_shape(a, b))
                        && same_shape(&a.1, &b.1)
                }
                _ => a.is_toplevel_callable() && b.is_toplevel_callable(),
            }
        }
        let first = results.first();
        results.iter().skip(1).all(|other| same_shape(first, other))
    }

    fn try_combine_reconstructed_overload(reconstructed: &[Type]) -> Option<Type> {
        let metadata = reconstructed
            .first()?
            .toplevel_func_metadata()
            .cloned()
            .unwrap_or(FuncMetadata {
                kind: FunctionKind::Overload,
                flags: FuncFlags::default(),
            });
        let signatures = reconstructed
            .iter()
            .cloned()
            .map(|branch_ty| branch_ty.into_overload_signatures(&metadata))
            .collect::<Option<Vec<_>>>()?
            .into_iter()
            .flatten()
            .collect();
        let signatures = Vec1::try_from_vec(signatures).ok()?;
        Some(Type::Overload(Overload {
            signatures,
            metadata: Box::new(metadata),
        }))
    }

    fn into_overload_signatures(self, metadata: &FuncMetadata) -> Option<Vec<OverloadType>> {
        match self {
            Type::Function(function) => Some(vec![OverloadType::Function(*function)]),
            Type::Forall(forall) => match forall.body {
                Forallable::Function(function) => Some(vec![OverloadType::Forall(Forall {
                    tparams: forall.tparams,
                    body: function,
                })]),
                Forallable::Callable(callable) => Some(vec![OverloadType::Forall(Forall {
                    tparams: forall.tparams,
                    body: Function {
                        signature: callable,
                        metadata: metadata.clone(),
                    },
                })]),
                Forallable::TypeAlias(_) => None,
            },
            Type::Callable(callable) => Some(vec![OverloadType::Function(Function {
                signature: *callable,
                metadata: metadata.clone(),
            })]),
            Type::Overload(overload) => Some(overload.signatures.into_vec()),
            _ => None,
        }
    }

    /// Whether this type is a placeholder for deferred callable structure.
    pub fn is_placeholder(&self) -> bool {
        matches!(self, Type::Quantified(q) if q.needs_finalization)
    }
}

impl Callable {
    /// Quantifieds to declare in the scope of this callable.
    fn quantifieds_to_declare(
        &self,
        in_scope: &[Quantified],
        declared_here: Option<&TParams>,
    ) -> Vec<Quantified> {
        let mut found = SmallSet::new();
        self.params.visit(&mut |ty: &Type| {
            ty.for_each_free_quantified(&mut |q| {
                found.insert(q);
            })
        });
        let mut to_declare = found
            .into_iter()
            .filter(|q| {
                q.needs_finalization
                    && !in_scope.contains(q)
                    && !declared_here.is_some_and(|tparams| tparams.iter().any(|x| x == *q))
            })
            .map(|q| {
                let mut q = q.clone();
                q.needs_finalization = false;
                q
            })
            .collect::<Vec<_>>();
        to_declare.sort(); // Sort by stable identity for determinism.
        to_declare
    }
}
