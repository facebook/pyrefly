/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Finishing a type at a boundary: giving each free type parameter a home.

use std::sync::Arc;

use pyrefly_util::visit::Visit;
use pyrefly_util::visit::VisitMut;
use starlark_map::small_set::SmallSet;

use crate::callable::Callable;
use crate::quantified::Quantified;
use crate::types::TParams;
use crate::types::Type;

impl Type {
    /// Scope each free quantified to the outermost callable whose parameter types contain it.
    /// Replace quantifieds with no possible scope with their gradual fallback.
    pub fn finalize_free_quantifieds(mut self) -> Type {
        self.finalize_free_quantifieds_mut();
        self
    }

    pub fn finalize_free_quantifieds_mut(&mut self) {
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
