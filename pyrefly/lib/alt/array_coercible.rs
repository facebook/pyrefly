/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! List-literal inference for `shape_extensions.ArrayCoercible`.
//!
//! The marker representation and subset logic live in `crate::solver::shape_markers`; this module
//! keeps only the `AnswersSolver` work of projecting list literals against a domain.

use pyrefly_types::dimension::Int;
use pyrefly_types::shaped_array::IntTuple;
use pyrefly_types::shaped_array::IntTupleView;
use pyrefly_types::types::Type;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprList;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;

use crate::alt::answers::LookupAnswer;
use crate::alt::answers_solver::AnswersSolver;
use crate::alt::unwrap::HintRef;
use crate::alt::unwrap::MAX_HINT_WIDTH;
use crate::error::collector::ErrorCollector;
use crate::solver::shape_markers::array_coercible;
use crate::solver::solver::VarSnapshot;

impl<'ctx, 'answer, Ans: LookupAnswer> AnswersSolver<'ctx, 'answer, Ans> {
    /// Project a rectangular, unstarred list literal against one scalar-leaf domain.
    fn project_array_coercible_list(
        &self,
        list: &ExprList,
        domain: &Type,
        errors: &ErrorCollector,
        inference_snapshots: &mut Vec<VarSnapshot>,
        projected_lists: &mut Vec<(TextRange, IntTuple)>,
    ) -> Option<IntTuple> {
        if list
            .elts
            .iter()
            .any(|element| matches!(element, Expr::Starred(_)))
        {
            return None;
        }
        let mut child_shape: Option<IntTuple> = None;
        let concrete_matches_unpacked = |concrete: &[Int], prefix: &[Int], suffix: &[Int]| {
            concrete.len() >= prefix.len() + suffix.len()
                && concrete.starts_with(prefix)
                && concrete.ends_with(suffix)
        };
        for element in &list.elts {
            let shape = match element {
                Expr::List(child) => Some(self.project_array_coercible_list(
                    child,
                    domain,
                    errors,
                    inference_snapshots,
                    projected_lists,
                )?),
                _ => {
                    let ty =
                        self.expr_infer_with_hint(element, Some(HintRef::soft(domain)), errors);
                    if self.type_order().is_deferred_array_coercible_container(&ty) {
                        return None;
                    }
                    if ty.is_any() {
                        None
                    } else {
                        let snapshot = self
                            .solver()
                            .snapshot_for_speculative_inference(&[&ty, domain]);
                        if !self.is_subset_eq(&ty, domain) {
                            self.solver().restore_vars(snapshot);
                            return None;
                        }
                        // Each AnswersSolver subset check owns a transient subset, so retain only
                        // the solver-variable snapshot needed for outer rollback.
                        inference_snapshots.push(snapshot);
                        Some(IntTuple::new(Vec::new()))
                    }
                }
            };
            if let Some(shape) = shape {
                match child_shape.as_ref() {
                    Some(expected) if expected == &shape => {}
                    Some(expected) => match (expected.view(), shape.view()) {
                        (IntTupleView::Gradual, _) => {}
                        (_, IntTupleView::Gradual) => child_shape = Some(shape),
                        (
                            IntTupleView::Concrete(concrete),
                            IntTupleView::Unpacked { prefix, suffix, .. },
                        ) if concrete_matches_unpacked(concrete, prefix, suffix) => {}
                        (
                            IntTupleView::Unpacked { prefix, suffix, .. },
                            IntTupleView::Concrete(concrete),
                        ) if concrete_matches_unpacked(concrete, prefix, suffix) => {
                            child_shape = Some(shape)
                        }
                        _ => return None,
                    },
                    None => child_shape = Some(shape),
                }
            }
        }
        let outer = Int::Literal(list.elts.len() as i64);
        let shape = match child_shape {
            Some(child_shape) => match child_shape.view() {
                IntTupleView::Concrete(child) => {
                    let mut dims = Vec::with_capacity(child.len() + 1);
                    dims.push(outer);
                    dims.extend(child.iter().cloned());
                    IntTuple::new(dims)
                }
                IntTupleView::Gradual => IntTuple::unpacked(
                    vec![outer],
                    IntTuple::shapeless().to_shape_arg_type(),
                    Vec::new(),
                ),
                IntTupleView::Unpacked {
                    prefix,
                    middle,
                    suffix,
                } => {
                    let mut outer_prefix = Vec::with_capacity(prefix.len() + 1);
                    outer_prefix.push(outer);
                    outer_prefix.extend(prefix.iter().cloned());
                    IntTuple::unpacked(outer_prefix, middle.clone(), suffix.to_vec())
                }
            },
            None if list.is_empty() => IntTuple::new(vec![outer]),
            None => IntTuple::unpacked(
                vec![outer],
                IntTuple::shapeless().to_shape_arg_type(),
                Vec::new(),
            ),
        };
        projected_lists.push((list.range(), shape.clone()));
        Some(shape)
    }

    /// Infer a list literal from an `ArrayCoercible` hint, returning `None` to use ordinary list
    /// inference when the literal cannot be projected safely.
    pub(crate) fn infer_array_coercible_list(
        &self,
        list: &ExprList,
        hint: HintRef,
        errors: &ErrorCollector,
    ) -> Option<Type> {
        if !self.solver().tensor_shapes {
            return None;
        }
        if hint.types().len() > MAX_HINT_WIDTH {
            return None;
        }
        // Each marker domain is a distinct soft hint and can change leaf inference, so projection
        // results cannot be shared across alternatives. The hint-width limit bounds this work.
        for hint_ty in hint.types() {
            let Some(marker) = array_coercible(hint_ty) else {
                continue;
            };
            let branch_errors = self.error_collector();
            let mut inference_snapshots = Vec::new();
            let mut projected_lists = Vec::new();
            let snapshot = self
                .solver()
                .snapshot_for_speculative_inference(&[&marker.shape, &marker.domain]);
            let rollback = |inference_snapshots: Vec<VarSnapshot>| {
                for inference_snapshot in inference_snapshots.into_iter().rev() {
                    self.solver().restore_vars(inference_snapshot);
                }
                self.solver().restore_vars(snapshot);
            };
            match self.project_array_coercible_list(
                list,
                &marker.domain,
                &branch_errors,
                &mut inference_snapshots,
                &mut projected_lists,
            ) {
                Some(shape) if !branch_errors.has_hard() => {
                    let projected = marker.with_shape(shape);
                    if !self.is_subset_eq(&projected, hint_ty) {
                        rollback(inference_snapshots);
                        continue;
                    }
                    for (range, shape) in projected_lists {
                        self.record_type_trace(range, &marker.with_shape(shape));
                    }
                    errors.extend(branch_errors);
                    return Some(projected);
                }
                Some(_) | None => {
                    // Projection failures are alternative-specific, and an ordinary union
                    // alternative may still accept the literal.
                    rollback(inference_snapshots);
                }
            }
        }
        None
    }
}
