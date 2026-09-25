/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Contextual projection of list literals for shape-extension markers.

use std::cell::RefCell;

use pyrefly_types::types::Type;
use ruff_python_ast::ExprList;
use ruff_text_size::TextRange;

use crate::alt::answers::LookupAnswer;
use crate::alt::answers_solver::AnswersSolver;
use crate::alt::int_list_literal::int_list_literal;
use crate::alt::regular_nested_list::regular_nested_list;
use crate::alt::unwrap::HintRef;
use crate::error::collector::ErrorCollector;

impl<'ctx, 'answer, Ans: LookupAnswer> AnswersSolver<'ctx, 'answer, Ans> {
    fn project_list_literal_marker<D>(
        &self,
        hint: Option<HintRef>,
        errors: &ErrorCollector,
        decompose: impl Fn(&Type) -> Option<D>,
        project: impl Fn(&D, &ErrorCollector) -> Option<(Type, Vec<(TextRange, Type)>)>,
    ) -> Option<Type> {
        let hint = hint?;
        let raw_hints = hint.types();
        let flattened_hints = self.flatten_alias_union_hints(raw_hints);
        if !flattened_hints
            .as_deref()
            .unwrap_or(raw_hints)
            .iter()
            .any(|hint| decompose(hint).is_some())
        {
            return None;
        }

        let successful_projections = RefCell::new(Vec::new());
        let projected = self.infer_with_decomposed_hint(Some(hint), decompose, |marker, _| {
            let Some(marker) = marker else {
                return self.stdlib.object().clone().to_type();
            };
            let branch_errors = self.error_collector();
            match project(&marker, &branch_errors) {
                Some((ty, traces)) if !branch_errors.has_hard() => {
                    successful_projections
                        .borrow_mut()
                        .push((ty.clone(), branch_errors, traces));
                    ty
                }
                Some(_) | None => self.stdlib.object().clone().to_type(),
            }
        });
        let (_, branch_errors, traces) = successful_projections
            .into_inner()
            .into_iter()
            .find(|(ty, _, _)| ty == &projected)?;
        errors.extend(branch_errors);
        for (range, ty) in traces {
            self.record_type_trace(range, &ty);
        }
        Some(projected)
    }

    pub(crate) fn project_shape_list_literal(
        &self,
        list: &ExprList,
        hint: Option<HintRef>,
        errors: &ErrorCollector,
    ) -> Option<Type> {
        if !self.solver().config.tensor_shapes {
            return None;
        }
        // Marker projections must precede ordinary list arms, which otherwise win overload
        // selection when a marker contains an unsolved shape variable.
        self.project_list_literal_marker(hint, errors, int_list_literal, |marker, errors| {
            self.project_int_list_literal_hint(list, marker, errors)
                .map(|ty| (ty, Vec::new()))
        })
        .or_else(|| {
            self.project_list_literal_marker(hint, errors, regular_nested_list, |marker, errors| {
                self.project_regular_nested_list_hint(list, marker, errors)
            })
        })
    }
}
