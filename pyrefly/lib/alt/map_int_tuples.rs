/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Solver integration for the experimental `shape_extensions.MapIntTuples` operation.
//!
//! `MapIntTuples` is a shape-specific type operator rather than part of the type-level DSL
//! evaluator. Keeping its solver behavior here prevents the general annotation and DSL paths from
//! accumulating details of its mapper binding, evaluation, and parameter-pattern semantics.

use pyrefly_types::map_int_tuples::map_int_tuples_mapper_binder;
use pyrefly_types::types::Type;
use ruff_python_ast::Identifier;

use crate::alt::answers::LookupAnswer;
use crate::alt::answers_solver::AnswersSolver;

impl<'ctx, 'answer, Ans: LookupAnswer> AnswersSolver<'ctx, 'answer, Ans> {
    pub(crate) fn resolve_map_int_tuples_mapper_parameter(&self, name: &Identifier) -> Type {
        let binder = map_int_tuples_mapper_binder(self.module().name(), name);
        self.heap.mk_type_of(binder.to_type(self.heap))
    }
}
