/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use ruff_python_ast::name::Name;

use crate::alt::answers::LookupAnswer;
use crate::alt::answers_solver::AnswersSolver;
use crate::types::callable::Callable;
use crate::types::callable::FuncMetadata;
use crate::types::callable::Function;
use crate::types::callable::Param;
use crate::types::callable::ParamList;
use crate::types::class::Class;
use crate::types::types::Type;

impl<'a, Ans: LookupAnswer> AnswersSolver<'a, Ans> {
    /// Wrap a parameter list and return type into a synthesized method `Type` bound to `cls`. Shared
    /// by the dataclass, attrs, and pydantic synthesizers, which all build methods this same way.
    pub(crate) fn synthesized_method(
        &self,
        cls: &Class,
        name: Name,
        params: Vec<Param>,
        ret: Type,
    ) -> Type {
        self.heap.mk_function(Function {
            signature: Callable::list(ParamList::new(params), ret),
            metadata: FuncMetadata::method(cls, name),
        })
    }
}
