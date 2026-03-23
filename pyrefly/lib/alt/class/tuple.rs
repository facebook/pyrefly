/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use pyrefly_python::dunder;

use crate::alt::answers::LookupAnswer;
use crate::alt::answers_solver::AnswersSolver;
use crate::types::class::ClassType;
use crate::types::simplify::simplify_tuples;
use crate::types::tuple::Tuple;
use crate::types::types::Type;

impl<'a, Ans: LookupAnswer> AnswersSolver<'a, Ans> {
    pub fn as_tuple(&self, cls: &ClassType) -> Option<Tuple> {
        if let Some(named_tuple_elements) = self.named_tuple_element_types(cls) {
            return Some(Tuple::Concrete(named_tuple_elements));
        }
        if cls.class_object().is_builtin("tuple") && cls.targs().as_slice().len() == 1 {
            let tuple_targ: Type = cls.targs().as_slice()[0].clone();
            return Some(Tuple::Unbounded(Box::new(tuple_targ)));
        }
        let class_bases = self.get_base_types_for_class(cls.class_object());
        if let Some(Type::Tuple(tuple)) = class_bases
            .tuple_ancestor()
            .cloned()
            .map(Type::Tuple)
            .map(|ty| cls.targs().substitute_into(ty))
        {
            return Some(simplify_tuples(tuple, self.heap));
        }
        None
    }
    /// Check whether tuple indexing should use tuple's special subscript handling
    /// rather than a user-defined `__getitem__` override.
    pub fn tuple_uses_builtin_getitem(&self, cls: &ClassType) -> bool {
        self.get_non_synthesized_class_member_and_defining_class(
            cls.class_object(),
            &dunder::GETITEM,
        )
        .is_none_or(|member| member.defining_class.is_builtin("tuple"))
    }
}
