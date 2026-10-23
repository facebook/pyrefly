/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Literal-value inference for `shape_extensions.IntListLiteral`.

use pyrefly_types::class::ClassType;
use pyrefly_types::literal::LitStyle;
use pyrefly_types::shaped_array::IntTuple;
use pyrefly_types::types::Type;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprList;

use crate::alt::answers::LookupAnswer;
use crate::alt::answers_solver::AnswersSolver;
use crate::error::collector::ErrorCollector;

pub(crate) struct IntListLiteral {
    class_type: ClassType,
}

pub(crate) fn int_list_literal(ty: &Type) -> Option<IntListLiteral> {
    let Type::ClassType(cls) = ty else {
        return None;
    };
    if !cls.has_qname("shape_extensions", "IntListLiteral") {
        return None;
    }
    let [_] = cls.targs().as_slice() else {
        return None;
    };
    Some(IntListLiteral {
        class_type: cls.clone(),
    })
}

impl IntListLiteral {
    fn with_values(&self, values: IntTuple) -> Type {
        let mut class_type = self.class_type.clone();
        class_type.targs_mut().as_mut()[0] = values.to_shape_arg_type();
        class_type.to_type()
    }
}

impl<'ctx, 'answer, Ans: LookupAnswer> AnswersSolver<'ctx, 'answer, Ans> {
    pub(crate) fn project_int_list_literal_hint(
        &self,
        list: &ExprList,
        marker: &IntListLiteral,
        errors: &ErrorCollector,
    ) -> Option<Type> {
        if list
            .elts
            .iter()
            .any(|element| matches!(element, Expr::Starred(_)))
        {
            return None;
        }
        let int_type = self.stdlib.int().clone().to_type();
        let mut values = Vec::with_capacity(list.elts.len());
        for element in &list.elts {
            let value = self
                .expr_infer(element, errors)
                .with_literal_style(LitStyle::Explicit);
            if !self.is_subset_eq_with_reason(&value, &int_type).is_ok() {
                return None;
            }
            values.push(value);
        }
        Some(marker.with_values(IntTuple::from_types(values)))
    }

    pub(crate) fn int_list_literal_parameter_body_type(&self, mut ty: Type) -> Type {
        ty.transform_mut(&mut |ty| {
            if int_list_literal(ty).is_some() {
                *ty = self
                    .heap
                    .mk_class_type(self.stdlib.list(self.stdlib.int().clone().to_type()));
            }
        });
        ty
    }
}
