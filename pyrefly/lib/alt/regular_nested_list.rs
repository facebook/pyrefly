/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! List-literal inference for `shape_extensions.RegularNestedList`.
//!
//! A regular nested list has the same shape at every sibling position: "regular" is the
//! opposite of jagged or irregular. The marker is a contextual structural type for APIs that
//! accept nested Python list literals and want to infer their shape while checking every non-list
//! leaf against a domain type. It does not describe arbitrary existing containers. Only unstarred
//! list syntax is projected; jagged literals, hidden container structure, and failed domain checks
//! fall back to ordinary contextual typing.

use pyrefly_types::class::ClassType;
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
use crate::error::collector::ErrorCollector;
use crate::solver::solver::SubsetError;

/// A decomposed contextual marker, not a nominal type for existing list values.
pub(crate) struct RegularNestedList {
    domain: Type,
    class_type: ClassType,
}

pub(crate) fn regular_nested_list(ty: &Type) -> Option<RegularNestedList> {
    let Type::ClassType(cls) = ty else {
        return None;
    };
    if !cls.has_qname("shape_extensions", "RegularNestedList") {
        return None;
    }
    let [_, domain] = cls.targs().as_slice() else {
        return None;
    };
    Some(RegularNestedList {
        domain: domain.clone(),
        class_type: cls.clone(),
    })
}

impl RegularNestedList {
    fn with_shape(&self, shape: IntTuple) -> Type {
        let mut class_type = self.class_type.clone();
        class_type.targs_mut().as_mut()[0] = shape.to_shape_arg_type();
        class_type.to_type()
    }
}

impl<'ctx, 'answer, Ans: LookupAnswer> AnswersSolver<'ctx, 'answer, Ans> {
    /// Project a regular, unstarred nested list literal against one scalar-leaf domain.
    fn project_regular_nested_list(
        &self,
        list: &ExprList,
        domain: &Type,
        errors: &ErrorCollector,
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
        for element in &list.elts {
            let shape = match element {
                Expr::List(child) => {
                    self.project_regular_nested_list(child, domain, errors, projected_lists)?
                }
                _ => {
                    let domain_vars = domain.collect_maybe_placeholder_vars();
                    let mut leaf_shape = None;
                    let projected = self.solver().with_snapshot(&domain_vars, || {
                        let hint =
                            matches!(element, Expr::Lambda(_)).then(|| HintRef::soft(domain));
                        let ty = self.expr_infer_with_hint(element, hint, errors);
                        if ty.is_any() {
                            leaf_shape = Some(IntTuple::shapeless());
                            return Ok(());
                        }

                        let mut expanded = ty.clone();
                        self.solver().expand_with_bounds(&mut expanded);
                        let hides_structure = match &expanded {
                            Type::Literal(_) | Type::LiteralString(_) => false,
                            Type::ClassType(cls) if cls.is_builtin("list") => true,
                            Type::Tuple(_) | Type::Var(_) => true,
                            _ => self.is_sequence_for_pattern(&expanded),
                        };
                        if hides_structure {
                            return Err(SubsetError::Other);
                        }
                        let vars = ty.collect_maybe_placeholder_vars();
                        if !self
                            .solver()
                            .with_snapshot(&vars, || self.is_subset_eq_with_reason(&ty, domain))
                            .is_ok()
                        {
                            return Err(SubsetError::Other);
                        }
                        leaf_shape = Some(IntTuple::new(Vec::new()));
                        Ok(())
                    });
                    if !projected.is_ok() {
                        return None;
                    }
                    leaf_shape.expect("successful leaf projection records its shape")
                }
            };
            match child_shape.as_ref() {
                Some(expected) if expected == &shape => {}
                Some(expected)
                    if self.is_subset_eq(
                        &shape.to_shape_arg_type(),
                        &expected.to_shape_arg_type(),
                    ) => {}
                Some(expected)
                    if self.is_subset_eq(
                        &expected.to_shape_arg_type(),
                        &shape.to_shape_arg_type(),
                    ) =>
                {
                    child_shape = Some(shape)
                }
                Some(_) => return None,
                None => child_shape = Some(shape),
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
                    let mut combined_prefix = Vec::with_capacity(prefix.len() + 1);
                    combined_prefix.push(outer);
                    combined_prefix.extend_from_slice(prefix);
                    IntTuple::unpacked(combined_prefix, middle.clone(), suffix.to_vec())
                }
            },
            None => IntTuple::new(vec![outer]),
        };
        projected_lists.push((list.range(), shape.clone()));
        Some(shape)
    }

    /// Project a list literal from one decomposed `RegularNestedList` hint.
    pub(crate) fn project_regular_nested_list_hint(
        &self,
        list: &ExprList,
        marker: &RegularNestedList,
        errors: &ErrorCollector,
    ) -> Option<(Type, Vec<(TextRange, Type)>)> {
        let mut projected_lists = Vec::new();
        let shape =
            self.project_regular_nested_list(list, &marker.domain, errors, &mut projected_lists)?;
        Some((
            marker.with_shape(shape),
            projected_lists
                .into_iter()
                .map(|(range, shape)| (range, marker.with_shape(shape)))
                .collect(),
        ))
    }
}
