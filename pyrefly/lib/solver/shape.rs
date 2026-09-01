/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Pure shape-type normalization used at solver boundaries.
//!
//! The generic solver remains responsible for variable state and subset traversal. This module
//! isolates the shape-specific decisions it needs when admitting or simplifying solver values.

use pyrefly_types::dimension::Int;
use pyrefly_types::dimension::canonicalize;
use pyrefly_types::dimension::gradual_size;
use pyrefly_types::dimension::is_gradual_size;
use pyrefly_types::dimension::is_optional_int;
use pyrefly_types::heap::TypeHeap;
use pyrefly_types::quantified::Quantified;
use pyrefly_types::quantified::QuantifiedKind;
use pyrefly_types::simplify::unions;
use pyrefly_types::stdlib::Stdlib;
use pyrefly_types::type_var::Restriction;
use pyrefly_types::types::Type;

/// Normalize a candidate answer for an `IntVar`.
///
/// Existing `IntVar` leaves stay as bare quantified/type-var values so substitution preserves
/// source-level spellings like `Int[N]`; compound dimension expressions are canonicalized to
/// `Type::Int`.
pub(crate) fn type_as_intvar_solution(ty: &Type) -> Option<Type> {
    match ty {
        _ if ty.is_any() => Some(gradual_size()),
        Type::ClassType(cls) if cls.is_builtin("int") => Some(gradual_size()),
        Type::Quantified(q) if q.kind() == QuantifiedKind::IntVar => Some(ty.clone()),
        Type::TypeVar(tv) if tv.kind() == QuantifiedKind::IntVar => Some(ty.clone()),
        // An unsolved solver variable has no dimension structure for `Int::from_type` to retain.
        // Wrap it as a symbolic dimension so later solving can preserve the eventual value.
        Type::Var(_) => Some(Type::Int(Int::Symbolic(Box::new(ty.clone())))),
        _ => Int::from_type(ty).map(|dimension| canonicalize(Type::Int(dimension))),
    }
}

fn shape_int_bound_solution(ty: &Type) -> Option<Type> {
    match ty {
        Type::Var(_) => None,
        Type::Any(_) => Some(gradual_size()),
        Type::ClassType(cls) if cls.is_builtin("int") => Some(gradual_size()),
        _ => Int::from_type(ty).map(|dimension| canonicalize(Type::Int(dimension))),
    }
}

/// A normalized answer for a type variable bounded by shape `Int` or `Int | None`.
/// `precise_union` is present only when `answer` was widened to gradual `Int`; the solver uses the
/// original union instead when that widening would violate an accumulated upper bound.
pub(crate) struct ShapeIntBoundSolution {
    pub(crate) answer: Type,
    pub(crate) precise_union: Option<Type>,
}

/// Whether `q` is a `TypeVar` whose bound represents an entire shape.
pub(crate) fn has_int_tuple_bound(q: &Quantified) -> bool {
    q.kind() == QuantifiedKind::TypeVar
        && matches!(q.restriction(), Restriction::Bound(Type::IntTuple(_)))
}

/// Preserve dimension precision when solving an ordinary type variable bounded by `Int` or
/// `Int | None`.
///
/// Normal type-variable solving promotes integer literals to `int`. The shape `Int` bound instead
/// promises that every accepted solution is a dimension, so normalize each union member through
/// the dimension representation before the generic solver checks the bound.
pub(crate) fn normalize_shape_int_bound_solution(
    quantified: &Quantified,
    ty: &Type,
    stdlib: &Stdlib,
    heap: &TypeHeap,
) -> Option<ShapeIntBoundSolution> {
    if quantified.kind() != QuantifiedKind::TypeVar {
        return None;
    }
    let Restriction::Bound(bound) = quantified.restriction() else {
        return None;
    };
    let exact_int_bound = is_gradual_size(bound);
    let optional_int_bound = is_optional_int(bound);
    if !exact_int_bound && !optional_int_bound {
        return None;
    }
    let normalize_member = |member: &Type| {
        shape_int_bound_solution(member)
            .unwrap_or_else(|| member.clone().promote_implicit_literals(stdlib))
    };
    if optional_int_bound {
        let normalize_optional_member = |member: &Type| match member {
            Type::Any(_) => member.clone(),
            _ => normalize_member(member),
        };
        return Some(ShapeIntBoundSolution {
            answer: match ty {
                Type::Union(union) => unions(
                    union
                        .members
                        .iter()
                        .map(normalize_optional_member)
                        .collect(),
                    heap,
                ),
                member => normalize_optional_member(member),
            },
            precise_union: None,
        });
    }
    Some(match ty {
        Type::Union(union) => {
            let members = union
                .members
                .iter()
                .map(normalize_member)
                .collect::<Vec<_>>();
            if members.iter().all(|member| matches!(member, Type::Int(_))) {
                ShapeIntBoundSolution {
                    answer: gradual_size(),
                    precise_union: Some(unions(members, heap)),
                }
            } else {
                ShapeIntBoundSolution {
                    answer: unions(members, heap),
                    precise_union: None,
                }
            }
        }
        member => ShapeIntBoundSolution {
            answer: normalize_member(member),
            precise_union: None,
        },
    })
}

fn canonicalize_int(ty: &mut Type) {
    if let Type::Int(_) = ty {
        let canonical = canonicalize(ty.clone());
        if &canonical != ty {
            *ty = canonical;
        }
    }
}

/// Canonicalize dimension expressions after the solver expands their variables.
pub(crate) fn canonicalize_ints_in_type(ty: &mut Type) {
    ty.transform_mut(&mut canonicalize_int);
}

/// Simplify one shape-related node during the solver's post-order type simplification.
pub(crate) fn simplify_shape_type(ty: &mut Type) {
    canonicalize_int(ty);
    if let Type::IntTuple(shape) = ty {
        **shape = shape.normalize();
    }
    if let Type::ShapedArray(tensor) = ty {
        match tensor.tuple_carrier_shape_arg_index() {
            Some(index)
                if !matches!(
                    tensor.base_class.targs().as_slice().get(index),
                    Some(Type::IntTuple(_))
                ) =>
            {
                let shape = tensor.shape();
                tensor.set_shape(shape);
            }
            None => {
                let shape = tensor.shape().normalize();
                tensor.set_shape(shape);
            }
            // This traversal is post-order, so the first-class shape argument was normalized
            // before its containing shaped array.
            Some(_) => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use pyrefly_types::types::Var;
    use pyrefly_util::uniques::UniqueFactory;

    use super::*;

    #[test]
    fn unresolved_solver_variable_is_not_a_dimension_solution() {
        let variable = Type::Var(Var::new(&UniqueFactory::new()));
        assert_eq!(shape_int_bound_solution(&variable), None);
    }
}
