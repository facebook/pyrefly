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

use std::sync::Arc;

use pyrefly_types::callable::Param;
use pyrefly_types::callable::Params;
use pyrefly_types::dimension::Int;
use pyrefly_types::dimension::canonicalize;
use pyrefly_types::dimension::gradual_size;
use pyrefly_types::dimension::is_gradual_size;
use pyrefly_types::dimension::is_optional_int;
use pyrefly_types::heap::TypeHeap;
use pyrefly_types::quantified::Quantified;
use pyrefly_types::quantified::QuantifiedKind;
use pyrefly_types::shaped_array::IntTuple;
use pyrefly_types::shaped_array::IntTupleView;
use pyrefly_types::shaped_array::is_int_tuples_type;
use pyrefly_types::shaped_array::tuple_carrier_to_shape;
use pyrefly_types::simplify::unions;
use pyrefly_types::stdlib::Stdlib;
use pyrefly_types::tuple::Tuple;
use pyrefly_types::type_var::Restriction;
use pyrefly_types::types::TParams;
use pyrefly_types::types::Type;
use pyrefly_types::types::Var;

/// Join concrete equal-rank shapes dimension-wise; other differing shapes become gradual.
pub(crate) fn join_int_tuples(left: &IntTuple, right: &IntTuple) -> IntTuple {
    if left == right {
        return left.clone();
    }
    match (left.view(), right.view()) {
        (IntTupleView::Concrete(left), IntTupleView::Concrete(right))
            if left.len() == right.len() =>
        {
            IntTuple::new(
                left.iter()
                    .zip(right)
                    .map(|(left, right)| {
                        if left == right {
                            left.clone()
                        } else {
                            Int::Int
                        }
                    })
                    .collect(),
            )
        }
        _ => IntTuple::shapeless(),
    }
}

fn contains_parameter(ty: &Type, parameter: Var) -> bool {
    ty.any(|ty| matches!(ty, Type::Var(candidate) if *candidate == parameter))
}

/// Index of the unique `IntTuple`-bounded type parameter, if there is exactly one.
fn unique_int_tuple_bound_index(tparams: &TParams) -> Option<usize> {
    let mut indices = tparams
        .iter()
        .enumerate()
        .filter_map(|(index, parameter)| has_int_tuple_bound(parameter).then_some(index));
    let shape_index = indices.next()?;
    indices.next().is_none().then_some(shape_index)
}

/// Extract the sole `IntTuple`-bounded argument when it is a solver variable used by no other
/// argument of the same class.
fn shape_parameter(ty: &Type) -> Option<Var> {
    let (targs, shape_index) = match ty {
        Type::ShapedArray(array) => (
            array.base_class.targs().as_slice(),
            array.tuple_carrier_shape_arg_index()?,
        ),
        Type::ClassType(cls) => (
            cls.targs().as_slice(),
            unique_int_tuple_bound_index(cls.tparams())?,
        ),
        _ => return None,
    };
    let Type::Var(parameter) = targs.get(shape_index)? else {
        return None;
    };
    has_only_shape_argument_occurrences(targs, Some(shape_index), *parameter).then_some(*parameter)
}

/// Extract the sole eligible shape variable from a union or a single-spelled parameter.
///
/// Union arms that do not mention the variable are allowed so ordinary alternatives such as
/// `None` do not disable shape joining. Any arm that mentions the variable must use it solely as
/// its shape argument.
fn union_shape_parameter(ty: &Type) -> Option<Var> {
    match ty {
        Type::Union(union) => {
            let mut parameters = union.members.iter().filter_map(shape_parameter);
            let parameter = parameters.next()?;
            if parameters.any(|candidate| candidate != parameter) {
                return None;
            }
            union
                .members
                .iter()
                .all(|member| {
                    shape_parameter(member) == Some(parameter)
                        || !contains_parameter(member, parameter)
                })
                .then_some(parameter)
        }
        _ => shape_parameter(ty),
    }
}

/// Check that every occurrence of a variable is an array-like shape argument in the return type.
fn has_only_return_shape_occurrences(ty: &Type, parameter: Var) -> bool {
    match ty {
        Type::Union(union) => union.members.iter().all(|member| {
            !contains_parameter(member, parameter)
                || has_only_return_shape_occurrences(member, parameter)
        }),
        Type::ShapedArray(array) => has_only_shape_argument_occurrences(
            array.base_class.targs().as_slice(),
            array.tuple_carrier_shape_arg_index(),
            parameter,
        ),
        Type::ClassType(cls) => has_only_shape_argument_occurrences(
            cls.targs().as_slice(),
            unique_int_tuple_bound_index(cls.tparams()),
            parameter,
        ),
        _ => !contains_parameter(ty, parameter),
    }
}

/// Check that a variable occurs in the unique shape argument and nowhere else in the arguments.
fn has_only_shape_argument_occurrences(
    targs: &[Type],
    shape_index: Option<usize>,
    parameter: Var,
) -> bool {
    let Some(shape_index) = shape_index else {
        return false;
    };
    targs
        .get(shape_index)
        .is_some_and(|shape| contains_parameter(shape, parameter))
        && !targs
            .iter()
            .enumerate()
            .any(|(index, ty)| index != shape_index && contains_parameter(ty, parameter))
}

/// Return shape variables whose union-derived gradual solution cannot weaken another parameter or
/// a non-shape return occurrence.
pub(crate) fn union_shape_widening_vars(params: &Params, ret: &Type) -> Option<Arc<Vec<Var>>> {
    // A partial signature omits already-bound parameters, so it cannot prove that the shape
    // variable is owned solely by one remaining parameter.
    let Params::List(params) = params else {
        return None;
    };
    let vars = params
        .items()
        .iter()
        .enumerate()
        .filter(|(_, param)| !matches!(param, Param::Varargs(..) | Param::Kwargs(..)))
        .filter_map(|(index, param)| {
            union_shape_parameter(param.as_type()).map(|parameter| (index, parameter))
        })
        .filter_map(|(owner_index, parameter)| {
            let solely_owned = !params.items().iter().enumerate().any(|(index, param)| {
                index != owner_index && contains_parameter(param.as_type(), parameter)
            });
            let return_shaped_only = contains_parameter(ret, parameter)
                && has_only_return_shape_occurrences(ret, parameter);
            (solely_owned && return_shaped_only).then_some(parameter)
        })
        .collect::<Vec<_>>();
    (!vars.is_empty()).then(|| Arc::new(vars))
}

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

/// Whether `q` is a `TypeVar` whose bound represents one entire shape.
pub(crate) fn has_int_tuple_bound(q: &Quantified) -> bool {
    q.kind() == QuantifiedKind::TypeVar
        && matches!(q.restriction(), Restriction::Bound(Type::IntTuple(_)))
}

/// Preserve the shape domain when an `IntTuple`-bounded variable has no precise solution.
pub(crate) fn quantified_gradual_type(q: &Quantified) -> Type {
    if q.default().is_none()
        && q.kind() == QuantifiedKind::TypeVar
        && let Restriction::Bound(bound @ Type::IntTuple(_)) = q.restriction()
    {
        bound.clone()
    } else {
        q.as_gradual_type()
    }
}

fn int_tuples_member(ty: &Type) -> Option<Type> {
    IntTuple::from_shape_arg_type(ty)
        .or_else(|| tuple_carrier_to_shape(ty))
        .map(|shape| shape.to_shape_arg_type())
}

fn canonicalize_int_tuples_member(ty: &Type) -> Type {
    int_tuples_member(ty).unwrap_or_else(|| ty.clone())
}

fn canonicalize_int_tuples_sequence(candidate: &Type, heap: &TypeHeap) -> Type {
    match candidate {
        Type::Tuple(Tuple::Concrete(members)) => Type::Tuple(Tuple::Concrete(
            members.iter().map(canonicalize_int_tuples_member).collect(),
        )),
        Type::Tuple(Tuple::Unbounded(member)) => Type::Tuple(Tuple::Unbounded(Box::new(
            canonicalize_int_tuples_member(member),
        ))),
        Type::Tuple(Tuple::Unpacked(parts)) => {
            let (prefix, middle, suffix) = parts.parts();
            Type::Tuple(Tuple::unpacked(
                prefix.iter().map(canonicalize_int_tuples_member).collect(),
                canonicalize_int_tuples_sequence(middle, heap),
                suffix.iter().map(canonicalize_int_tuples_member).collect(),
            ))
        }
        Type::Union(union) => {
            let display_name = union.display_name.clone();
            let mut normalized = unions(
                union
                    .members
                    .iter()
                    .map(|member| canonicalize_int_tuples_sequence(member, heap))
                    .collect(),
                heap,
            );
            if let Type::Union(normalized_union) = &mut normalized {
                normalized_union.display_name = display_name;
            }
            normalized
        }
        _ => candidate.clone(),
    }
}

/// Normalize a candidate for a type variable bounded by `IntTuple` or a structural `IntTuples`
/// type.
///
/// Under `IntTuples`, for example, `tuple[tuple[Literal[2]], tuple[Literal[3], Literal[4]]]`
/// becomes `tuple[IntTuple[2], IntTuple[3, 4]]`. Bound checking remains responsible for rejecting
/// members that are not shapes.
pub(crate) fn normalize_shape_tuple_bound_candidate(
    quantified: &Quantified,
    candidate: &Type,
    heap: &TypeHeap,
) -> Option<Type> {
    if quantified.kind() != QuantifiedKind::TypeVar {
        return None;
    }
    match quantified.restriction() {
        Restriction::Bound(Type::IntTuple(_)) => Some(candidate.clone()),
        Restriction::Bound(bound) if is_int_tuples_type(bound) => {
            Some(canonicalize_int_tuples_sequence(candidate, heap))
        }
        _ => None,
    }
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
    use pyrefly_types::identity::IdentityIgnored;
    use pyrefly_types::lit_int::LitInt;
    use pyrefly_types::shaped_array::IntTuple;
    use pyrefly_types::tuple::Tuple;
    use pyrefly_types::types::Union;
    use pyrefly_types::types::Var;
    use pyrefly_util::uniques::UniqueFactory;

    use super::*;

    #[test]
    fn unresolved_solver_variable_is_not_a_dimension_solution() {
        let variable = Type::Var(Var::new(&UniqueFactory::new()));
        assert_eq!(shape_int_bound_solution(&variable), None);
    }

    #[test]
    fn canonicalizes_int_tuples_sequence_members() {
        let raw_member = Type::Tuple(Tuple::Concrete(vec![LitInt::new(2).to_explicit_type()]));
        let candidate = Type::Tuple(Tuple::Concrete(vec![raw_member.clone()]));
        let expected = Type::Tuple(Tuple::Concrete(vec![
            IntTuple::new(vec![Int::Literal(2)]).to_shape_arg_type(),
        ]));
        assert_eq!(
            canonicalize_int_tuples_sequence(&candidate, &TypeHeap::new()),
            expected
        );

        let unpacked = Type::Tuple(Tuple::unpacked(
            vec![raw_member.clone()],
            Type::Tuple(Tuple::Unbounded(Box::new(raw_member.clone()))),
            vec![raw_member],
        ));
        let canonical_member = IntTuple::new(vec![Int::Literal(2)]).to_shape_arg_type();
        assert_eq!(
            canonicalize_int_tuples_sequence(&unpacked, &TypeHeap::new()),
            Type::Tuple(Tuple::unpacked(
                vec![canonical_member.clone()],
                Type::Tuple(Tuple::Unbounded(Box::new(canonical_member.clone()))),
                vec![canonical_member],
            ))
        );
    }

    #[test]
    fn canonicalizes_union_of_int_tuples_sequences() {
        let raw_shape = |n| Type::Tuple(Tuple::Concrete(vec![LitInt::new(n).to_explicit_type()]));
        let candidate = Type::Union(Box::new(Union {
            members: vec![
                Type::Tuple(Tuple::Concrete(vec![raw_shape(2)])),
                Type::Tuple(Tuple::Concrete(vec![raw_shape(3)])),
            ],
            display_name: IdentityIgnored(None),
        }));
        assert_eq!(
            canonicalize_int_tuples_sequence(&candidate, &TypeHeap::new()),
            Type::Union(Box::new(Union {
                members: vec![
                    Type::Tuple(Tuple::Concrete(vec![
                        IntTuple::new(vec![Int::Literal(2)]).to_shape_arg_type(),
                    ])),
                    Type::Tuple(Tuple::Concrete(vec![
                        IntTuple::new(vec![Int::Literal(3)]).to_shape_arg_type(),
                    ])),
                ],
                display_name: IdentityIgnored(None),
            }))
        );
    }
}
