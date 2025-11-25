/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use starlark_map::small_map::SmallMap;
use starlark_map::small_set::SmallSet;

use crate::class::Class;
use crate::class::ClassType;
use crate::literal::Lit;
use crate::stdlib::Stdlib;
use crate::tuple::Tuple;
use crate::types::Type;
use crate::types::Union;

/// Turn unions of unions into a flattened list for one union, and return the deduped list.
fn flatten_and_dedup(xs: Vec<Type>) -> Vec<Type> {
    fn flatten(xs: Vec<Type>, res: &mut Vec<Type>) {
        for x in xs {
            match x {
                Type::Union(box Union { members, .. }) => flatten(members, res),
                Type::Never(_) => {}
                _ => res.push(x),
            }
        }
    }
    let mut flattened = Vec::with_capacity(xs.len());
    flatten(xs, &mut flattened);
    simplify_intersections(&mut flattened);
    let mut res = Vec::with_capacity(flattened.len());
    flatten(flattened, &mut res);

    res.sort();
    res.dedup();
    res
}

/// Given a list of types to union together,
/// - If there's 0 element in the list, return `Ok` with `Type::never()`.
/// - If there's 1 element in the list, return `Ok` with that element.
/// - Otherwise, return `Err` along with `xs`.
fn try_collapse(mut xs: Vec<Type>) -> Result<Type, Vec<Type>> {
    if xs.is_empty() {
        Ok(Type::never())
    } else if xs.len() == 1 {
        Ok(xs.pop().unwrap())
    } else {
        Err(xs)
    }
}

fn simplify_intersections(xs: &mut [Type]) {
    // Simplify `A | (A & B)` to `A`
    let (mut intersects, non_intersects): (Vec<_>, Vec<_>) =
        xs.iter_mut().partition(|x| matches!(x, Type::Intersect(_)));
    for x in intersects.iter_mut() {
        if let Type::Intersect(y) = x
            && y.0.iter_mut().any(|t| non_intersects.contains(&t))
        {
            **x = Type::never();
        }
    }
}

fn unions_internal(
    xs: Vec<Type>,
    stdlib: Option<&Stdlib>,
    enum_members: Option<&dyn Fn(&Class) -> Option<usize>>,
) -> Type {
    try_collapse(xs).unwrap_or_else(|xs| {
        let mut res = flatten_and_dedup(xs);
        if let Some(stdlib) = stdlib {
            collapse_literals(&mut res, stdlib, enum_members.unwrap_or(&|_| None));
        }
        collapse_tuple_unions_with_empty(&mut res);
        // `res` is collapsible again if `flatten_and_dedup` drops `xs` to 0 or 1 elements
        try_collapse(res).unwrap_or_else(|members| {
            Type::Union(Box::new(Union {
                members,
                display_name: None,
            }))
        })
    })
}

/// Union a set of types together, simplifying as much as you can.
pub fn unions(xs: Vec<Type>) -> Type {
    unions_internal(xs, None, None)
}

/// Like `unions`, but also simplify away things regarding literals if you can,
/// e.g. `Literal[True, False] ==> bool`.
pub fn unions_with_literals(
    xs: Vec<Type>,
    stdlib: &Stdlib,
    enum_members: &dyn Fn(&Class) -> Option<usize>,
) -> Type {
    unions_internal(xs, Some(stdlib), Some(enum_members))
}

pub fn intersect(ts: Vec<Type>, fallback: Type) -> Type {
    let mut flattened = Vec::new();
    for t in ts {
        match t {
            Type::Union(_) => {
                // TODO: Flatten these instead of giving up.
                return fallback;
            }
            Type::Intersect(x) => flattened.extend(x.0),
            t => flattened.push(t),
        }
    }
    flattened.sort();
    flattened.dedup();
    if flattened.is_empty() || flattened.iter().any(|t| t.is_never()) {
        Type::never()
    } else if flattened.len() == 1 {
        flattened.into_iter().next().unwrap()
    } else {
        Type::Intersect(Box::new((flattened, fallback)))
    }
}

fn remove_maximum<T: Ord>(xs: &mut Vec<T>) {
    // Remove the maximum element, if it exists.
    if xs.len() <= 1 {
        xs.clear();
        return;
    }

    // There are only three elements at most, so sort is pretty cheap
    xs.sort();
    xs.pop();
}

/// Perform all literal transformations we can think of.
///
/// 1. Literal[True, False] ==> bool
/// 2. Literal[0] | int => int (and for bool, int, str, bytes, enums)
/// 3. LiteralString | str => str
/// 4. LiteralString | Literal["x"] => LiteralString
/// 5. Any | Any => Any (if the Any are different variants)
/// 6. Never | Never => Never (if the Never are different variants)
fn collapse_literals(
    types: &mut Vec<Type>,
    stdlib: &Stdlib,
    enum_members: &dyn Fn(&Class) -> Option<usize>,
) {
    // All literal types we see, plus `true` to indicate they are found
    let mut literal_types = SmallMap::new();
    // Specific flags to watch out for
    let mut has_literal_string = false;
    let mut has_specific_str = false;
    let mut has_true = false;
    let mut has_false = false;

    let mut any_styles = Vec::new();
    let mut never_styles = Vec::new();

    // Mapping of enum classes to the number of members contained in the union
    let mut enums: SmallMap<ClassType, usize> = SmallMap::new();

    // Invariant (from the sorting order) is that all Literal/Lit values occur
    // before any instances of the types.
    // Therefore we only need to check if a ClassType is already in the map, rather than
    // inserting them all.
    for t in types.iter() {
        match t {
            Type::LiteralString => {
                has_literal_string = true;
                literal_types.insert(stdlib.str().clone(), false);
            }
            Type::Literal(x) => {
                match x {
                    Lit::Bool(true) => has_true = true,
                    Lit::Bool(false) => has_false = true,
                    Lit::Str(_) => has_specific_str = true,
                    Lit::Enum(x) => {
                        let v = enums.entry(x.class.clone()).or_insert(0);
                        *v += 1;
                    }
                    _ => {}
                }
                literal_types.insert(x.general_class_type(stdlib).clone(), false);
            }
            Type::ClassType(class)
                if !literal_types.is_empty()
                    && let Some(found) = literal_types.get_mut(class) =>
            {
                // Note: Check if literal_types is empty first, and if so, avoid hashing the class object.
                *found = true;
            }
            Type::Any(style) => any_styles.push(*style),
            Type::Never(style) => never_styles.push(*style),
            _ => {}
        }
    }

    let enums_to_delete: SmallSet<ClassType> = enums
        .into_iter()
        .filter(|(k, n)| {
            if let Some(num_members) = enum_members(k.class_object()) {
                return *n >= num_members;
            }
            false
        })
        .map(|x| x.0)
        .collect();
    for e in &enums_to_delete {
        types.push(Type::ClassType(e.clone()));
    }
    remove_maximum(&mut any_styles);
    remove_maximum(&mut never_styles);

    if literal_types.values().any(|x| *x)
        || (has_true && has_false)
        || (has_literal_string && has_specific_str)
        || !any_styles.is_empty()
        || !never_styles.is_empty()
        || !enums_to_delete.is_empty()
    {
        // We actually have some things to delete
        types.retain(|x| match x {
            Type::LiteralString => literal_types.get(stdlib.str()) == Some(&false),
            Type::Literal(x) => {
                match x {
                    Lit::Bool(_) if has_true && has_false => return false,
                    Lit::Str(_) if has_literal_string => return false,
                    Lit::Enum(lit_enum) if enums_to_delete.contains(&lit_enum.class) => {
                        if enums_to_delete.contains(&lit_enum.class) {
                            return false;
                        }
                    }
                    _ => {}
                }
                literal_types.get(x.general_class_type(stdlib)) == Some(&false)
            }
            Type::Any(style) => !any_styles.contains(style),
            Type::Never(style) => !never_styles.contains(style),
            _ => true,
        });

        if (has_true && has_false)
            && let bool = stdlib.bool()
            && literal_types.get(bool) == Some(&false)
            && let bool = bool.clone().to_type()
            && let Err(new_pos) = types.binary_search(&bool)
        {
            types.insert(new_pos, bool);
        }
    }
}

fn collapse_tuple_unions_with_empty(types: &mut Vec<Type>) {
    let Some(empty_idx) = types.iter().position(|t| match t {
        Type::Tuple(Tuple::Concrete(elts)) => elts.is_empty(),
        _ => false,
    }) else {
        return;
    };

    let mut empty_is_redundant = false;
    for (idx, ty) in types.iter_mut().enumerate() {
        if idx == empty_idx {
            continue;
        }
        match ty {
            Type::Tuple(Tuple::Unbounded(_)) => {
                empty_is_redundant = true;
            }
            Type::Tuple(Tuple::Unpacked(unpacked)) => {
                let (prefix, middle, suffix) = &**unpacked;
                if prefix.len() + suffix.len() == 1
                    && let Type::Tuple(Tuple::Unbounded(elem)) = middle
                    && prefix
                        .iter()
                        .chain(suffix.iter())
                        .all(|fixed| fixed == elem.as_ref())
                {
                    *ty = Type::unbounded_tuple(elem.as_ref().clone());
                    empty_is_redundant = true;
                }
            }
            _ => {}
        }
    }

    if empty_is_redundant {
        types.remove(empty_idx);
        types.sort();
        types.dedup();
    }
}

fn flatten_unpacked_concrete_tuples(elts: Vec<Type>) -> Vec<Type> {
    let mut result = Vec::new();
    for elt in elts {
        match elt {
            Type::Unpack(box Type::Tuple(Tuple::Concrete(elts))) => {
                result.extend(elts);
            }
            _ => result.push(elt),
        }
    }
    result
}

// After a TypeVarTuple gets substituted with a tuple type, try to simplify the type
pub fn simplify_tuples(tuple: Tuple) -> Tuple {
    match tuple {
        Tuple::Concrete(elts) => Tuple::Concrete(flatten_unpacked_concrete_tuples(elts)),
        Tuple::Unpacked(box (prefix, Type::Tuple(middle), suffix))
            if prefix.is_empty() && suffix.is_empty() =>
        {
            middle
        }
        Tuple::Unpacked(box (prefix, middle, suffix)) => match middle {
            Type::Tuple(Tuple::Concrete(elts)) => {
                Tuple::Concrete(flatten_unpacked_concrete_tuples(
                    prefix
                        .into_iter()
                        .chain(elts)
                        .chain(suffix)
                        .collect::<Vec<_>>(),
                ))
            }
            Type::Tuple(Tuple::Unpacked(box (m_prefix, m_middle, m_suffix))) => {
                let mut new_prefix = flatten_unpacked_concrete_tuples(prefix);
                new_prefix.extend(flatten_unpacked_concrete_tuples(m_prefix));
                let mut new_suffix = flatten_unpacked_concrete_tuples(m_suffix);
                new_suffix.extend(flatten_unpacked_concrete_tuples(suffix));
                Tuple::unpacked(new_prefix, m_middle, new_suffix)
            }
            _ => Tuple::unpacked(
                flatten_unpacked_concrete_tuples(prefix),
                middle,
                flatten_unpacked_concrete_tuples(suffix),
            ),
        },
        _ => tuple,
    }
}

#[cfg(test)]
mod tests {
    use crate::simplify::intersect;
    use crate::simplify::unions;
    use crate::tuple::Tuple;
    use crate::types::NeverStyle;
    use crate::types::Type;

    #[test]
    fn test_flatten_never() {
        let xs = vec![
            Type::Never(NeverStyle::Never),
            Type::Never(NeverStyle::NoReturn),
        ];
        let res = unions(xs);
        assert_eq!(res, Type::never());
    }

    #[test]
    fn test_intersect_simple() {
        let xs = vec![Type::any_tuple(), Type::any_implicit()];
        assert_eq!(
            intersect(xs.clone(), Type::never()),
            Type::Intersect(Box::new((xs, Type::never())))
        );
    }

    #[test]
    fn test_intersect_empty() {
        let xs = Vec::new();
        assert_eq!(intersect(xs, Type::any_implicit()), Type::never())
    }

    #[test]
    fn test_intersect_never() {
        let xs = vec![Type::any_implicit(), Type::never()];
        assert_eq!(intersect(xs, Type::any_implicit()), Type::never());
    }

    #[test]
    fn test_intersect_one() {
        let xs = vec![Type::None];
        assert_eq!(intersect(xs, Type::never()), Type::None);
    }

    #[test]
    fn test_simplify_union_with_intersect() {
        let xs = vec![
            Type::any_implicit(),
            intersect(vec![Type::any_implicit(), Type::any_tuple()], Type::never()),
        ];
        assert_eq!(unions(xs), Type::any_implicit());
    }

    #[test]
    fn test_union_empty_with_prefix_variadic_tuple() {
        let xs = vec![
            Type::concrete_tuple(vec![]),
            Type::Tuple(Tuple::unpacked(
                vec![Type::None],
                Type::unbounded_tuple(Type::None),
                Vec::new(),
            )),
        ];
        assert_eq!(unions(xs), Type::unbounded_tuple(Type::None));
    }

    #[test]
    fn test_union_empty_with_suffix_variadic_tuple() {
        let xs = vec![
            Type::concrete_tuple(vec![]),
            Type::Tuple(Tuple::unpacked(
                Vec::new(),
                Type::unbounded_tuple(Type::None),
                vec![Type::None],
            )),
        ];
        assert_eq!(unions(xs), Type::unbounded_tuple(Type::None));
    }
}
