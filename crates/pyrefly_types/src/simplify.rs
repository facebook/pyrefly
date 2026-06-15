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
use crate::heap::TypeHeap;
use crate::literal::Lit;
use crate::quantified::Quantified;
use crate::stdlib::Stdlib;
use crate::tuple::Tuple;
use crate::type_var::Restriction;
use crate::typed_dict::TypedDict;
use crate::types::Type;

/// Turn unions of unions into a flattened list for one union, and return the deduped list.
fn flatten_and_dedup(xs: Vec<Type>, heap: &TypeHeap) -> Vec<Type> {
    fn flatten(xs: Vec<Type>, res: &mut Vec<Type>) {
        for x in xs {
            match x {
                Type::Union(u) => flatten(u.members, res),
                Type::Never(_) => {}
                _ => res.push(x),
            }
        }
    }
    let mut flattened = Vec::with_capacity(xs.len());
    flatten(xs, &mut flattened);
    simplify_intersections(&mut flattened, heap);
    // `simplify_intersections` only ever introduces `Never`s, so dropping them in place suffices.
    flattened.retain(|x| !x.is_never());
    flattened.sort();
    flattened.dedup();
    flattened
}

/// Given a list of types to union together,
/// - If there's 0 element in the list, return `Ok` with `Type::never()`.
/// - If there's 1 element in the list, return `Ok` with that element.
/// - Otherwise, return `Err` along with `xs`.
fn try_collapse(mut xs: Vec<Type>, heap: &TypeHeap) -> Result<Type, Vec<Type>> {
    if xs.is_empty() {
        Ok(heap.mk_never())
    } else if xs.len() == 1 {
        Ok(xs.pop().unwrap())
    } else {
        Err(xs)
    }
}

fn simplify_intersections(xs: &mut [Type], heap: &TypeHeap) {
    // Simplify `A | (A & B)` to `A`
    let (mut intersects, non_intersects): (Vec<_>, Vec<_>) =
        xs.iter_mut().partition(|x| matches!(x, Type::Intersect(_)));
    for x in intersects.iter_mut() {
        if let Type::Intersect(y) = x
            && y.0.iter_mut().any(|t| non_intersects.contains(&t))
        {
            **x = heap.mk_never();
        }
    }
}

/// After literal/enum squashing, a union with more than this many members is widened to
/// `Any` to bound type complexity.
static MAX_UNION_MEMBERS: usize = 4096;

fn unions_internal(
    xs: Vec<Type>,
    stdlib: Option<&Stdlib>,
    enum_members: Option<&dyn Fn(&Class) -> Option<usize>>,
    heap: &TypeHeap,
) -> Type {
    try_collapse(xs, heap).unwrap_or_else(|xs| {
        let mut res = flatten_and_dedup(xs, heap);
        if let Some(stdlib) = stdlib {
            collapse_literals(&mut res, stdlib, enum_members.unwrap_or(&|_| None), heap);
            promote_anonymous_typed_dicts(&mut res, stdlib, heap);
        }
        collapse_tuple_unions_with_empty(&mut res, heap);
        collapse_builtins_type(&mut res, heap);
        collapse_wide_tuple_unions(&mut res, heap);
        collapse_quantifieds(&mut res, heap);
        // Second pass: squashing can still leave a pathologically large union (e.g. thousands
        // of distinct class types). Widen anything still over the cap to `Any`.
        if res.len() > MAX_UNION_MEMBERS {
            return heap.mk_any_implicit();
        }
        // `res` is collapsible again if `flatten_and_dedup` drops `xs` to 0 or 1 elements
        try_collapse(res, heap).unwrap_or_else(|members| heap.mk_union(members))
    })
}

/// Union a set of types together, simplifying as much as you can.
pub fn unions(xs: Vec<Type>, heap: &TypeHeap) -> Type {
    unions_internal(xs, None, None, heap)
}

/// Like `unions`, but also simplify away things regarding literals if you can,
/// e.g. `Literal[True, False] ==> bool`.
pub fn unions_with_literals(
    xs: Vec<Type>,
    stdlib: &Stdlib,
    enum_members: &dyn Fn(&Class) -> Option<usize>,
    heap: &TypeHeap,
) -> Type {
    unions_internal(xs, Some(stdlib), Some(enum_members), heap)
}

pub fn intersect(ts: Vec<Type>, fallback: Type, heap: &TypeHeap) -> Type {
    let is_object = |t: &Type| matches!(t, Type::ClassType(cls) if cls.is_builtin("object"));
    let has_non_object = ts.iter().any(|t| !is_object(t));
    let mut flattened = Vec::new();
    for t in ts {
        match t {
            Type::Union(_) => {
                // TODO: Flatten these instead of giving up.
                return fallback;
            }
            Type::Intersect(x) => flattened.extend(x.0),
            t => {
                // `object & T` is just `T`
                if !has_non_object || !is_object(&t) {
                    flattened.push(t);
                }
            }
        }
    }
    flattened.sort();
    flattened.dedup();
    if flattened.is_empty() || flattened.iter().any(|t| t.is_never()) {
        heap.mk_never()
    } else if flattened.len() == 1 {
        flattened.into_iter().next().unwrap()
    } else {
        heap.mk_intersect(flattened, fallback)
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

static MAX_LITERAL_UNION_MEMBERS: usize = 256;
static MAX_ENUM_UNION_MEMBERS: usize = 4096;

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
    heap: &TypeHeap,
) {
    // All literal types we see, plus `true` to indicate the promoted class was found
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
    // Number of literals of these types contained in the union
    let mut strings = 0;
    let mut ints = 0;
    let mut bytes = 0;

    // Invariant (from the sorting order) is that all Literal/Lit values occur
    // before any instances of the types.
    // Therefore we only need to check if a ClassType is already in the map, rather than
    // inserting them all.
    for t in types.iter() {
        match t {
            Type::LiteralString(_) => {
                has_literal_string = true;
                literal_types.insert(stdlib.str().clone(), false);
            }
            Type::Literal(x) => {
                match &x.value {
                    Lit::Bool(true) => has_true = true,
                    Lit::Bool(false) => has_false = true,
                    Lit::Str(_) => {
                        has_specific_str = true;
                        strings += 1;
                    }
                    Lit::Bytes(_) => {
                        bytes += 1;
                    }
                    Lit::Int(_) => {
                        ints += 1;
                    }
                    Lit::Enum(x) => {
                        let v = enums.entry(x.class.clone()).or_insert(0);
                        *v += 1;
                    }
                }
                literal_types.insert(x.value.general_class_type(stdlib).clone(), false);
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

    // True when a literal kind appears too many times to be worth tracking precisely.
    let over_cap = |count: usize| count > MAX_LITERAL_UNION_MEMBERS;

    let enums_to_delete: SmallSet<ClassType> = enums
        .into_iter()
        .filter(|(k, n)| {
            // `enum_members` returns a count only for classes it classifies as enums, and
            // `None` otherwise. That `None` case is reachable, so when we can't get a count
            // we leave the literals alone rather than promoting.
            let Some(num_members) = enum_members(k.class_object()) else {
                return false;
            };
            // Promote to the enum class once every member is present, or once the union
            // exceeds the cap (some generated enums have thousands of members).
            *n >= num_members || *n >= MAX_ENUM_UNION_MEMBERS
        })
        .map(|x| x.0)
        .collect();
    for e in &enums_to_delete {
        if !literal_types.is_empty() && literal_types.get(e) == Some(&false) {
            types.push(heap.mk_class_type(e.clone()));
        }
    }
    // Promote each non-enum literal kind whose count exceeds the cap to its general class
    // (e.g. 257 distinct `str` literals -> `str`). The class is resolved lazily via a fn
    // pointer because the `stdlib` accessors panic before that builtin is bootstrapped, and
    // only an over-cap kind ever needs its class.
    for (count, get_cls) in [
        (strings, Stdlib::str as fn(&Stdlib) -> &ClassType),
        (ints, Stdlib::int),
        (bytes, Stdlib::bytes),
    ] {
        if !over_cap(count) {
            continue;
        }
        let cls = get_cls(stdlib);
        if literal_types.get(cls) == Some(&false) {
            types.push(heap.mk_class_type(cls.clone()));
            // Mark the class as present so the retain step below drops the now-redundant
            // literals (and `LiteralString`, for `str`) rather than leaving e.g.
            // `LiteralString | str`.
            literal_types.insert(cls.clone(), true);
        }
    }

    remove_maximum(&mut any_styles);
    remove_maximum(&mut never_styles);

    if over_cap(ints)
        || over_cap(strings)
        || over_cap(bytes)
        || literal_types.values().any(|x| *x)
        || (has_true && has_false)
        || (has_literal_string && has_specific_str)
        || !any_styles.is_empty()
        || !never_styles.is_empty()
        || !enums_to_delete.is_empty()
    {
        // We actually have some things to delete
        types.retain(|x| match x {
            Type::LiteralString(_) => literal_types.get(stdlib.str()) == Some(&false),
            Type::Literal(x) => {
                match &x.value {
                    Lit::Bool(_) if has_true && has_false => return false,
                    Lit::Str(_) if has_literal_string || over_cap(strings) => return false,
                    Lit::Int(_) if over_cap(ints) => return false,
                    Lit::Bytes(_) if over_cap(bytes) => return false,
                    Lit::Enum(lit_enum) if enums_to_delete.contains(&lit_enum.class) => {
                        return false;
                    }
                    _ => {}
                }
                literal_types.get(x.value.general_class_type(stdlib)) == Some(&false)
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

/// Promote anonymous typed dicts to `dict[str, value_type]`
fn promote_anonymous_typed_dicts(types: &mut [Type], stdlib: &Stdlib, heap: &TypeHeap) {
    for ty in types.iter_mut() {
        if let Type::TypedDict(TypedDict::Anonymous(inner)) = ty {
            let value_type = inner.compute_value_type(heap);
            *ty = heap.mk_class_type(stdlib.dict(stdlib.str().clone().to_type(), value_type));
        }
    }
}

fn collapse_tuple_unions_with_empty(types: &mut Vec<Type>, heap: &TypeHeap) {
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
                    *ty = heap.mk_unbounded_tuple(elem.as_ref().clone());
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
            Type::Unpack(inner) if matches!(*inner, Type::Tuple(Tuple::Concrete(_))) => {
                // Repeated match because pattern guards cannot move out of bindings.
                if let Type::Tuple(Tuple::Concrete(elts)) = *inner {
                    result.extend(elts);
                } else {
                    unreachable!("guarded by matches! above")
                }
            }
            _ => result.push(elt),
        }
    }
    result
}

/// `type[int] | type[str]` => `type[int | str]`
fn collapse_builtins_type(types: &mut Vec<Type>, heap: &TypeHeap) {
    let mut idx = 0;
    let mut first_elt = None;
    let mut additional_elts = Vec::new();
    types.retain(|t| {
        let retain = match t {
            Type::Type(t) if first_elt.is_none() => {
                first_elt = Some((idx, (**t).clone()));
                true
            }
            Type::Type(t) => {
                additional_elts.push((**t).clone());
                false
            }
            _ => true,
        };
        idx += 1;
        retain
    });
    if let Some((idx, first_elt)) = first_elt
        && !additional_elts.is_empty()
    {
        let mut elts = vec![first_elt.clone()];
        elts.extend(additional_elts);
        *(types
            .get_mut(idx)
            .expect("idx out of bounds when collapsing type members in union")) =
            heap.mk_type_of(heap.mk_union(elts));
    }
}

/// A restricted quantified `Q` whose restriction consists of the types `c_1, ..., c_n` is fully
/// covered by the union `(Q & c_1) | ... | (Q & c_n)`: every value of `Q` satisfies one of its
/// restrictions, so the union collapses to just `Q`.
fn collapse_quantifieds(types: &mut Vec<Type>, heap: &TypeHeap) {
    // For each quantified appearing in a `Q & t` member, gather the `t`s.
    let mut quantified_intersects: SmallMap<&Quantified, Vec<(usize, &Type)>> = SmallMap::new();
    for (idx, ty) in types.iter().enumerate() {
        if let Some((q, Some(t))) = ty.as_quantified() {
            quantified_intersects.entry(q).or_default().push((idx, t));
        }
    }

    // A quantified collapses when its restriction is fully covered by the types it is intersected with.
    let mut indices_to_remove = SmallSet::new();
    let mut quantifieds_to_collapse = Vec::new();
    for (q, ts) in quantified_intersects {
        let restrictions = match q.restriction() {
            Restriction::Constraints(cs) => cs.iter().collect(),
            Restriction::Bound(Type::Union(u)) => u.members.iter().collect(),
            Restriction::Bound(b) => vec![b],
            Restriction::Unrestricted => continue,
        };
        if restrictions.iter().all(|r| ts.iter().any(|(_, t)| t == r)) {
            for (idx, t) in ts {
                if restrictions.contains(&t) {
                    indices_to_remove.insert(idx);
                }
            }
            quantifieds_to_collapse.push(q.clone());
        }
    }
    if quantifieds_to_collapse.is_empty() {
        return;
    }

    // Drop the `Q & t` members of every collapsed quantified, then add `Q`.
    let mut idx = 0;
    types.retain(|_| {
        let keep = !indices_to_remove.contains(&idx);
        idx += 1;
        keep
    });
    types.extend(
        quantifieds_to_collapse
            .into_iter()
            .map(|q| heap.mk_quantified(q)),
    );
    types.sort();
    types.dedup();
}

// After a TypeVarTuple gets substituted with a tuple type, try to simplify the type
pub fn simplify_tuples(tuple: Tuple, _heap: &TypeHeap) -> Tuple {
    match tuple {
        Tuple::Concrete(elts) => Tuple::Concrete(flatten_unpacked_concrete_tuples(elts)),
        Tuple::Unpacked(unpacked) => {
            let (prefix, middle, suffix) = *unpacked;
            if prefix.is_empty()
                && suffix.is_empty()
                && let Type::Tuple(middle) = middle
            {
                return middle;
            }
            match middle {
                Type::Tuple(Tuple::Concrete(elts)) => {
                    Tuple::Concrete(flatten_unpacked_concrete_tuples(
                        prefix
                            .into_iter()
                            .chain(elts)
                            .chain(suffix)
                            .collect::<Vec<_>>(),
                    ))
                }
                Type::Tuple(Tuple::Unpacked(m_unpacked)) => {
                    let (m_prefix, m_middle, m_suffix) = *m_unpacked;
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
            }
        }
        _ => tuple,
    }
}

const TUPLE_UNION_WIDENING_THRESHOLD: usize = 256;

/// Collapses unions containing more than `TUPLE_UNION_WIDENING_THRESHOLD` tuple variants
/// into a single tuple type, preventing combinatorial explosion of concrete tuple types at
/// control-flow joins (keeping type complexity linear O(N) rather than exponential O(2^N)).
fn collapse_wide_tuple_unions(res: &mut Vec<Type>, heap: &TypeHeap) {
    let tuples: Vec<&Tuple> = res
        .iter()
        .filter_map(|t| match t {
            Type::Tuple(tuple) => Some(tuple),
            _ => None,
        })
        .collect();
    if tuples.len() <= TUPLE_UNION_WIDENING_THRESHOLD {
        return;
    }
    let mut element_types = Vec::new();
    for tuple in &tuples {
        collect_tuple_elements(tuple, heap, &mut element_types);
    }
    let merged = heap.mk_unbounded_tuple(unions(element_types, heap));
    res.retain(|t| !matches!(t, Type::Tuple(_)));
    res.push(merged);
    res.sort();
    res.dedup();
}

fn collect_tuple_elements(tuple: &Tuple, heap: &TypeHeap, out: &mut Vec<Type>) {
    match tuple {
        Tuple::Concrete(elts) => {
            for elt in elts {
                collect_tuple_member(elt, heap, out);
            }
        }
        Tuple::Unbounded(elem) => {
            out.push((**elem).clone());
        }
        Tuple::Unpacked(unpacked) => {
            let (prefix, middle, suffix) = unpacked.as_ref();
            for elt in prefix {
                collect_tuple_member(elt, heap, out);
            }
            collect_unpacked_member(middle, heap, out);
            for elt in suffix {
                collect_tuple_member(elt, heap, out);
            }
        }
    }
}

fn collect_tuple_member(ty: &Type, heap: &TypeHeap, out: &mut Vec<Type>) {
    match ty {
        Type::Unpack(inner) => {
            collect_unpacked_member(inner, heap, out);
        }
        other => {
            out.push(other.clone());
        }
    }
}

fn collect_unpacked_member(ty: &Type, heap: &TypeHeap, out: &mut Vec<Type>) {
    match ty {
        Type::Tuple(tuple) => {
            collect_tuple_elements(tuple, heap, out);
        }
        Type::Quantified(q) if q.is_type_var_tuple() => {
            out.push(heap.mk_element_of_type_var_tuple((**q).clone()));
        }
        other => {
            out.push(other.clone());
        }
    }
}

#[cfg(test)]
mod tests {
    use pyrefly_python::module_name::ModuleName;
    use ruff_python_ast::name::Name;
    use ruff_text_size::TextRange;

    use crate::heap::TypeHeap;
    use crate::lit_int::LitInt;
    use crate::quantified::AnchorIndex;
    use crate::quantified::Quantified;
    use crate::quantified::QuantifiedIdentity;
    use crate::quantified::QuantifiedKind;
    use crate::quantified::QuantifiedOrigin;
    use crate::simplify::intersect;
    use crate::simplify::unions;
    use crate::tuple::Tuple;
    use crate::type_var::PreInferenceVariance;
    use crate::type_var::Restriction;
    use crate::types::NeverStyle;
    use crate::types::Type;

    #[test]
    fn test_flatten_never() {
        let heap = TypeHeap::new();
        let xs = vec![
            Type::Never(NeverStyle::Never),
            Type::Never(NeverStyle::NoReturn),
        ];
        let res = unions(xs, &heap);
        assert_eq!(res, Type::never());
    }

    #[test]
    fn test_oversized_union_widens_to_any() {
        let heap = TypeHeap::new();
        // More than `MAX_UNION_MEMBERS` distinct members survive squashing, so the union is
        // widened to `Any`.
        let xs: Vec<Type> = (0..=(super::MAX_UNION_MEMBERS as i64))
            .map(|i| LitInt::new(i).to_implicit_type())
            .collect();
        assert!(matches!(unions(xs, &heap), Type::Any(_)));

        // A union under the cap is left untouched.
        let xs: Vec<Type> = (0..8).map(|i| LitInt::new(i).to_implicit_type()).collect();
        assert!(matches!(unions(xs, &heap), Type::Union(_)));
    }

    #[test]
    fn test_intersect_simple() {
        let heap = TypeHeap::new();
        let xs = vec![Type::any_tuple(), Type::any_implicit()];
        assert_eq!(
            intersect(xs.clone(), Type::never(), &heap),
            Type::Intersect(Box::new((xs, Type::never())))
        );
    }

    #[test]
    fn test_intersect_empty() {
        let heap = TypeHeap::new();
        let xs = Vec::new();
        assert_eq!(intersect(xs, Type::any_implicit(), &heap), Type::never())
    }

    #[test]
    fn test_intersect_never() {
        let heap = TypeHeap::new();
        let xs = vec![Type::any_implicit(), Type::never()];
        assert_eq!(intersect(xs, Type::any_implicit(), &heap), Type::never());
    }

    #[test]
    fn test_intersect_one() {
        let heap = TypeHeap::new();
        let xs = vec![Type::None];
        assert_eq!(intersect(xs, Type::never(), &heap), Type::None);
    }

    #[test]
    fn test_simplify_union_with_intersect() {
        let heap = TypeHeap::new();
        let xs = vec![
            Type::any_implicit(),
            intersect(
                vec![Type::any_implicit(), Type::any_tuple()],
                Type::never(),
                &heap,
            ),
        ];
        assert_eq!(unions(xs, &heap), Type::any_implicit());
    }

    #[test]
    fn test_union_empty_with_prefix_variadic_tuple() {
        let heap = TypeHeap::new();
        let xs = vec![
            Type::concrete_tuple(vec![]),
            Type::Tuple(Tuple::unpacked(
                vec![Type::None],
                Type::unbounded_tuple(Type::None),
                Vec::new(),
            )),
        ];
        assert_eq!(unions(xs, &heap), Type::unbounded_tuple(Type::None));
    }

    fn mk_quantified(restriction: Restriction) -> Quantified {
        Quantified::new(
            QuantifiedIdentity::new(
                ModuleName::from_str("__test__"),
                AnchorIndex::new(TextRange::default(), 0),
                QuantifiedOrigin::Pep695,
            ),
            Name::new_static("T"),
            QuantifiedKind::TypeVar,
            None,
            restriction,
            PreInferenceVariance::Invariant,
        )
    }

    #[test]
    fn test_collapse_quantified_constraints() {
        // For `T: (None, tuple)`, `(T & None) | (T & tuple)` collapses to `T`.
        let heap = TypeHeap::new();
        let (c1, c2) = (Type::None, Type::any_tuple());
        let q = mk_quantified(Restriction::Constraints(vec![c1.clone(), c2.clone()]));
        let xs = vec![
            intersect(vec![q.clone().to_type(&heap), c1], Type::never(), &heap),
            intersect(vec![q.clone().to_type(&heap), c2], Type::never(), &heap),
        ];
        assert_eq!(unions(xs, &heap), q.to_type(&heap));
    }

    #[test]
    fn test_collapse_quantified_bound() {
        // For `T: None | tuple`, `(T & None) | (T & tuple)` collapses to `T`.
        let heap = TypeHeap::new();
        let (b1, b2) = (Type::None, Type::any_tuple());
        let bound = unions(vec![b1.clone(), b2.clone()], &heap);
        let q = mk_quantified(Restriction::Bound(bound));
        let xs = vec![
            intersect(vec![q.clone().to_type(&heap), b1], Type::never(), &heap),
            intersect(vec![q.clone().to_type(&heap), b2], Type::never(), &heap),
        ];
        assert_eq!(unions(xs, &heap), q.to_type(&heap));
    }

    #[test]
    fn test_union_empty_with_suffix_variadic_tuple() {
        let heap = TypeHeap::new();
        let xs = vec![
            Type::concrete_tuple(vec![]),
            Type::Tuple(Tuple::unpacked(
                Vec::new(),
                Type::unbounded_tuple(Type::None),
                vec![Type::None],
            )),
        ];
        assert_eq!(unions(xs, &heap), Type::unbounded_tuple(Type::None));
    }

    #[test]
    fn test_collapse_wide_tuple_unions() {
        let heap = TypeHeap::new();
        // Build more than the threshold of distinct-length tuples so the widening fires. Mixed
        // arities can't merge per-position, so they collapse to an unbounded `tuple[None, ...]`.
        let xs = (1..=(super::TUPLE_UNION_WIDENING_THRESHOLD + 1))
            .map(|i| Type::concrete_tuple(vec![Type::None; i]))
            .collect();
        let res = unions(xs, &heap);
        assert_eq!(res, Type::unbounded_tuple(Type::None));
    }

    #[test]
    fn test_collapse_wide_tuple_unions_corner_cases() {
        let heap = TypeHeap::new();

        use pyrefly_python::module_name::ModuleName;
        use ruff_python_ast::name::Name;
        use ruff_text_size::TextRange;
        use ruff_text_size::TextSize;

        use crate::quantified::AnchorIndex;
        use crate::quantified::Quantified;
        use crate::quantified::QuantifiedIdentity;
        use crate::quantified::QuantifiedKind;
        use crate::quantified::QuantifiedOrigin;
        use crate::type_var::PreInferenceVariance;
        use crate::type_var::Restriction;

        let q_id = QuantifiedIdentity::new(
            ModuleName::from_str("test"),
            AnchorIndex::new(TextRange::new(TextSize::new(0), TextSize::new(10)), 0),
            QuantifiedOrigin::ScopedLegacy,
        );
        let tvt = Quantified::new(
            q_id,
            Name::new_static("Ts"),
            QuantifiedKind::TypeVarTuple,
            None,
            Restriction::Unrestricted,
            PreInferenceVariance::Invariant,
        )
        .to_type(&heap);

        let mut xs = vec![
            // A tuple with unpacked concrete tuple: tuple[any_explicit, *tuple[any_implicit, any_error]]
            Type::concrete_tuple(vec![
                Type::any_explicit(),
                Type::Unpack(Box::new(Type::concrete_tuple(vec![
                    Type::any_implicit(),
                    Type::any_error(),
                ]))),
            ]),
            // A tuple with regular nested tuple: tuple[any_explicit, tuple[any_implicit, any_error]]
            Type::concrete_tuple(vec![
                Type::any_explicit(),
                Type::concrete_tuple(vec![Type::any_implicit(), Type::any_error()]),
            ]),
            // A tuple with unpacked TypeVarTuple: tuple[any_explicit, *Ts]
            Type::concrete_tuple(vec![
                Type::any_explicit(),
                Type::Unpack(Box::new(tvt.clone())),
            ]),
            // A tuple with invalid unpacked type: tuple[any_explicit, *None]
            Type::concrete_tuple(vec![
                Type::any_explicit(),
                Type::Unpack(Box::new(Type::None)),
            ]),
            Type::concrete_tuple(vec![Type::any_explicit()]),
            Type::concrete_tuple(vec![Type::any_implicit()]),
            Type::concrete_tuple(vec![Type::any_error()]),
            Type::concrete_tuple(vec![Type::None]),
            Type::concrete_tuple(vec![Type::any_tuple()]),
        ];

        // Pad with distinct-length all-`None` tuples so the union exceeds the widening threshold
        // and the collapse fires. Their only element, `None`, is already part of the expected
        // union, and the varying arities keep the result on the unbounded-widening path.
        xs.extend(
            (3..=(super::TUPLE_UNION_WIDENING_THRESHOLD + 3))
                .map(|i| Type::concrete_tuple(vec![Type::None; i])),
        );

        let res = unions(xs, &heap);
        let expected_element_type = unions(
            vec![
                Type::any_explicit(),
                Type::any_implicit(),
                Type::any_error(),
                Type::concrete_tuple(vec![Type::any_implicit(), Type::any_error()]),
                heap.mk_element_of_type_var_tuple(tvt.as_quantified().unwrap().0.clone()),
                Type::None,
                Type::any_tuple(),
            ],
            &heap,
        );

        assert_eq!(res, Type::unbounded_tuple(expected_element_type));
    }
}
