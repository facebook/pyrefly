/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! The experimental `shape_extensions.MapIntTuples` type operator.
//!
//! `MapIntTuples[lambda S: F[S], Shapes]` applies a type-level function to every `IntTuple` in the
//! tuple-like `IntTuples` value `Shapes`. It connects to the shape DSL in both directions:
//!
//! - Forward mapping composes over an `IntTuples` value, including one returned by a DSL function,
//!   and constructs the corresponding tuple of result types.
//! - In a parameter annotation, the same syntax is retained as a pattern. The pattern stores its
//!   mapped member type and source while exposing the ordinary collection view appropriate to the
//!   parameter: `Sequence[...]` for an ordinary parameter or `tuple[..., ...]` for an unpacked
//!   variadic parameter.
//!
//! Concrete sources reduce eagerly, while symbolic sources stay deferred until enough type
//! information is available. The underlying concept is more general: a future operator could map
//! suitable type-level functions over arbitrary tuple element domains. This implementation is
//! deliberately restricted to `IntTuple` elements needed by the shape DSL.

use std::cmp::Ordering;
use std::hash::Hash;
use std::hash::Hasher;

use pyrefly_derive::TypeEq;
use pyrefly_derive::Visit;
use pyrefly_derive::VisitMut;
use pyrefly_python::module_name::ModuleName;
use pyrefly_util::visit::Visit;
use pyrefly_util::visit::VisitMut;
use ruff_python_ast::Identifier;
use ruff_python_ast::name::Name;
use ruff_text_size::TextRange;

use crate::dimension::ShapeError;
use crate::equality::TypeEq;
use crate::equality::TypeEqCtx;
use crate::quantified::AnchorIndex;
use crate::quantified::Quantified;
use crate::quantified::QuantifiedIdentity;
use crate::quantified::QuantifiedOrigin;
use crate::shaped_array::IntTuple;
use crate::tuple::Tuple;
use crate::type_level_dsl::TypeLevelDslCall;
use crate::type_level_dsl::TypeLevelDslFunction;
use crate::type_level_dsl::TypeShapeDslDomain;
use crate::type_var::PreInferenceVariance;
use crate::type_var::Restriction;
use crate::types::Type;

const NORMALIZED_MAPPER_MODULE: &str = "__pyrefly_map_int_tuples_normalization__";

fn normalized_mapper(depth: u32) -> Quantified {
    Quantified::type_var(
        Name::new_static("__map_lambda__"),
        QuantifiedIdentity::new(
            ModuleName::from_str(NORMALIZED_MAPPER_MODULE),
            AnchorIndex::new(TextRange::default(), depth),
            QuantifiedOrigin::NormalizedMapIntTuplesParameter,
        ),
        None,
        Restriction::Unrestricted,
        PreInferenceVariance::Invariant,
    )
}

/// Whether a deferred `MapIntTuples` computes forward from its source or describes a parameter.
///
/// The same syntax has opposite information flow at a function boundary. In
/// `def make[S: IntTuples](s: S) -> MapIntTuples[lambda T: Box[T], S]`, the source `S` is known
/// and the map computes the return type. In
/// `def consume[S: IntTuples](xs: MapIntTuples[lambda T: Box[T], S]) -> S`, the argument is known;
/// the parameter pattern derives an ordinary `Sequence[Box[IntTuple]]` view from the mapped member
/// stored here, and parameter inference recovers `S` from the argument's elements.
#[derive(Debug, Clone, PartialEq, Eq, TypeEq, PartialOrd, Ord, Hash)]
#[derive(Visit, VisitMut)]
pub enum MapIntTuplesInterpretation {
    Forward,
    ParameterPattern { mapped_member: Type },
}

/// A possibly deferred `MapIntTuples[<mapper>, <source>]` operation.
#[derive(Debug, Clone, PartialEq, Eq, TypeEq, PartialOrd, Ord, Hash)]
#[derive(Visit, VisitMut)]
pub struct MapIntTuples {
    mapper: TypeLambda,
    source: Box<Type>,
    interpretation: MapIntTuplesInterpretation,
}

impl MapIntTuples {
    fn new(mapper: TypeLambda, source: Type) -> Self {
        Self {
            mapper,
            source: Box::new(source),
            interpretation: MapIntTuplesInterpretation::Forward,
        }
    }

    /// Returns the mapper, interpretation, and source retained by this operation.
    pub fn parts(&self) -> (&TypeLambda, &MapIntTuplesInterpretation, &Type) {
        (&self.mapper, &self.interpretation, &self.source)
    }

    /// Changes a forward operation into a parameter pattern with the given mapped member type.
    pub fn make_parameter_pattern(&mut self, mapped_member: Type) {
        self.interpretation = MapIntTuplesInterpretation::ParameterPattern { mapped_member };
    }

    pub(crate) fn fallback(&self) -> Type {
        match &self.interpretation {
            MapIntTuplesInterpretation::Forward => {
                Type::unbounded_tuple(self.mapper.apply(IntTuple::shapeless().to_shape_arg_type()))
            }
            MapIntTuplesInterpretation::ParameterPattern { mapped_member } => {
                Type::unbounded_tuple(mapped_member.clone())
            }
        }
    }

    /// Returns `IntTuples` when every mapped element is structurally an `IntTuple`.
    pub(crate) fn result_domain(&self) -> Option<TypeShapeDslDomain> {
        let result = self.mapper.apply(IntTuple::shapeless().to_shape_arg_type());
        match result {
            result if IntTuple::from_shape_arg_or_tuple_carrier(&result).is_some() => {
                Some(TypeShapeDslDomain::IntTuples)
            }
            Type::TypeLevelDslCall(call)
                if call.result_domain() == Some(TypeShapeDslDomain::IntTuple) =>
            {
                Some(TypeShapeDslDomain::IntTuples)
            }
            _ => None,
        }
    }

    pub(crate) fn evaluate(&self) -> Result<Type, ShapeError> {
        if let MapIntTuplesInterpretation::ParameterPattern { mapped_member } = &self.interpretation
        {
            return Ok(Type::unbounded_tuple(mapped_member.clone()));
        }
        self.evaluate_source(&self.source)
    }

    fn evaluate_source(&self, source: &Type) -> Result<Type, ShapeError> {
        match source {
            Type::Tuple(tuple) => self.mapper.apply_to_tuple(tuple.clone()),
            Type::Union(union) => {
                let mut mapped = union
                    .members
                    .iter()
                    .map(|member| self.evaluate_source(member))
                    .collect::<Result<Vec<_>, _>>()?;
                mapped.sort();
                mapped.dedup();
                Ok(match mapped.as_slice() {
                    [result] => result.clone(),
                    _ => Type::union(mapped),
                })
            }
            Type::Any(_) | Type::Quantified(_) | Type::TypeVar(_) | Type::Var(_) => {
                Ok(self.fallback())
            }
            Type::Never(style) => Ok(Type::Never(*style)),
            _ => Err(invalid_source()),
        }
    }

    /// Recurses through the operation while respecting the mapper's bound parameter.
    pub(crate) fn subst_parts_mut(
        &mut self,
        shadowed: &mut Vec<Quantified>,
        f: &mut dyn FnMut(&mut Type, &mut Vec<Quantified>),
    ) {
        f(&mut self.source, shadowed);
        if let MapIntTuplesInterpretation::ParameterPattern { mapped_member } =
            &mut self.interpretation
        {
            f(mapped_member, shadowed);
        }
        // `Quantified` equality is identity-based, so mutating types nested in the binder does not
        // change which body occurrences it shadows.
        self.mapper.parameter.recurse_mut(&mut |ty| f(ty, shadowed));
        let old_len = shadowed.len();
        shadowed.push(self.mapper.parameter.clone());
        f(&mut self.mapper.body, shadowed);
        shadowed.truncate(old_len);
        self.mapper.refresh_normalized_body();
    }
}

impl TypeLevelDslCall {
    /// Constructs a forward `shape_extensions.MapIntTuples` application.
    pub fn map_int_tuples(mapper: TypeLambda, source: Type) -> Self {
        Self {
            function: TypeLevelDslFunction::MapIntTuples(MapIntTuples::new(mapper, source)),
            args: Vec::new(),
        }
    }

    /// Returns the mapper, interpretation, and source of a `MapIntTuples` call.
    pub fn as_map_int_tuples(&self) -> Option<(&TypeLambda, &MapIntTuplesInterpretation, &Type)> {
        let TypeLevelDslFunction::MapIntTuples(map) = &self.function else {
            return None;
        };
        Some(map.parts())
    }

    /// Returns a mutable experimental `shape_extensions.MapIntTuples` application.
    pub fn as_map_int_tuples_mut(&mut self) -> Option<&mut MapIntTuples> {
        let TypeLevelDslFunction::MapIntTuples(map) = &mut self.function else {
            return None;
        };
        Some(map)
    }

    /// Returns the source of an experimental `MapIntTuples` application for recursive
    /// finalization before the map is evaluated.
    pub(crate) fn map_int_tuples_source_mut(&mut self) -> Option<&mut Type> {
        let TypeLevelDslFunction::MapIntTuples(map) = &mut self.function else {
            return None;
        };
        Some(&mut map.source)
    }
}

fn invalid_source() -> ShapeError {
    ShapeError::ShapeComputation {
        message: "Source argument to `MapIntTuples` must be an `IntTuples` value".to_owned(),
    }
}

/// Builds the binder denoted by an `IntTuples` mapper parameter.
///
/// The binder depends only on the parameter's module, name, and source location. Binding and
/// solving can therefore reconstruct the same binder independently, regardless of solve order.
/// Mapper parsing and standalone parameter resolution must both call this constructor so their
/// `QuantifiedIdentity` values compare equal during substitution.
pub fn map_int_tuples_mapper_binder(module: ModuleName, parameter: &Identifier) -> Quantified {
    Quantified::type_var(
        parameter.id.clone(),
        QuantifiedIdentity::new(
            module,
            AnchorIndex::first(parameter.range),
            QuantifiedOrigin::MapIntTuplesParameter,
        ),
        None,
        Restriction::Bound(IntTuple::shapeless().to_shape_arg_type()),
        PreInferenceVariance::Invariant,
    )
}

/// The unary type-level function accepted by experimental `shape_extensions.MapIntTuples`.
///
/// Applying the function substitutes its quantified parameter in the body type. It is stored only
/// inside the experimental map representation and is not a general Python type-system construct.
#[derive(Debug, Clone)]
pub struct TypeLambda {
    parameter: Quantified,
    body: Type,
    normalized_body: Box<Type>,
}

impl TypeLambda {
    /// Creates a unary type-level function with the supplied bound parameter.
    pub fn new(parameter: Quantified, body: Type) -> Self {
        let normalized_body = Box::new(Self::normalize_body(&parameter, &body));
        Self {
            parameter,
            body,
            normalized_body,
        }
    }

    /// The synthetic parameter introduced by the mapper lambda.
    pub fn parameter(&self) -> &Quantified {
        &self.parameter
    }

    /// The type expression produced by applying the mapper.
    pub fn body(&self) -> &Type {
        &self.body
    }

    /// Replaces bound mapper parameters with de Bruijn markers for alpha-equivalent comparison.
    ///
    /// For example, `lambda S: tuple[S, MapIntTuples[lambda T: tuple[S, T], Xs]]` compares equal
    /// to the same expression spelled with `Outer` and `Inner`. The outer occurrence in the
    /// nested body has depth 1 and the inner occurrence has depth 0, so swapping them still
    /// compares unequal even though both source names are discarded.
    fn normalize_body(parameter: &Quantified, body: &Type) -> Type {
        let mut body = body.clone();
        body.transform_mut(&mut |ty| {
            if let Type::Quantified(q) = ty
                && let Some(depth) = q.normalized_map_int_tuples_parameter_depth()
            {
                *ty = Type::Quantified(Box::new(normalized_mapper(depth + 1)));
            }
        });
        body.subst_mut_fn(&mut |candidate| {
            (candidate == parameter).then(|| Type::Quantified(Box::new(normalized_mapper(0))))
        });
        body
    }

    fn refresh_normalized_body(&mut self) {
        *self.normalized_body = Self::normalize_body(&self.parameter, &self.body);
    }

    /// Instantiates the function at one argument.
    pub fn apply(&self, argument: Type) -> Type {
        let mut result = self.body.clone();
        result.subst_mut_fn(&mut |candidate| {
            (candidate == &self.parameter).then(|| argument.clone())
        });
        result
    }

    /// Applies this function to every `IntTuple` in an eagerly reducible tuple.
    pub fn apply_to_tuple(&self, tuple: Tuple) -> Result<Type, ShapeError> {
        let map = |ty: &Type| {
            IntTuple::from_shape_arg_or_tuple_carrier(ty)
                .map(|shape| self.apply(shape.to_shape_arg_type()))
                .ok_or_else(|| ShapeError::ShapeComputation {
                    message:
                        "Source argument to `MapIntTuples` must contain only `IntTuple` values"
                            .to_owned(),
                })
        };
        let map_all = |types: Vec<Type>| types.iter().map(map).collect::<Result<Vec<_>, _>>();

        Ok(match normalize_tuple(tuple)? {
            NormalizedTuple::Fixed(elements) => Type::concrete_tuple(map_all(elements)?),
            NormalizedTuple::Unbounded {
                prefix,
                element,
                suffix,
            } => Type::unpacked_tuple(
                map_all(prefix)?,
                Type::unbounded_tuple(map(&element)?),
                map_all(suffix)?,
            ),
        })
    }
}

impl Visit<Type> for TypeLambda {
    fn recurse<'a>(&'a self, f: &mut dyn FnMut(&'a Type)) {
        self.parameter.visit(f);
        self.body.visit(f);
    }
}

impl VisitMut<Type> for TypeLambda {
    fn recurse_mut(&mut self, f: &mut dyn FnMut(&mut Type)) {
        self.parameter.visit_mut(f);
        f(&mut self.body);
        self.refresh_normalized_body();
    }
}

impl PartialEq for TypeLambda {
    fn eq(&self, other: &Self) -> bool {
        self.parameter.kind() == other.parameter.kind()
            && self.parameter.restriction() == other.parameter.restriction()
            && self.normalized_body == other.normalized_body
    }
}

impl Eq for TypeLambda {}

impl Hash for TypeLambda {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.parameter.kind().hash(state);
        self.parameter.restriction().hash(state);
        self.normalized_body.hash(state);
    }
}

impl Ord for TypeLambda {
    fn cmp(&self, other: &Self) -> Ordering {
        self.parameter
            .kind()
            .cmp(&other.parameter.kind())
            .then_with(|| {
                self.parameter
                    .restriction()
                    .cmp(other.parameter.restriction())
            })
            .then_with(|| self.normalized_body.cmp(&other.normalized_body))
    }
}

impl PartialOrd for TypeLambda {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl TypeEq for TypeLambda {
    fn type_eq(&self, other: &Self, ctx: &mut TypeEqCtx) -> bool {
        self.parameter.kind() == other.parameter.kind()
            && self
                .parameter
                .restriction()
                .type_eq(other.parameter.restriction(), ctx)
            && self
                .normalized_body
                .as_ref()
                .type_eq(other.normalized_body.as_ref(), ctx)
    }
}

/// A tuple reduced to the two forms that an eager elementwise operation can preserve.
enum NormalizedTuple {
    Fixed(Vec<Type>),
    Unbounded {
        prefix: Vec<Type>,
        element: Type,
        suffix: Vec<Type>,
    },
}

fn normalize_tuple(tuple: Tuple) -> Result<NormalizedTuple, ShapeError> {
    match tuple {
        Tuple::Concrete(elements) => Ok(NormalizedTuple::Fixed(elements)),
        Tuple::Unbounded(element) => Ok(NormalizedTuple::Unbounded {
            prefix: Vec::new(),
            element: *element,
            suffix: Vec::new(),
        }),
        Tuple::Unpacked(unpacked) => {
            let (mut prefix, middle, suffix) = unpacked.into_parts();
            if middle.is_kind_type_var_tuple() {
                return Err(ShapeError::ShapeComputation {
                    message: "`MapIntTuples` does not support an unresolved `TypeVarTuple`"
                        .to_owned(),
                });
            }
            let Type::Tuple(middle) = middle else {
                return Err(ShapeError::ShapeComputation {
                    message:
                        "Source argument to `MapIntTuples` must be an eagerly reducible tuple type"
                            .to_owned(),
                });
            };
            match normalize_tuple(middle)? {
                NormalizedTuple::Fixed(elements) => {
                    prefix.extend(elements);
                    prefix.extend(suffix);
                    Ok(NormalizedTuple::Fixed(prefix))
                }
                NormalizedTuple::Unbounded {
                    prefix: middle_prefix,
                    element,
                    suffix: middle_suffix,
                } => {
                    prefix.extend(middle_prefix);
                    let suffix = middle_suffix.into_iter().chain(suffix).collect();
                    Ok(NormalizedTuple::Unbounded {
                        prefix,
                        element,
                        suffix,
                    })
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::hash_map::DefaultHasher;

    use pyrefly_python::module_name::ModuleName;
    use ruff_python_ast::name::Name;
    use ruff_text_size::TextRange;

    use super::*;
    use crate::dimension::Int;
    use crate::equality::TypeEqCtx;
    use crate::lit_int::LitInt;
    use crate::quantified::AnchorIndex;
    use crate::quantified::QuantifiedIdentity;
    use crate::quantified::QuantifiedOrigin;
    use crate::type_level_dsl::TypeLevelDslCall;
    use crate::type_var::PreInferenceVariance;
    use crate::type_var::Restriction;
    use crate::types::AnyStyle;

    fn binder(name: &'static str, index: u32) -> Quantified {
        Quantified::type_var(
            Name::new_static(name),
            QuantifiedIdentity::new(
                ModuleName::from_str("__test__"),
                AnchorIndex::new(TextRange::default(), index),
                QuantifiedOrigin::synthetic(),
            ),
            None,
            Restriction::Unrestricted,
            PreInferenceVariance::Invariant,
        )
    }

    fn identity_lambda() -> TypeLambda {
        let parameter = binder("T", 0);
        TypeLambda::new(parameter.clone(), Type::Quantified(Box::new(parameter)))
    }

    fn tuple_lambda() -> TypeLambda {
        let parameter = identity_lambda().parameter;
        TypeLambda::new(
            parameter.clone(),
            Type::concrete_tuple(vec![Type::Quantified(Box::new(parameter))]),
        )
    }

    fn int_tuple(dimensions: &[i64]) -> Type {
        IntTuple::new(dimensions.iter().copied().map(Int::Literal).collect()).to_shape_arg_type()
    }

    fn hash(call: &TypeLevelDslCall) -> u64 {
        let mut hasher = DefaultHasher::new();
        call.hash(&mut hasher);
        hasher.finish()
    }

    #[test]
    fn maps_fixed_tuple() {
        let source = Tuple::Concrete(vec![int_tuple(&[1]), int_tuple(&[2, 3])]);

        assert_eq!(
            tuple_lambda().apply_to_tuple(source).unwrap(),
            Type::concrete_tuple(vec![
                Type::concrete_tuple(vec![int_tuple(&[1])]),
                Type::concrete_tuple(vec![int_tuple(&[2, 3])]),
            ])
        );
    }

    #[test]
    fn maps_structural_tuple_members() {
        let source = Tuple::Concrete(vec![Type::concrete_tuple(vec![
            LitInt::new(1).to_explicit_type(),
        ])]);

        assert_eq!(
            identity_lambda().apply_to_tuple(source).unwrap(),
            Type::concrete_tuple(vec![int_tuple(&[1])])
        );
    }

    #[test]
    fn substitutes_only_the_parameter_identity() {
        let lambda = identity_lambda();
        let same_name = Quantified::type_var(
            Name::new_static("T"),
            QuantifiedIdentity::new(
                ModuleName::from_str("__test__"),
                AnchorIndex::new(TextRange::default(), 1),
                QuantifiedOrigin::synthetic(),
            ),
            None,
            Restriction::Unrestricted,
            PreInferenceVariance::Invariant,
        );
        let lambda = TypeLambda::new(
            lambda.parameter.clone(),
            Type::concrete_tuple(vec![
                Type::Quantified(Box::new(lambda.parameter)),
                Type::Quantified(Box::new(same_name.clone())),
            ]),
        );

        assert_eq!(
            lambda.apply(Type::None),
            Type::concrete_tuple(vec![Type::None, Type::Quantified(Box::new(same_name))])
        );
    }

    #[test]
    fn maps_unbounded_tuple() {
        let source = Tuple::Unbounded(Box::new(int_tuple(&[1, 2])));

        assert_eq!(
            identity_lambda().apply_to_tuple(source).unwrap(),
            Type::unbounded_tuple(int_tuple(&[1, 2]))
        );
    }

    #[test]
    fn maps_union_with_gradual_source() {
        let exact_source = Type::concrete_tuple(vec![int_tuple(&[1])]);
        let gradual_source = Type::Any(AnyStyle::Implicit);
        let call = TypeLevelDslCall::map_int_tuples(
            identity_lambda(),
            Type::union(vec![exact_source, gradual_source]),
        );

        assert_eq!(
            call.evaluate().unwrap(),
            Type::union(vec![
                Type::concrete_tuple(vec![int_tuple(&[1])]),
                Type::unbounded_tuple(IntTuple::shapeless().to_shape_arg_type()),
            ])
        );
    }

    #[test]
    fn preserves_never_source() {
        let call = TypeLevelDslCall::map_int_tuples(identity_lambda(), Type::never());

        assert_eq!(call.evaluate().unwrap(), Type::never());
    }

    #[test]
    fn preserves_order_around_unbounded_tuple() {
        let source = Tuple::unpacked(
            vec![int_tuple(&[1])],
            Type::unbounded_tuple(int_tuple(&[2])),
            vec![int_tuple(&[3]), int_tuple(&[4])],
        );

        assert_eq!(
            identity_lambda().apply_to_tuple(source).unwrap(),
            Type::unpacked_tuple(
                vec![int_tuple(&[1])],
                Type::unbounded_tuple(int_tuple(&[2])),
                vec![int_tuple(&[3]), int_tuple(&[4])],
            )
        );
    }

    #[test]
    fn rejects_non_int_tuple_member() {
        let error = identity_lambda()
            .apply_to_tuple(Tuple::Concrete(vec![int_tuple(&[1]), Type::None]))
            .unwrap_err();

        assert_eq!(
            error.to_string(),
            "Source argument to `MapIntTuples` must contain only `IntTuple` values"
        );
    }

    #[test]
    fn rejects_unresolved_type_var_tuple() {
        let middle = Type::Quantified(Box::new(Quantified::type_var_tuple(
            Name::new_static("Ts"),
            QuantifiedIdentity::new(
                ModuleName::from_str("__test__"),
                AnchorIndex::new(TextRange::default(), 1),
                QuantifiedOrigin::synthetic(),
            ),
            None,
        )));
        let source = Tuple::unpacked(vec![int_tuple(&[1])], middle, Vec::new());

        assert_eq!(
            identity_lambda()
                .apply_to_tuple(source)
                .unwrap_err()
                .to_string(),
            "`MapIntTuples` does not support an unresolved `TypeVarTuple`"
        );
    }

    #[test]
    fn alpha_equivalent_maps_have_equal_identity() {
        let left =
            TypeLevelDslCall::map_int_tuples(identity_lambda(), Type::Any(AnyStyle::Implicit));
        let parameter = binder("Renamed", 2);
        let right = TypeLevelDslCall::map_int_tuples(
            TypeLambda::new(parameter.clone(), Type::Quantified(Box::new(parameter))),
            Type::Any(AnyStyle::Implicit),
        );

        assert!(left.type_eq(&right, &mut TypeEqCtx::default()));
        assert_eq!(left, right);
        assert_eq!(hash(&left), hash(&right));
        assert_eq!(left.cmp(&right), Ordering::Equal);
    }

    fn nested_map(outer: Quantified, inner: Quantified, swap: bool) -> TypeLevelDslCall {
        let (first, second) = if swap {
            (inner.clone(), outer.clone())
        } else {
            (outer.clone(), inner.clone())
        };
        let inner_map = TypeLevelDslCall::map_int_tuples(
            TypeLambda::new(
                inner,
                Type::concrete_tuple(vec![
                    Type::Quantified(Box::new(first)),
                    Type::Quantified(Box::new(second)),
                ]),
            ),
            Type::Any(AnyStyle::Implicit),
        );
        TypeLevelDslCall::map_int_tuples(
            TypeLambda::new(outer, Type::TypeLevelDslCall(Box::new(inner_map))),
            Type::Any(AnyStyle::Implicit),
        )
    }

    #[test]
    fn nested_alpha_equivalence_preserves_capture_depth() {
        let left = nested_map(binder("LeftOuter", 0), binder("LeftInner", 1), false);
        let renamed = nested_map(binder("RenamedOuter", 2), binder("RenamedInner", 3), false);
        let swapped = nested_map(binder("SwappedOuter", 4), binder("SwappedInner", 5), true);

        assert_eq!(left, renamed);
        assert_eq!(hash(&left), hash(&renamed));
        assert_eq!(left.cmp(&renamed), Ordering::Equal);
        assert_ne!(left, swapped);
        assert_ne!(left.cmp(&swapped), Ordering::Equal);
    }

    #[test]
    fn mutable_visit_updates_lambda_body_and_normalized_identity() {
        let parameter = binder("Original", 0);
        let mut lambda = TypeLambda::new(
            parameter.clone(),
            Type::concrete_tuple(vec![
                Type::None,
                Type::Quantified(Box::new(parameter.clone())),
            ]),
        );
        lambda.visit_mut(&mut |ty| {
            if matches!(ty, Type::Tuple(_)) {
                *ty = Type::concrete_tuple(vec![
                    Type::Any(AnyStyle::Explicit),
                    Type::Quantified(Box::new(parameter.clone())),
                ]);
            }
        });

        let renamed = binder("Renamed", 1);
        let expected = TypeLambda::new(
            renamed.clone(),
            Type::concrete_tuple(vec![
                Type::Any(AnyStyle::Explicit),
                Type::Quantified(Box::new(renamed)),
            ]),
        );
        assert_eq!(
            lambda.apply(Type::None),
            Type::concrete_tuple(vec![Type::Any(AnyStyle::Explicit), Type::None])
        );
        assert_eq!(lambda, expected);

        let left = TypeLevelDslCall::map_int_tuples(lambda, Type::Any(AnyStyle::Implicit));
        let right = TypeLevelDslCall::map_int_tuples(expected, Type::Any(AnyStyle::Implicit));
        assert_eq!(hash(&left), hash(&right));
    }

    #[test]
    fn parameter_pattern_fallback_uses_its_mapped_member() {
        let mut pattern =
            TypeLevelDslCall::map_int_tuples(identity_lambda(), Type::Any(AnyStyle::Implicit));
        let member = int_tuple(&[1]);
        let view = Type::unbounded_tuple(member.clone());
        let forward = pattern.clone();
        pattern
            .as_map_int_tuples_mut()
            .expect("test call should be a MapIntTuples application")
            .make_parameter_pattern(member);

        assert_eq!(pattern.fallback(), view);
        assert_eq!(pattern.evaluate().unwrap(), view);
        assert_ne!(pattern, forward);
    }

    #[test]
    fn substitution_respects_mapper_scope_and_reaches_the_view() {
        let mapper_parameter = identity_lambda().parameter;
        let captured = Quantified::type_var(
            Name::new_static("Captured"),
            QuantifiedIdentity::new(
                ModuleName::from_str("__test__"),
                AnchorIndex::new(TextRange::default(), 3),
                QuantifiedOrigin::synthetic(),
            ),
            None,
            Restriction::Unrestricted,
            PreInferenceVariance::Invariant,
        );
        let mut call = TypeLevelDslCall::map_int_tuples(
            TypeLambda::new(
                mapper_parameter.clone(),
                Type::concrete_tuple(vec![
                    Type::Quantified(Box::new(captured.clone())),
                    Type::Quantified(Box::new(mapper_parameter.clone())),
                ]),
            ),
            Type::Quantified(Box::new(captured.clone())),
        );
        call.as_map_int_tuples_mut()
            .expect("test call should be a MapIntTuples application")
            .make_parameter_pattern(Type::Quantified(Box::new(captured.clone())));
        call.args.push(Type::Quantified(Box::new(captured.clone())));
        let replacement = int_tuple(&[7]);
        let mut ty = Type::TypeLevelDslCall(Box::new(call));

        ty.subst_mut_fn(&mut |candidate| {
            (candidate == &captured || candidate == &mapper_parameter).then(|| replacement.clone())
        });

        let Type::TypeLevelDslCall(call) = ty else {
            unreachable!("substitution preserves a deferred MapIntTuples application")
        };
        let Some((mapper, MapIntTuplesInterpretation::ParameterPattern { mapped_member }, source)) =
            call.as_map_int_tuples()
        else {
            unreachable!("the application remains a parameter pattern")
        };
        assert_eq!(source, &replacement);
        assert_eq!(mapped_member, &replacement);
        assert_eq!(call.args, vec![replacement.clone()]);
        assert_eq!(
            mapper.body,
            Type::concrete_tuple(vec![
                replacement,
                Type::Quantified(Box::new(mapper_parameter)),
            ])
        );
    }
}
