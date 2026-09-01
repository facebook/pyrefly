/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Internal support for the experimental `shape_extensions.MapIntTuples` operation.
//!
//! `MapIntTuples` eagerly applies a type-level function to every `IntTuple` in a tuple. Mapping
//! produces an ordinary tuple type immediately, so the type system does not need a persistent
//! representation for either the function or the mapped result.

use crate::dimension::ShapeError;
use crate::quantified::Quantified;
use crate::shaped_array::IntTuple;
use crate::tuple::Tuple;
use crate::types::Type;

/// The unary type-level function accepted by experimental `shape_extensions.MapIntTuples`.
///
/// Applying the function substitutes its quantified parameter in the body type. It exists only
/// while `MapIntTuples` is reduced eagerly and is not a general Python type-system construct.
#[derive(Debug, Clone)]
pub struct TypeLambda {
    parameter: Quantified,
    body: Type,
}

impl TypeLambda {
    /// Creates a type-level function with one `IntTuple` parameter.
    pub fn new(parameter: Quantified, body: Type) -> Self {
        Self { parameter, body }
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
                    message: "Mapped tuple must contain only `IntTuple` values".to_owned(),
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
                    message: "Cannot eagerly map an unresolved `TypeVarTuple`".to_owned(),
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
    use pyrefly_python::module_name::ModuleName;
    use ruff_python_ast::name::Name;
    use ruff_text_size::TextRange;

    use super::*;
    use crate::dimension::Int;
    use crate::lit_int::LitInt;
    use crate::quantified::AnchorIndex;
    use crate::quantified::QuantifiedIdentity;
    use crate::quantified::QuantifiedOrigin;
    use crate::type_var::PreInferenceVariance;
    use crate::type_var::Restriction;

    fn identity_lambda() -> TypeLambda {
        let parameter = Quantified::type_var(
            Name::new_static("T"),
            QuantifiedIdentity::new(
                ModuleName::from_str("__test__"),
                AnchorIndex::first(TextRange::default()),
                QuantifiedOrigin::synthetic(),
            ),
            None,
            Restriction::Unrestricted,
            PreInferenceVariance::Invariant,
        );
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
            "Mapped tuple must contain only `IntTuple` values"
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
            "Cannot eagerly map an unresolved `TypeVarTuple`"
        );
    }
}
