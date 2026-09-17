/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Parsing and shape evaluation for `einops.rearrange` patterns.

use std::collections::HashMap;
use std::collections::HashSet;

use crate::dimension::Int;
use crate::dimension::ShapeError;
use crate::shaped_array::IntTuple;
use crate::shaped_array::IntTupleView;
use crate::shaped_array::canonicalize_int_dim;
use crate::types::Type;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum Axis {
    Named(String),
    Ellipsis,
    GroupedEllipsis,
}

#[derive(Debug, Clone)]
struct Expression {
    compositions: Vec<Vec<Axis>>,
    axes: HashSet<String>,
    has_ellipsis: bool,
}

#[derive(Debug, Clone)]
pub(crate) struct RearrangePattern {
    input: Expression,
    output: Expression,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RearrangeUnsupported {
    /// Splitting one input dimension requires lengths for all but one component axis.
    UnresolvedInputComposition,
}

impl RearrangeUnsupported {
    pub(crate) fn message(self) -> String {
        match self {
            Self::UnresolvedInputComposition => {
                "einops.rearrange: cannot infer the component axes of an input composition"
                    .to_owned()
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum RearrangePatternError {
    ArrowCount(usize),
    UnknownCharacter(char),
    InvalidAxis(String),
    DuplicateAxis(String),
    NestedParentheses,
    UnbalancedParentheses,
    RepeatedEllipsis,
    InputGroupedEllipsis,
    OutputOnlyEllipsis,
    NonUnitAnonymousAxis(i64),
    AxisMismatch,
}

impl RearrangePatternError {
    pub(crate) fn message(&self) -> String {
        let detail = match self {
            Self::ArrowCount(count) => {
                return format!(
                    "einops.rearrange: pattern must contain exactly one '->', got {count}"
                );
            }
            Self::UnknownCharacter(character) => {
                return format!("einops.rearrange: unknown character '{character}' in pattern");
            }
            Self::InvalidAxis(axis) => return format!("einops.rearrange: invalid axis '{axis}'"),
            Self::DuplicateAxis(axis) => {
                return format!("einops.rearrange: duplicate axis '{axis}' in expression");
            }
            Self::NestedParentheses => "axis compositions cannot be nested",
            Self::UnbalancedParentheses => "unbalanced parentheses in pattern",
            Self::RepeatedEllipsis => "each expression may contain at most one ellipsis",
            Self::InputGroupedEllipsis => "ellipsis cannot appear inside an input composition",
            Self::OutputOnlyEllipsis => "ellipsis appears in the output but not the input",
            Self::NonUnitAnonymousAxis(axis) => {
                return format!(
                    "einops.rearrange: anonymous axis {axis} is not supported in rearrange"
                );
            }
            Self::AxisMismatch => "named axes must appear on both sides of the pattern",
        };
        format!("einops.rearrange: {detail}")
    }
}

pub(crate) enum RearrangeClassification {
    Supported(RearrangePattern),
    Invalid(RearrangePatternError),
}

fn parse_expression(spec: &str, is_input: bool) -> Result<Expression, RearrangePatternError> {
    let mut compositions = Vec::new();
    let mut axes = HashSet::new();
    let mut group: Option<Vec<Axis>> = None;
    let mut token = String::new();
    let mut has_ellipsis = false;

    let flush = |token: &mut String,
                 group: &mut Option<Vec<Axis>>,
                 compositions: &mut Vec<Vec<Axis>>,
                 axes: &mut HashSet<String>,
                 has_ellipsis: &mut bool|
     -> Result<(), RearrangePatternError> {
        if token.is_empty() {
            return Ok(());
        }
        let axis = if token == "..." {
            if *has_ellipsis {
                return Err(RearrangePatternError::RepeatedEllipsis);
            }
            *has_ellipsis = true;
            if group.is_some() {
                Axis::GroupedEllipsis
            } else {
                Axis::Ellipsis
            }
        } else if token.chars().all(|character| character.is_ascii_digit()) {
            let value = token
                .parse::<i64>()
                .map_err(|_| RearrangePatternError::InvalidAxis(token.clone()))?;
            if value != 1 {
                return Err(RearrangePatternError::NonUnitAnonymousAxis(value));
            }
            token.clear();
            if group.is_none() {
                compositions.push(Vec::new());
            }
            return Ok(());
        } else {
            let valid = token.chars().next().is_some_and(char::is_alphabetic)
                && token
                    .chars()
                    .all(|character| character.is_alphanumeric() || character == '_')
                && !token.ends_with('_');
            if !valid {
                return Err(RearrangePatternError::InvalidAxis(token.clone()));
            }
            if !axes.insert(token.clone()) {
                return Err(RearrangePatternError::DuplicateAxis(token.clone()));
            }
            Axis::Named(token.clone())
        };
        token.clear();
        match group {
            Some(group) => group.push(axis),
            None => compositions.push(vec![axis]),
        }
        Ok(())
    };

    let mut characters = spec.chars().peekable();
    while let Some(character) = characters.next() {
        match character {
            character if character.is_whitespace() => flush(
                &mut token,
                &mut group,
                &mut compositions,
                &mut axes,
                &mut has_ellipsis,
            )?,
            '(' => {
                flush(
                    &mut token,
                    &mut group,
                    &mut compositions,
                    &mut axes,
                    &mut has_ellipsis,
                )?;
                if group.is_some() {
                    return Err(RearrangePatternError::NestedParentheses);
                }
                group = Some(Vec::new());
            }
            ')' => {
                flush(
                    &mut token,
                    &mut group,
                    &mut compositions,
                    &mut axes,
                    &mut has_ellipsis,
                )?;
                let Some(composition) = group.take() else {
                    return Err(RearrangePatternError::UnbalancedParentheses);
                };
                if is_input && composition.contains(&Axis::GroupedEllipsis) {
                    return Err(RearrangePatternError::InputGroupedEllipsis);
                }
                compositions.push(composition);
            }
            '.' => {
                token.push(character);
                if characters.peek() != Some(&'.') && token != "..." {
                    return Err(RearrangePatternError::UnknownCharacter(character));
                }
            }
            character if character.is_alphanumeric() || character == '_' => token.push(character),
            _ => return Err(RearrangePatternError::UnknownCharacter(character)),
        }
    }
    flush(
        &mut token,
        &mut group,
        &mut compositions,
        &mut axes,
        &mut has_ellipsis,
    )?;
    if group.is_some() {
        return Err(RearrangePatternError::UnbalancedParentheses);
    }
    Ok(Expression {
        compositions,
        axes,
        has_ellipsis,
    })
}

pub(crate) fn parse_rearrange_pattern(spec: &str) -> RearrangeClassification {
    let parts = spec.split("->").collect::<Vec<_>>();
    if parts.len() != 2 {
        return RearrangeClassification::Invalid(RearrangePatternError::ArrowCount(
            parts.len().saturating_sub(1),
        ));
    }
    let input = match parse_expression(parts[0], true) {
        Ok(input) => input,
        Err(error) => return RearrangeClassification::Invalid(error),
    };
    let output = match parse_expression(parts[1], false) {
        Ok(output) => output,
        Err(error) => return RearrangeClassification::Invalid(error),
    };
    if output.has_ellipsis && !input.has_ellipsis {
        return RearrangeClassification::Invalid(RearrangePatternError::OutputOnlyEllipsis);
    }
    if input.axes != output.axes || input.has_ellipsis != output.has_ellipsis {
        return RearrangeClassification::Invalid(RearrangePatternError::AxisMismatch);
    }
    RearrangeClassification::Supported(RearrangePattern { input, output })
}

fn product(dimensions: impl IntoIterator<Item = Int>) -> Int {
    canonicalize_int_dim(dimensions.into_iter().fold(Int::Literal(1), |left, right| {
        Int::mul(Type::Int(left), Type::Int(right))
    }))
}

fn dimensions_compatible(left: &Int, right: &Int) -> bool {
    left == right
        || !matches!((left, right), (Int::Literal(left), Int::Literal(right)) if left != right)
}

/// Computes the shape produced by a parsed rearrange pattern.
///
/// `axis_lengths` contains the named lengths passed as keyword arguments to einops. The initial
/// DSL intrinsic passes an empty map; the map-based interface keeps input-axis splitting available
/// to a future call-site integration without reparsing the pattern.
pub(crate) fn evaluate_rearrange(
    pattern: &RearrangePattern,
    input: &IntTuple,
    axis_lengths: &HashMap<String, Int>,
) -> Result<IntTuple, ShapeError> {
    if let Some(name) = axis_lengths
        .keys()
        .find(|name| !pattern.input.axes.contains(*name))
    {
        return Err(ShapeError::ShapeComputation {
            message: format!("einops.rearrange: axis '{name}' is not used in the pattern"),
        });
    }
    let IntTupleView::Concrete(input_dimensions) = input.view() else {
        return Err(ShapeError::Unsupported {
            message: "einops.rearrange: a statically known input rank is required".to_owned(),
        });
    };
    let fixed_rank = pattern.input.compositions.len() - usize::from(pattern.input.has_ellipsis);
    let ellipsis_rank = match pattern.input.has_ellipsis {
        true if input_dimensions.len() >= fixed_rank => input_dimensions.len() - fixed_rank,
        true => {
            return Err(ShapeError::ShapeComputation {
                message: format!(
                    "einops.rearrange: expected input rank at least {fixed_rank}, got {}",
                    input_dimensions.len()
                ),
            });
        }
        false if input_dimensions.len() == fixed_rank => 0,
        false => {
            return Err(ShapeError::ShapeComputation {
                message: format!(
                    "einops.rearrange: expected input rank {fixed_rank}, got {}",
                    input_dimensions.len()
                ),
            });
        }
    };

    let mut bindings = axis_lengths.clone();
    let mut ellipsis = Vec::new();
    let mut input_index = 0;
    for composition in &pattern.input.compositions {
        if composition == &[Axis::Ellipsis] {
            ellipsis = input_dimensions[input_index..input_index + ellipsis_rank].to_vec();
            input_index += ellipsis_rank;
            continue;
        }
        let dimension = &input_dimensions[input_index];
        input_index += 1;
        if composition.is_empty() {
            if matches!(dimension, Int::Literal(value) if *value != 1) {
                return Err(ShapeError::ShapeComputation {
                    message: format!(
                        "einops.rearrange: expected a unit input axis, got {dimension}"
                    ),
                });
            }
            continue;
        }
        if let [Axis::Named(name)] = composition.as_slice() {
            match bindings.get(name) {
                Some(length) if !dimensions_compatible(dimension, length) => {
                    return Err(ShapeError::ShapeComputation {
                        message: format!(
                            "einops.rearrange: axis '{name}' has conflicting dimensions {dimension} and {length}"
                        ),
                    });
                }
                Some(_) => {}
                None => {
                    bindings.insert(name.clone(), dimension.clone());
                }
            }
            continue;
        }

        let unresolved = composition
            .iter()
            .filter_map(|axis| match axis {
                Axis::Named(name) if !bindings.contains_key(name) => Some(name),
                _ => None,
            })
            .collect::<Vec<_>>();
        if unresolved.len() > 1 {
            return Err(ShapeError::Unsupported {
                message: RearrangeUnsupported::UnresolvedInputComposition.message(),
            });
        }
        if let Some(name) = unresolved.first() {
            let known = product(composition.iter().filter_map(|axis| match axis {
                Axis::Named(name) => bindings.get(name).cloned(),
                Axis::Ellipsis | Axis::GroupedEllipsis => None,
            }));
            if let (Int::Literal(total), Int::Literal(factor)) = (dimension, &known)
                && (*factor == 0 || total % factor != 0)
            {
                return Err(ShapeError::ShapeComputation {
                    message: format!(
                        "einops.rearrange: dimension {total} cannot be divided into the requested axes"
                    ),
                });
            }
            bindings.insert(
                (*name).clone(),
                Int::floor_div(Type::Int(dimension.clone()), Type::Int(known)),
            );
        } else {
            let composed = product(composition.iter().filter_map(|axis| match axis {
                Axis::Named(name) => bindings.get(name).cloned(),
                Axis::Ellipsis | Axis::GroupedEllipsis => None,
            }));
            if !dimensions_compatible(dimension, &composed) {
                return Err(ShapeError::ShapeComputation {
                    message: format!(
                        "einops.rearrange: input dimension {dimension} does not match composed dimension {composed}"
                    ),
                });
            }
        }
    }

    let mut output = Vec::new();
    for composition in &pattern.output.compositions {
        if composition == &[Axis::Ellipsis] {
            output.extend(ellipsis.iter().cloned());
            continue;
        }
        let dimensions = composition.iter().flat_map(|axis| match axis {
            Axis::Named(name) => vec![
                bindings
                    .get(name)
                    .expect("parsed output axes must be bound by the input")
                    .clone(),
            ],
            Axis::Ellipsis | Axis::GroupedEllipsis => ellipsis.clone(),
        });
        output.push(product(dimensions));
    }
    Ok(IntTuple::new(output))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn evaluate(spec: &str, dimensions: &[i64]) -> Result<IntTuple, ShapeError> {
        let RearrangeClassification::Supported(pattern) = parse_rearrange_pattern(spec) else {
            panic!("expected a supported pattern: {spec}");
        };
        evaluate_rearrange(
            &pattern,
            &IntTuple::new(dimensions.iter().copied().map(Int::Literal).collect()),
            &HashMap::new(),
        )
    }

    #[test]
    fn evaluates_permutations_compositions_singletons_and_ellipsis() {
        assert_eq!(
            evaluate("b c h w -> b h w c", &[2, 3, 5, 7]),
            Ok(IntTuple::new(
                vec![2, 5, 7, 3].into_iter().map(Int::Literal).collect()
            ))
        );
        assert_eq!(
            evaluate("b v c h w -> (b v) c (h w)", &[2, 3, 4, 5, 7]),
            Ok(IntTuple::new(
                vec![6, 4, 35].into_iter().map(Int::Literal).collect()
            ))
        );
        assert_eq!(
            evaluate("h w -> () h w", &[5, 7]),
            Ok(IntTuple::new(
                vec![1, 5, 7].into_iter().map(Int::Literal).collect()
            ))
        );
        assert_eq!(
            evaluate("... c -> (...) c", &[2, 3, 5]),
            Ok(IntTuple::new(
                vec![6, 5].into_iter().map(Int::Literal).collect()
            ))
        );
    }

    #[test]
    fn evaluates_input_compositions_with_named_lengths() {
        let RearrangeClassification::Supported(pattern) =
            parse_rearrange_pattern("(b v) c -> b v c")
        else {
            panic!("expected a supported pattern");
        };
        let axis_lengths = HashMap::from([("v".to_owned(), Int::Literal(3))]);
        assert_eq!(
            evaluate_rearrange(
                &pattern,
                &IntTuple::new(vec![Int::Literal(6), Int::Literal(5)]),
                &axis_lengths,
            ),
            Ok(IntTuple::new(vec![
                Int::Literal(2),
                Int::Literal(3),
                Int::Literal(5),
            ]))
        );

        for (dimension, axis_lengths) in [
            (7, HashMap::from([("v".to_owned(), Int::Literal(3))])),
            (
                7,
                HashMap::from([
                    ("b".to_owned(), Int::Literal(2)),
                    ("v".to_owned(), Int::Literal(3)),
                ]),
            ),
            (6, HashMap::from([("v".to_owned(), Int::Literal(0))])),
        ] {
            assert!(matches!(
                evaluate_rearrange(
                    &pattern,
                    &IntTuple::new(vec![Int::Literal(dimension), Int::Literal(5)]),
                    &axis_lengths,
                ),
                Err(ShapeError::ShapeComputation { .. })
            ));
        }
    }

    #[test]
    fn rejects_invalid_patterns_and_dimensions() {
        assert!(matches!(
            parse_rearrange_pattern("b c -> b d"),
            RearrangeClassification::Invalid(RearrangePatternError::AxisMismatch)
        ));
        assert!(matches!(
            parse_rearrange_pattern("... c -> c"),
            RearrangeClassification::Invalid(RearrangePatternError::AxisMismatch)
        ));
        assert!(matches!(
            evaluate("b c -> c b", &[2]),
            Err(ShapeError::ShapeComputation { .. })
        ));
        assert!(matches!(
            evaluate("(b v) c -> b v c", &[6, 5]),
            Err(ShapeError::Unsupported { .. })
        ));
    }
}
