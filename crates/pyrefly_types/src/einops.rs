/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Parsing and shape evaluation for `einops.rearrange`, `reduce`, and `repeat` patterns.

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
    Anonymous(i64),
    Ellipsis,
    GroupedEllipsis,
}

#[derive(Debug, Clone)]
struct Expression {
    compositions: Vec<Vec<Axis>>,
    axes: HashSet<String>,
    has_ellipsis: bool,
    has_nonunit_anonymous_axes: bool,
}

#[derive(Debug, Clone)]
pub(crate) struct EinopsPattern {
    input: Expression,
    output: Expression,
    operation: EinopsPatternOperation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EinopsPatternUnsupported {
    /// Splitting one input dimension requires lengths for all but one component axis.
    UnresolvedInputComposition,
}

impl EinopsPatternUnsupported {
    pub(crate) fn message(self, operation: EinopsPatternOperation) -> String {
        match self {
            Self::UnresolvedInputComposition => {
                format!(
                    "einops.{}: cannot infer the component axes of an input composition",
                    operation.name()
                )
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum EinopsPatternError {
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
    NonPositiveAnonymousAxis(i64),
    AxisMismatch,
}

impl EinopsPatternError {
    pub(crate) fn message(&self, operation: EinopsPatternOperation) -> String {
        let name = operation.name();
        let detail = match self {
            Self::ArrowCount(count) => {
                return format!(
                    "einops.{name}: pattern must contain exactly one '->', got {count}"
                );
            }
            Self::UnknownCharacter(character) => {
                return format!("einops.{name}: unknown character '{character}' in pattern");
            }
            Self::InvalidAxis(axis) => return format!("einops.{name}: invalid axis '{axis}'"),
            Self::DuplicateAxis(axis) => {
                return format!("einops.{name}: duplicate axis '{axis}' in expression");
            }
            Self::NestedParentheses => "axis compositions cannot be nested",
            Self::UnbalancedParentheses => "unbalanced parentheses in pattern",
            Self::RepeatedEllipsis => "each expression may contain at most one ellipsis",
            Self::InputGroupedEllipsis => "ellipsis cannot appear inside an input composition",
            Self::OutputOnlyEllipsis => "ellipsis appears in the output but not the input",
            Self::NonUnitAnonymousAxis(axis) => {
                return format!("einops.{name}: anonymous axis {axis} is not supported");
            }
            Self::NonPositiveAnonymousAxis(axis) => {
                return format!("einops.{name}: anonymous axis {axis} must be positive");
            }
            Self::AxisMismatch => match operation {
                EinopsPatternOperation::Rearrange => {
                    "named axes must appear on both sides of the pattern"
                }
                EinopsPatternOperation::Reduce => {
                    "output axes must also appear in the input pattern"
                }
                EinopsPatternOperation::Repeat => {
                    "input axes must also appear in the output pattern"
                }
            },
        };
        format!("einops.{name}: {detail}")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EinopsPatternOperation {
    Rearrange,
    Reduce,
    Repeat,
}

impl EinopsPatternOperation {
    fn name(self) -> &'static str {
        match self {
            Self::Rearrange => "rearrange",
            Self::Reduce => "reduce",
            Self::Repeat => "repeat",
        }
    }
}

pub(crate) enum EinopsPatternClassification {
    Supported(EinopsPattern),
    Invalid(EinopsPatternError),
}

pub(crate) fn valid_einops_axis(axis: &str) -> bool {
    axis.chars().next().is_some_and(char::is_alphabetic)
        && axis
            .chars()
            .all(|character| character.is_alphanumeric() || character == '_')
        && !axis.ends_with('_')
}

pub(crate) fn split_einops_arrow(spec: &str) -> Result<(&str, &str), usize> {
    let mut parts = spec.split("->");
    let input = parts.next().expect("split always produces an initial part");
    let Some(output) = parts.next() else {
        return Err(0);
    };
    if parts.next().is_some() {
        return Err(spec.matches("->").count());
    }
    Ok((input, output))
}

fn parse_expression(spec: &str, is_input: bool) -> Result<Expression, EinopsPatternError> {
    let mut compositions = Vec::new();
    let mut axes = HashSet::new();
    let mut group: Option<Vec<Axis>> = None;
    let mut token = String::new();
    let mut has_ellipsis = false;
    let mut has_nonunit_anonymous_axes = false;

    let flush = |token: &mut String,
                 group: &mut Option<Vec<Axis>>,
                 compositions: &mut Vec<Vec<Axis>>,
                 axes: &mut HashSet<String>,
                 has_ellipsis: &mut bool,
                 has_nonunit_anonymous_axes: &mut bool|
     -> Result<(), EinopsPatternError> {
        if token.is_empty() {
            return Ok(());
        }
        let axis = if token == "..." {
            if *has_ellipsis {
                return Err(EinopsPatternError::RepeatedEllipsis);
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
                .map_err(|_| EinopsPatternError::InvalidAxis(token.clone()))?;
            token.clear();
            if value < 1 {
                return Err(EinopsPatternError::NonPositiveAnonymousAxis(value));
            }
            if value == 1 {
                if group.is_none() {
                    compositions.push(Vec::new());
                }
                return Ok(());
            }
            *has_nonunit_anonymous_axes = true;
            let axis = Axis::Anonymous(value);
            match group {
                Some(group) => group.push(axis),
                None => compositions.push(vec![axis]),
            }
            return Ok(());
        } else {
            if !valid_einops_axis(token) {
                return Err(EinopsPatternError::InvalidAxis(token.clone()));
            }
            if !axes.insert(token.clone()) {
                return Err(EinopsPatternError::DuplicateAxis(token.clone()));
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
                &mut has_nonunit_anonymous_axes,
            )?,
            '(' => {
                flush(
                    &mut token,
                    &mut group,
                    &mut compositions,
                    &mut axes,
                    &mut has_ellipsis,
                    &mut has_nonunit_anonymous_axes,
                )?;
                if group.is_some() {
                    return Err(EinopsPatternError::NestedParentheses);
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
                    &mut has_nonunit_anonymous_axes,
                )?;
                let Some(composition) = group.take() else {
                    return Err(EinopsPatternError::UnbalancedParentheses);
                };
                if is_input && composition.contains(&Axis::GroupedEllipsis) {
                    return Err(EinopsPatternError::InputGroupedEllipsis);
                }
                compositions.push(composition);
            }
            '.' => {
                token.push(character);
                if characters.peek() != Some(&'.') && token != "..." {
                    return Err(EinopsPatternError::UnknownCharacter(character));
                }
            }
            character if character.is_alphanumeric() || character == '_' => token.push(character),
            _ => return Err(EinopsPatternError::UnknownCharacter(character)),
        }
    }
    flush(
        &mut token,
        &mut group,
        &mut compositions,
        &mut axes,
        &mut has_ellipsis,
        &mut has_nonunit_anonymous_axes,
    )?;
    if group.is_some() {
        return Err(EinopsPatternError::UnbalancedParentheses);
    }
    Ok(Expression {
        compositions,
        axes,
        has_ellipsis,
        has_nonunit_anonymous_axes,
    })
}

fn parse_pattern(spec: &str) -> Result<(Expression, Expression), EinopsPatternError> {
    let (input, output) = split_einops_arrow(spec).map_err(EinopsPatternError::ArrowCount)?;
    let input = parse_expression(input, true)?;
    let output = parse_expression(output, false)?;
    if output.has_ellipsis && !input.has_ellipsis {
        return Err(EinopsPatternError::OutputOnlyEllipsis);
    }
    Ok((input, output))
}

pub(crate) fn parse_einops_pattern(
    spec: &str,
    operation: EinopsPatternOperation,
) -> EinopsPatternClassification {
    let (input, output) = match parse_pattern(spec) {
        Ok(pattern) => pattern,
        Err(error) => return EinopsPatternClassification::Invalid(error),
    };
    if matches!(operation, EinopsPatternOperation::Rearrange)
        && (input.has_nonunit_anonymous_axes || output.has_nonunit_anonymous_axes)
    {
        let axis = input
            .compositions
            .iter()
            .chain(&output.compositions)
            .flatten()
            .find_map(|axis| match axis {
                Axis::Anonymous(value) => Some(*value),
                _ => None,
            })
            .expect("a non-unit anonymous axis was recorded");
        return EinopsPatternClassification::Invalid(EinopsPatternError::NonUnitAnonymousAxis(
            axis,
        ));
    }
    let axes_match = match operation {
        EinopsPatternOperation::Rearrange => {
            input.axes == output.axes && input.has_ellipsis == output.has_ellipsis
        }
        EinopsPatternOperation::Reduce => {
            output.axes.is_subset(&input.axes) && !output.has_nonunit_anonymous_axes
        }
        EinopsPatternOperation::Repeat => {
            input.axes.is_subset(&output.axes)
                && !input.has_nonunit_anonymous_axes
                && input.has_ellipsis == output.has_ellipsis
        }
    };
    if !axes_match {
        return EinopsPatternClassification::Invalid(EinopsPatternError::AxisMismatch);
    }
    EinopsPatternClassification::Supported(EinopsPattern {
        input,
        output,
        operation,
    })
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
/// `axis_lengths` contains the named lengths passed as keyword arguments to einops. The DSL
/// intrinsics pass an empty map; the map-based interface keeps input-axis splitting and named
/// repetition available to a future call-site integration without reparsing the pattern.
pub(crate) fn evaluate_einops_pattern(
    pattern: &EinopsPattern,
    input: &IntTuple,
    axis_lengths: &HashMap<String, Int>,
) -> Result<IntTuple, ShapeError> {
    let operation = pattern.operation.name();
    if let Some(name) = axis_lengths
        .keys()
        .find(|name| !pattern.input.axes.contains(*name) && !pattern.output.axes.contains(*name))
    {
        return Err(ShapeError::ShapeComputation {
            message: format!("einops.{operation}: axis '{name}' is not used in the pattern"),
        });
    }
    let IntTupleView::Concrete(input_dimensions) = input.view() else {
        return Err(ShapeError::Unsupported {
            message: format!("einops.{operation}: a statically known input rank is required"),
        });
    };
    let fixed_rank = pattern.input.compositions.len() - usize::from(pattern.input.has_ellipsis);
    let ellipsis_rank = match pattern.input.has_ellipsis {
        true if input_dimensions.len() >= fixed_rank => input_dimensions.len() - fixed_rank,
        true => {
            return Err(ShapeError::ShapeComputation {
                message: format!(
                    "einops.{operation}: expected input rank at least {fixed_rank}, got {}",
                    input_dimensions.len()
                ),
            });
        }
        false if input_dimensions.len() == fixed_rank => 0,
        false => {
            return Err(ShapeError::ShapeComputation {
                message: format!(
                    "einops.{operation}: expected input rank {fixed_rank}, got {}",
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
                        "einops.{operation}: expected a unit input axis, got {dimension}"
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
                            "einops.{operation}: axis '{name}' has conflicting dimensions {dimension} and {length}"
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
                message: EinopsPatternUnsupported::UnresolvedInputComposition
                    .message(pattern.operation),
            });
        }
        if let Some(name) = unresolved.first() {
            let known = product(composition.iter().filter_map(|axis| match axis {
                Axis::Named(name) => bindings.get(name).cloned(),
                Axis::Anonymous(value) => Some(Int::Literal(*value)),
                Axis::Ellipsis | Axis::GroupedEllipsis => None,
            }));
            if let (Int::Literal(total), Int::Literal(factor)) = (dimension, &known)
                && (*factor == 0 || total % factor != 0)
            {
                return Err(ShapeError::ShapeComputation {
                    message: format!(
                        "einops.{operation}: dimension {total} cannot be divided into the requested axes"
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
                Axis::Anonymous(value) => Some(Int::Literal(*value)),
                Axis::Ellipsis | Axis::GroupedEllipsis => None,
            }));
            if !dimensions_compatible(dimension, &composed) {
                return Err(ShapeError::ShapeComputation {
                    message: format!(
                        "einops.{operation}: input dimension {dimension} does not match composed dimension {composed}"
                    ),
                });
            }
        }
    }

    if pattern
        .output
        .axes
        .iter()
        .any(|name| !bindings.contains_key(name))
    {
        return Err(ShapeError::Unsupported {
            message: format!("einops.{operation}: a length is required for each new axis"),
        });
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
            Axis::Anonymous(value) => vec![Int::Literal(*value)],
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
        let EinopsPatternClassification::Supported(pattern) =
            parse_einops_pattern(spec, EinopsPatternOperation::Rearrange)
        else {
            panic!("expected a supported pattern: {spec}");
        };
        evaluate_einops_pattern(
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
        let EinopsPatternClassification::Supported(pattern) =
            parse_einops_pattern("(b v) c -> b v c", EinopsPatternOperation::Rearrange)
        else {
            panic!("expected a supported pattern");
        };
        let axis_lengths = HashMap::from([("v".to_owned(), Int::Literal(3))]);
        assert_eq!(
            evaluate_einops_pattern(
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
                evaluate_einops_pattern(
                    &pattern,
                    &IntTuple::new(vec![Int::Literal(dimension), Int::Literal(5)]),
                    &axis_lengths,
                ),
                Err(ShapeError::ShapeComputation { .. })
            ));
        }
    }

    #[test]
    fn evaluates_reduce_and_repeat_patterns() {
        let EinopsPatternClassification::Supported(reduction) =
            parse_einops_pattern("... bucket -> ... ()", EinopsPatternOperation::Reduce)
        else {
            panic!("expected a supported reduction pattern");
        };
        assert_eq!(
            evaluate_einops_pattern(
                &reduction,
                &IntTuple::new(vec![Int::Literal(2), Int::Literal(3), Int::Literal(5)]),
                &HashMap::new(),
            ),
            Ok(IntTuple::new(vec![
                Int::Literal(2),
                Int::Literal(3),
                Int::Literal(1)
            ]))
        );

        let EinopsPatternClassification::Supported(grouped_reduction) =
            parse_einops_pattern("b c (h 2) -> b c h", EinopsPatternOperation::Reduce)
        else {
            panic!("expected a supported grouped reduction pattern");
        };
        assert_eq!(
            evaluate_einops_pattern(
                &grouped_reduction,
                &IntTuple::new(vec![Int::Literal(2), Int::Literal(3), Int::Literal(20)]),
                &HashMap::new(),
            ),
            Ok(IntTuple::new(vec![
                Int::Literal(2),
                Int::Literal(3),
                Int::Literal(10),
            ]))
        );

        let EinopsPatternClassification::Supported(repetition) =
            parse_einops_pattern("b c -> b c copies", EinopsPatternOperation::Repeat)
        else {
            panic!("expected a supported repeat pattern");
        };
        assert_eq!(
            evaluate_einops_pattern(
                &repetition,
                &IntTuple::new(vec![Int::Literal(2), Int::Literal(3)]),
                &HashMap::from([("copies".to_owned(), Int::Literal(4))]),
            ),
            Ok(IntTuple::new(vec![
                Int::Literal(2),
                Int::Literal(3),
                Int::Literal(4),
            ]))
        );
        let EinopsPatternClassification::Supported(anonymous_repetition) =
            parse_einops_pattern("b c -> b c 2", EinopsPatternOperation::Repeat)
        else {
            panic!("expected a supported anonymous-axis repeat pattern");
        };
        assert_eq!(
            evaluate_einops_pattern(
                &anonymous_repetition,
                &IntTuple::new(vec![Int::Literal(2), Int::Literal(3)]),
                &HashMap::new(),
            ),
            Ok(IntTuple::new(vec![
                Int::Literal(2),
                Int::Literal(3),
                Int::Literal(2),
            ]))
        );
    }

    #[test]
    fn rejects_invalid_patterns_and_dimensions() {
        assert!(matches!(
            parse_einops_pattern("b c -> b d", EinopsPatternOperation::Rearrange),
            EinopsPatternClassification::Invalid(EinopsPatternError::AxisMismatch)
        ));
        assert!(matches!(
            parse_einops_pattern("... c -> c", EinopsPatternOperation::Rearrange),
            EinopsPatternClassification::Invalid(EinopsPatternError::AxisMismatch)
        ));
        assert!(matches!(
            parse_einops_pattern("b -> b new", EinopsPatternOperation::Reduce),
            EinopsPatternClassification::Invalid(EinopsPatternError::AxisMismatch)
        ));
        assert!(matches!(
            parse_einops_pattern("b c -> b", EinopsPatternOperation::Repeat),
            EinopsPatternClassification::Invalid(EinopsPatternError::AxisMismatch)
        ));
        assert!(matches!(
            parse_einops_pattern("b 2 -> b 2", EinopsPatternOperation::Rearrange),
            EinopsPatternClassification::Invalid(EinopsPatternError::NonUnitAnonymousAxis(2))
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
