/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Parsing and output-shape projection for the explicit subset of einsum equations understood by
//! shape evaluation.
//!
//! Parsing has three outcomes. A supported equation has a typed representation that evaluators
//! can consume. An unsupported equation is valid at runtime but uses syntax this implementation
//! does not model, so shape evaluation should fall back silently. An invalid equation is malformed
//! and should produce a user-facing error.
//!
//! Projection checks the equation against the operand ranks and repeated dimensions, then selects
//! the dimensions named by the explicit output. It preserves known and symbolic dimensions when
//! they agree and widens only output dimensions whose equality cannot be established.

use std::collections::HashMap;

use crate::dimension::Int;
use crate::dimension::ShapeError;
use crate::shaped_array::IntTuple;
use crate::shaped_array::IntTupleView;

/// One occurrence of a label: the input term it appears in and its position in that term.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct EinsumLocation {
    pub(crate) input: usize,
    pub(crate) dimension: usize,
}

/// Every occurrence of one label, in source order.
#[derive(Debug, Clone)]
pub(crate) struct EinsumLabel {
    pub(crate) name: char,
    pub(crate) locations: Vec<EinsumLocation>,
}

/// An equation in the explicit subset supported by shape evaluation.
#[derive(Debug, Clone)]
pub(crate) struct EinsumEquation {
    pub(crate) input_ranks: Vec<usize>,
    /// Distinct labels, ordered by their first appearance.
    pub(crate) labels: Vec<EinsumLabel>,
    /// Indices into `labels`, ordered by their appearance in the output term.
    pub(crate) output: Vec<usize>,
}

impl EinsumEquation {
    /// Returns every pair of occurrences constrained to have the same extent.
    ///
    /// This returns all pairs rather than a chain. A consumer that can compare only known
    /// dimensions would otherwise miss a mismatch when an unknown dimension splits the chain.
    pub(crate) fn equalities(&self) -> impl Iterator<Item = (EinsumLocation, EinsumLocation)> + '_ {
        self.labels.iter().flat_map(|label| {
            label
                .locations
                .iter()
                .enumerate()
                .flat_map(|(index, first)| {
                    label.locations[index + 1..]
                        .iter()
                        .map(move |second| (*first, *second))
                })
        })
    }
}

/// A well-formed equation that shape evaluation does not yet model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EinsumUnsupported {
    /// The output is inferred from labels rather than written after `->`.
    ImplicitOutput,
    /// A `...` group whose rank depends on the operands.
    Ellipsis,
}

impl EinsumUnsupported {
    pub(crate) fn message(self) -> String {
        let feature = match self {
            Self::ImplicitOutput => "implicit output",
            Self::Ellipsis => "ellipsis",
        };
        format!("einsum: {feature} equations are not supported")
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum EinsumEquationError {
    UnsupportedCharacter(char),
    IncompleteEllipsis,
    RepeatedEllipsis,
    ArrowCount(usize),
    MissingOutputLabel(char),
    RepeatedOutputLabel(char),
}

impl EinsumEquationError {
    pub(crate) fn message(&self) -> String {
        match self {
            Self::UnsupportedCharacter(character) => {
                format!("einsum: unsupported character '{character}' in equation")
            }
            Self::IncompleteEllipsis => {
                "einsum: incomplete ellipsis in equation; expected '...'".to_owned()
            }
            Self::RepeatedEllipsis => {
                "einsum: each input or output term may contain at most one ellipsis".to_owned()
            }
            Self::ArrowCount(count) => {
                format!("einsum: equation must contain exactly one '->', got {count}")
            }
            Self::MissingOutputLabel(label) => {
                format!("einsum: output index '{label}' not found in inputs")
            }
            Self::RepeatedOutputLabel(label) => {
                format!("einsum: output index '{label}' appears more than once")
            }
        }
    }
}

pub(crate) enum EinsumClassification {
    Supported(EinsumEquation),
    Unsupported(EinsumUnsupported),
    Invalid(EinsumEquationError),
}

/// One lexical unit of an einsum equation. Whitespace is insignificant.
enum EinsumToken {
    Label(char),
    Comma,
    Arrow,
    Ellipsis,
}

fn tokenize(spec: &str) -> Result<Vec<EinsumToken>, EinsumEquationError> {
    let mut tokens = Vec::new();
    let mut characters = spec.chars().peekable();
    while let Some(character) = characters.next() {
        match character {
            _ if character.is_whitespace() => {}
            ',' => tokens.push(EinsumToken::Comma),
            'a'..='z' | 'A'..='Z' => tokens.push(EinsumToken::Label(character)),
            '-' if characters.peek() == Some(&'>') => {
                characters.next();
                tokens.push(EinsumToken::Arrow);
            }
            '.' if characters.peek() == Some(&'.') => {
                characters.next();
                if characters.next_if_eq(&'.').is_none() {
                    return Err(EinsumEquationError::IncompleteEllipsis);
                }
                tokens.push(EinsumToken::Ellipsis);
            }
            _ => return Err(EinsumEquationError::UnsupportedCharacter(character)),
        }
    }
    Ok(tokens)
}

/// Parses and classifies an einsum equation.
pub(crate) fn parse_einsum_equation(spec: &str) -> EinsumClassification {
    let tokens = match tokenize(spec) {
        Ok(tokens) => tokens,
        Err(error) => return EinsumClassification::Invalid(error),
    };
    let arrows = tokens
        .iter()
        .filter(|token| matches!(token, EinsumToken::Arrow))
        .count();
    if arrows > 1 {
        return EinsumClassification::Invalid(EinsumEquationError::ArrowCount(arrows));
    }

    // Locations recorded after an ellipsis are not meaningful because the ellipsis has unknown
    // rank. They are still useful for checking the remaining syntax and are never consumed because
    // an ellipsis equation cannot be classified as supported.
    let mut has_ellipsis = false;
    let mut input_ranks = vec![0];
    let mut labels: Vec<EinsumLabel> = Vec::new();
    let mut label_indices: HashMap<char, usize> = HashMap::new();
    let mut output_labels = Vec::new();
    let mut in_output = false;
    let mut term_has_ellipsis = false;
    for token in tokens {
        match token {
            EinsumToken::Arrow => {
                in_output = true;
                term_has_ellipsis = false;
            }
            EinsumToken::Comma if in_output => {
                return EinsumClassification::Invalid(EinsumEquationError::UnsupportedCharacter(
                    ',',
                ));
            }
            EinsumToken::Comma => {
                input_ranks.push(0);
                term_has_ellipsis = false;
            }
            EinsumToken::Label(label) if in_output => output_labels.push(label),
            EinsumToken::Label(label) => {
                let input = input_ranks.len() - 1;
                let dimension = &mut input_ranks[input];
                let location = EinsumLocation {
                    input,
                    dimension: *dimension,
                };
                *dimension += 1;
                match label_indices.get(&label) {
                    Some(index) => labels[*index].locations.push(location),
                    None => {
                        label_indices.insert(label, labels.len());
                        labels.push(EinsumLabel {
                            name: label,
                            locations: vec![location],
                        });
                    }
                }
            }
            EinsumToken::Ellipsis => {
                if term_has_ellipsis {
                    return EinsumClassification::Invalid(EinsumEquationError::RepeatedEllipsis);
                }
                has_ellipsis = true;
                term_has_ellipsis = true;
            }
        }
    }

    let mut output = Vec::with_capacity(output_labels.len());
    for label in output_labels {
        let Some(index) = label_indices.get(&label).copied() else {
            return EinsumClassification::Invalid(EinsumEquationError::MissingOutputLabel(label));
        };
        if output.contains(&index) {
            return EinsumClassification::Invalid(EinsumEquationError::RepeatedOutputLabel(label));
        }
        output.push(index);
    }

    // Unsupported is decided only after validating the rest of the equation, so an unimplemented
    // feature never hides an independent typo.
    if has_ellipsis {
        return EinsumClassification::Unsupported(EinsumUnsupported::Ellipsis);
    }
    if arrows == 0 {
        return EinsumClassification::Unsupported(EinsumUnsupported::ImplicitOutput);
    }
    EinsumClassification::Supported(EinsumEquation {
        input_ranks,
        labels,
        output,
    })
}

/// Projects the output shape of a supported equation over its operand shapes.
///
/// The caller supplies a fixed operand list; an operand sequence with unknown cardinality cannot
/// justify evaluation. Known operand ranks must match the equation, and repeated literal
/// dimensions must agree. Symbolic dimensions that cannot be shown equal widen only the output
/// dimensions they reach unless a repeated literal constrains their value.
pub(crate) fn evaluate_einsum(
    equation: &EinsumEquation,
    operands: &[IntTuple],
) -> Result<IntTuple, ShapeError> {
    if operands.len() != equation.input_ranks.len() {
        return Err(ShapeError::ShapeComputation {
            message: format!(
                "einsum: expected {} operands, got {}",
                equation.input_ranks.len(),
                operands.len()
            ),
        });
    }

    for (index, expected) in equation.input_ranks.iter().enumerate() {
        if let IntTupleView::Concrete(dimensions) = operands[index].view()
            && dimensions.len() != *expected
        {
            return Err(ShapeError::ShapeComputation {
                message: format!(
                    "einsum: operand {index} expected rank {expected}, got {}",
                    dimensions.len()
                ),
            });
        }
    }

    let dimension = |location: &EinsumLocation| match operands[location.input].view() {
        IntTupleView::Concrete(dimensions) => dimensions.get(location.dimension).cloned(),
        _ => None,
    };
    let extents = equation
        .labels
        .iter()
        .map(|label| {
            let mut extent: Option<Int> = None;
            let mut literal = None;
            let mut agreed = true;
            for location in &label.locations {
                let Some(found) = dimension(location) else {
                    continue;
                };
                if let Int::Literal(value) = &found {
                    match literal {
                        Some(previous) if previous != *value => {
                            return Err(ShapeError::ShapeComputation {
                                message: format!(
                                    "einsum: index '{}' has conflicting dimensions {previous} and {value}",
                                    label.name
                                ),
                            });
                        }
                        _ => literal = Some(*value),
                    }
                }
                match &extent {
                    None => extent = Some(found),
                    Some(known) => agreed &= *known == found,
                }
            }
            Ok(literal
                .map(Int::Literal)
                .or_else(|| extent.filter(|_| agreed)))
        })
        .collect::<Result<Vec<_>, ShapeError>>()?;

    Ok(IntTuple::new(
        equation
            .output
            .iter()
            .map(|label| extents[*label].clone().unwrap_or(Int::Int))
            .collect(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn classify(spec: &str) -> String {
        match parse_einsum_equation(spec) {
            EinsumClassification::Supported(equation) => {
                let output = equation
                    .output
                    .iter()
                    .map(|label| {
                        let location = equation.labels[*label].locations[0];
                        (location.input, location.dimension)
                    })
                    .collect::<Vec<_>>();
                format!(
                    "supported ranks={:?} output={output:?}",
                    equation.input_ranks
                )
            }
            EinsumClassification::Unsupported(unsupported) => {
                format!("unsupported: {}", unsupported.message())
            }
            EinsumClassification::Invalid(error) => format!("invalid: {}", error.message()),
        }
    }

    #[test]
    fn supported_equations() {
        assert_eq!(
            classify("ij,jk->ik"),
            "supported ranks=[2, 2] output=[(0, 0), (1, 1)]"
        );
        assert_eq!(
            classify(" i j , j k -> k i "),
            "supported ranks=[2, 2] output=[(1, 1), (0, 0)]"
        );
        assert_eq!(classify("ij->"), "supported ranks=[2] output=[]");
        assert_eq!(classify("->"), "supported ranks=[0] output=[]");
    }

    #[test]
    fn equalities_include_every_pair() {
        let EinsumClassification::Supported(equation) = parse_einsum_equation("i,i,i->i") else {
            panic!("`i,i,i->i` should be supported");
        };
        let pairs = equation
            .equalities()
            .map(|(first, second)| {
                (
                    (first.input, first.dimension),
                    (second.input, second.dimension),
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(
            pairs,
            vec![((0, 0), (1, 0)), ((0, 0), (2, 0)), ((1, 0), (2, 0))]
        );
    }

    #[test]
    fn unsupported_equations_are_distinct_from_invalid_equations() {
        assert_eq!(
            classify("...ij,...jk->...ik"),
            "unsupported: einsum: ellipsis equations are not supported"
        );
        assert_eq!(
            classify("ij,jk"),
            "unsupported: einsum: implicit output equations are not supported"
        );
        assert_eq!(
            classify(""),
            "unsupported: einsum: implicit output equations are not supported"
        );
        assert_eq!(
            classify("   "),
            "unsupported: einsum: implicit output equations are not supported"
        );
        assert_eq!(
            classify("...ij->...ii"),
            "invalid: einsum: output index 'i' appears more than once"
        );
        assert_eq!(
            classify("...ij,jk->...ix"),
            "invalid: einsum: output index 'x' not found in inputs"
        );
        assert_eq!(
            classify("...ij->...i->j"),
            "invalid: einsum: equation must contain exactly one '->', got 2"
        );
        assert_eq!(
            classify("ij..,jk->ik"),
            "invalid: einsum: incomplete ellipsis in equation; expected '...'"
        );
        assert_eq!(
            classify("...i...->i"),
            "invalid: einsum: each input or output term may contain at most one ellipsis"
        );
        assert_eq!(
            classify("i->......"),
            "invalid: einsum: each input or output term may contain at most one ellipsis"
        );
        assert_eq!(
            classify("ij,!jk"),
            "invalid: einsum: unsupported character '!' in equation"
        );
    }
}
