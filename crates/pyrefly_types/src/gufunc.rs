/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Parsing for generalized universal function signatures.
//!
//! Parsing distinguishes the named-dimension subset consumed by shape evaluation from valid
//! extensions that are not modeled yet and malformed signatures that should produce diagnostics.

#![cfg_attr(
    not(test),
    expect(dead_code, reason = "used by the next stacked gufunc evaluator change")
)]

use std::collections::HashMap;
use std::collections::HashSet;

/// A gufunc signature in the single-output, named-dimension subset supported by shape evaluation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct GufuncSignature {
    pub(crate) inputs: Vec<Vec<String>>,
    pub(crate) output: Vec<String>,
}

/// A well-formed signature that shape evaluation does not yet model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GufuncUnsupported {
    MultipleOutputs,
    OptionalDimensions,
    FixedSizeDimensions,
}

impl GufuncUnsupported {
    pub(crate) fn message(self) -> String {
        let feature = match self {
            Self::MultipleOutputs => "multiple outputs",
            Self::OptionalDimensions => "optional core dimensions",
            Self::FixedSizeDimensions => "fixed-size core dimensions",
        };
        format!("gufunc: {feature} are not supported")
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum GufuncSignatureError {
    UnsupportedCharacter(char),
    ArrowCount(usize),
    MissingArguments(&'static str),
    Expected {
        expected: &'static str,
        found: Option<char>,
    },
    InvalidFrozenSize(String),
    InconsistentOptionalDimension(String),
    MissingOutputDimension(String),
}

impl GufuncSignatureError {
    pub(crate) fn message(&self) -> String {
        match self {
            Self::UnsupportedCharacter(character) => {
                format!("gufunc: unsupported character '{character}' in signature")
            }
            Self::ArrowCount(count) => {
                format!("gufunc: signature must contain exactly one '->', got {count}")
            }
            Self::MissingArguments(side) => {
                format!("gufunc: signature must contain at least one {side} argument")
            }
            Self::Expected { expected, found } => match found {
                Some(found) => format!("gufunc: expected {expected}, got '{found}'"),
                None => format!("gufunc: expected {expected}, got end of signature"),
            },
            Self::InvalidFrozenSize(size) => {
                format!("gufunc: expected a valid positive frozen size, got '{size}'")
            }
            Self::InconsistentOptionalDimension(dimension) => format!(
                "gufunc: core dimension '{dimension}' must use '?' consistently in every occurrence"
            ),
            Self::MissingOutputDimension(dimension) => {
                format!("gufunc: output core dimension '{dimension}' not found in inputs")
            }
        }
    }
}

pub(crate) enum GufuncClassification {
    Supported(GufuncSignature),
    Unsupported(GufuncUnsupported),
    Invalid(GufuncSignatureError),
}

#[derive(Default)]
struct ParsedSide {
    arguments: Vec<Vec<ParsedDimension>>,
}

enum ParsedDimension {
    Named { name: String, optional: bool },
    Frozen { size: isize, optional: bool },
}

struct SideParser {
    characters: Vec<char>,
    position: usize,
    side: &'static str,
}

impl SideParser {
    fn new(source: &str, side: &'static str) -> Self {
        Self {
            characters: source.chars().collect(),
            position: 0,
            side,
        }
    }

    fn current(&self) -> Option<char> {
        self.characters.get(self.position).copied()
    }

    fn consume(&mut self, character: char) -> bool {
        if self.current() == Some(character) {
            self.position += 1;
            true
        } else {
            false
        }
    }

    fn skip_whitespace(&mut self) {
        while self
            .current()
            .is_some_and(|character| matches!(character, ' ' | '\t'))
        {
            self.position += 1;
        }
    }

    fn expected<T>(&self, expected: &'static str) -> Result<T, GufuncSignatureError> {
        Err(GufuncSignatureError::Expected {
            expected,
            found: self.current(),
        })
    }

    fn parse(mut self) -> Result<ParsedSide, GufuncSignatureError> {
        self.skip_whitespace();
        if self.current().is_none() {
            return Err(GufuncSignatureError::MissingArguments(self.side));
        }

        let mut parsed = ParsedSide::default();
        loop {
            if !self.consume('(') {
                return self.expected("'('");
            }
            self.skip_whitespace();
            let mut dimensions = Vec::new();
            if !self.consume(')') {
                loop {
                    let start = self.position;
                    let named = self.current().is_some_and(|character| {
                        character.is_ascii_alphabetic() || character == '_'
                    });
                    if named {
                        self.position += 1;
                        while self.current().is_some_and(|character| {
                            character.is_ascii_alphanumeric() || character == '_'
                        }) {
                            self.position += 1;
                        }
                    } else if self
                        .current()
                        .is_some_and(|character| character.is_ascii_digit())
                    {
                        while self
                            .current()
                            .is_some_and(|character| character.is_ascii_digit())
                        {
                            self.position += 1;
                        }
                    } else {
                        return self.expected("a core dimension name, frozen size, or ')'");
                    }
                    let value = self.characters[start..self.position]
                        .iter()
                        .collect::<String>();
                    let optional = self.consume('?');
                    dimensions.push(if named {
                        ParsedDimension::Named {
                            name: value,
                            optional,
                        }
                    } else {
                        let Ok(size) = value.parse::<isize>() else {
                            return Err(GufuncSignatureError::InvalidFrozenSize(value));
                        };
                        // NumPy rejects sizes at or above `NPY_MAX_INTP`; `isize` has the
                        // corresponding platform-sized range.
                        if size <= 0 || size == isize::MAX {
                            return Err(GufuncSignatureError::InvalidFrozenSize(value));
                        }
                        ParsedDimension::Frozen { size, optional }
                    });

                    self.skip_whitespace();
                    if self.consume(')') {
                        break;
                    }
                    if !self.consume(',') {
                        return self.expected("',' or ')'");
                    }
                    self.skip_whitespace();
                }
            }
            parsed.arguments.push(dimensions);

            self.skip_whitespace();
            if self.position == self.characters.len() {
                return Ok(parsed);
            }
            if !self.consume(',') {
                return self.expected("',' or end of signature");
            }
            self.skip_whitespace();
        }
    }
}

/// Parses and classifies a generalized universal function signature.
pub(crate) fn parse_gufunc_signature(spec: &str) -> GufuncClassification {
    if let Some(character) = spec.chars().find(|character| {
        !character.is_ascii_alphanumeric()
            && !matches!(character, '_' | '?' | '(' | ')' | ',' | '-' | '>')
            && !matches!(character, ' ' | '\t')
    }) {
        return GufuncClassification::Invalid(GufuncSignatureError::UnsupportedCharacter(
            character,
        ));
    }

    let arrow_count = spec.matches("->").count();
    if arrow_count != 1 {
        return GufuncClassification::Invalid(GufuncSignatureError::ArrowCount(arrow_count));
    }
    let (inputs, outputs) = spec
        .split_once("->")
        .expect("the signature has exactly one arrow");
    let inputs = match SideParser::new(inputs, "input").parse() {
        Ok(inputs) => inputs,
        Err(error) => return GufuncClassification::Invalid(error),
    };
    let outputs = match SideParser::new(outputs, "output").parse() {
        Ok(outputs) => outputs,
        Err(error) => return GufuncClassification::Invalid(error),
    };

    let mut optional_named_dimensions = HashMap::new();
    let mut optional_frozen_dimensions = HashMap::new();
    for dimension in inputs.arguments.iter().chain(&outputs.arguments).flatten() {
        let inconsistent = match dimension {
            ParsedDimension::Named { name, optional } => optional_named_dimensions
                .insert(name.as_str(), *optional)
                .is_some_and(|previous| previous != *optional),
            ParsedDimension::Frozen { size, optional } => optional_frozen_dimensions
                .insert(*size, *optional)
                .is_some_and(|previous| previous != *optional),
        };
        if inconsistent {
            let name = match dimension {
                ParsedDimension::Named { name, .. } => name.clone(),
                ParsedDimension::Frozen { size, .. } => size.to_string(),
            };
            return GufuncClassification::Invalid(
                GufuncSignatureError::InconsistentOptionalDimension(name),
            );
        }
    }
    let input_dimensions = inputs
        .arguments
        .iter()
        .flatten()
        .filter_map(|dimension| match dimension {
            ParsedDimension::Named { name, .. } => Some(name.as_str()),
            ParsedDimension::Frozen { .. } => None,
        })
        .collect::<HashSet<_>>();
    for dimension in outputs.arguments.iter().flatten() {
        if let ParsedDimension::Named { name, .. } = dimension
            && !input_dimensions.contains(name.as_str())
        {
            return GufuncClassification::Invalid(GufuncSignatureError::MissingOutputDimension(
                name.clone(),
            ));
        }
    }
    if outputs.arguments.len() != 1 {
        return GufuncClassification::Unsupported(GufuncUnsupported::MultipleOutputs);
    }
    if inputs
        .arguments
        .iter()
        .chain(&outputs.arguments)
        .flatten()
        .any(|dimension| match dimension {
            ParsedDimension::Named { optional, .. } | ParsedDimension::Frozen { optional, .. } => {
                *optional
            }
        })
    {
        return GufuncClassification::Unsupported(GufuncUnsupported::OptionalDimensions);
    }
    if inputs
        .arguments
        .iter()
        .chain(&outputs.arguments)
        .flatten()
        .any(|dimension| matches!(dimension, ParsedDimension::Frozen { .. }))
    {
        return GufuncClassification::Unsupported(GufuncUnsupported::FixedSizeDimensions);
    }
    GufuncClassification::Supported(GufuncSignature {
        inputs: inputs
            .arguments
            .into_iter()
            .map(|argument| {
                argument
                    .into_iter()
                    .map(|dimension| match dimension {
                        ParsedDimension::Named { name, .. } => name,
                        ParsedDimension::Frozen { .. } => {
                            unreachable!("frozen dimensions were classified as unsupported")
                        }
                    })
                    .collect()
            })
            .collect(),
        output: outputs
            .arguments
            .into_iter()
            .next()
            .expect("one output")
            .into_iter()
            .map(|dimension| match dimension {
                ParsedDimension::Named { name, .. } => name,
                ParsedDimension::Frozen { .. } => {
                    unreachable!("frozen dimensions were classified as unsupported")
                }
            })
            .collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn classify(spec: &str) -> String {
        match parse_gufunc_signature(spec) {
            GufuncClassification::Supported(signature) => {
                format!(
                    "supported inputs={:?} output={:?}",
                    signature.inputs, signature.output
                )
            }
            GufuncClassification::Unsupported(unsupported) => {
                format!("unsupported: {}", unsupported.message())
            }
            GufuncClassification::Invalid(error) => format!("invalid: {}", error.message()),
        }
    }

    #[test]
    fn supported_signatures() {
        assert_eq!(
            classify("(m,n),(n,p)->(m,p)"),
            "supported inputs=[[\"m\", \"n\"], [\"n\", \"p\"]] output=[\"m\", \"p\"]"
        );
        assert_eq!(classify("(),()->()"), "supported inputs=[[], []] output=[]");
        assert_eq!(
            classify(" ( _batch2 , n ) -> ( n , _batch2 ) "),
            "supported inputs=[[\"_batch2\", \"n\"]] output=[\"n\", \"_batch2\"]"
        );
        assert_eq!(
            classify("(n,n)->(n,n)"),
            "supported inputs=[[\"n\", \"n\"]] output=[\"n\", \"n\"]"
        );
    }

    #[test]
    fn unsupported_signatures_are_distinct_from_invalid_signatures() {
        assert_eq!(
            classify("(n)->(n),(n)"),
            "unsupported: gufunc: multiple outputs are not supported"
        );
        assert_eq!(
            classify("(m,n?),(n?)->(m)"),
            "unsupported: gufunc: optional core dimensions are not supported"
        );
        assert_eq!(
            classify("(3),(n)->(n)"),
            "unsupported: gufunc: fixed-size core dimensions are not supported"
        );
        let largest_frozen_size = format!("({})->()", isize::MAX - 1);
        assert_eq!(
            classify(&largest_frozen_size),
            "unsupported: gufunc: fixed-size core dimensions are not supported"
        );
        assert_eq!(
            classify("(03?),(3?)->()"),
            "unsupported: gufunc: optional core dimensions are not supported"
        );
        assert_eq!(
            classify("(n?)->(missing)"),
            "invalid: gufunc: output core dimension 'missing' not found in inputs"
        );
    }

    #[test]
    fn malformed_signatures() {
        assert_eq!(
            classify("(n),(m)"),
            "invalid: gufunc: signature must contain exactly one '->', got 0"
        );
        assert_eq!(
            classify("(n)->(n)->(n)"),
            "invalid: gufunc: signature must contain exactly one '->', got 2"
        );
        assert_eq!(
            classify("->()"),
            "invalid: gufunc: signature must contain at least one input argument"
        );
        assert_eq!(
            classify("()->"),
            "invalid: gufunc: signature must contain at least one output argument"
        );
        assert_eq!(
            classify("(n,)->()"),
            "invalid: gufunc: expected a core dimension name, frozen size, or ')', got ')'"
        );
        assert_eq!(
            classify("(n)(m)->()"),
            "invalid: gufunc: expected ',' or end of signature, got '('"
        );
        assert_eq!(
            classify("(n)->(m)"),
            "invalid: gufunc: output core dimension 'm' not found in inputs"
        );
        assert_eq!(
            classify("(n!)->()"),
            "invalid: gufunc: unsupported character '!' in signature"
        );
        assert_eq!(
            classify("(n?),(n)->()"),
            "invalid: gufunc: core dimension 'n' must use '?' consistently in every occurrence"
        );
        assert_eq!(
            classify("(n?)->(n)"),
            "invalid: gufunc: core dimension 'n' must use '?' consistently in every occurrence"
        );
        assert_eq!(
            classify("(0)->()"),
            "invalid: gufunc: expected a valid positive frozen size, got '0'"
        );
        let reserved_maximum = format!("({})->()", isize::MAX);
        assert_eq!(
            classify(&reserved_maximum),
            format!(
                "invalid: gufunc: expected a valid positive frozen size, got '{}'",
                isize::MAX
            )
        );
        let overflow = format!("({})->()", isize::MAX as u128 + 1);
        assert_eq!(
            classify(&overflow),
            format!(
                "invalid: gufunc: expected a valid positive frozen size, got '{}'",
                isize::MAX as u128 + 1
            )
        );
        assert_eq!(
            classify("(2n)->()"),
            "invalid: gufunc: expected ',' or ')', got 'n'"
        );
        assert_eq!(
            classify("(03?),(3)->()"),
            "invalid: gufunc: core dimension '3' must use '?' consistently in every occurrence"
        );
        assert_eq!(
            classify("(core dim)->()"),
            "invalid: gufunc: expected ',' or ')', got 'd'"
        );
        assert_eq!(
            classify("(n)- >()"),
            "invalid: gufunc: signature must contain exactly one '->', got 0"
        );
        assert_eq!(classify("n->()"), "invalid: gufunc: expected '(', got 'n'");
        assert_eq!(
            classify("(n->()"),
            "invalid: gufunc: expected ',' or ')', got end of signature"
        );
        assert_eq!(
            classify("(n)->(n"),
            "invalid: gufunc: expected ',' or ')', got end of signature"
        );
    }
}
