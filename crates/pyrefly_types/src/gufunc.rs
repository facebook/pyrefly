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
    expect(
        dead_code,
        reason = "used by the next stacked gufunc DSL integration change"
    )
)]

use std::collections::HashMap;
use std::collections::HashSet;
use std::iter::repeat_n;

use crate::dimension::Int;
use crate::dimension::ShapeError;
use crate::shaped_array::IntTuple;
use crate::shaped_array::IntTupleView;
use crate::shaped_array::broadcast_dim;
use crate::shaped_array::broadcast_shapes;
use crate::shaped_array::is_gradual_shape_middle;

/// A gufunc signature in the single-output, named-dimension subset supported by shape evaluation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct GufuncSignature {
    inputs: Vec<Vec<String>>,
    output: Vec<String>,
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

struct GufuncOperand {
    batch: IntTuple,
    core: Vec<Option<Int>>,
}

fn split_operand(
    shape: &IntTuple,
    core_rank: usize,
    index: usize,
) -> Result<GufuncOperand, ShapeError> {
    match shape.view() {
        IntTupleView::Concrete(dimensions) => {
            if dimensions.len() < core_rank {
                return Err(ShapeError::ShapeComputation {
                    message: format!(
                        "gufunc: operand {index} requires at least rank {core_rank}, got {}",
                        dimensions.len()
                    ),
                });
            }
            let split = dimensions.len() - core_rank;
            Ok(GufuncOperand {
                batch: IntTuple::new(dimensions[..split].to_vec()),
                core: dimensions[split..].iter().cloned().map(Some).collect(),
            })
        }
        IntTupleView::Gradual => Ok(GufuncOperand {
            batch: IntTuple::shapeless(),
            core: vec![None; core_rank],
        }),
        IntTupleView::Unpacked {
            prefix,
            middle,
            suffix,
        } if suffix.len() >= core_rank => {
            let split = suffix.len() - core_rank;
            Ok(GufuncOperand {
                batch: IntTuple::unpacked(
                    prefix.to_vec(),
                    middle.clone(),
                    suffix[..split].to_vec(),
                ),
                core: suffix[split..].iter().cloned().map(Some).collect(),
            })
        }
        IntTupleView::Unpacked { suffix, .. } => Ok(GufuncOperand {
            // The variadic middle may supply any of the trailing core dimensions. Neither its
            // boundary nor the batch dimensions to its left can be aligned soundly, but the
            // fixed suffix still occupies the rightmost core positions.
            batch: IntTuple::shapeless(),
            core: repeat_n(None, core_rank - suffix.len())
                .chain(suffix.iter().cloned().map(Some))
                .collect(),
        }),
    }
}

fn known_batch_suffix(batch: &IntTuple) -> &[Int] {
    match batch.view() {
        IntTupleView::Concrete(dimensions) => dimensions,
        IntTupleView::Gradual => &[],
        IntTupleView::Unpacked { suffix, .. } => suffix,
    }
}

/// Broadcast three or more batches collectively so abstract precision and errors do not depend on
/// operand order. A variadic operand makes the leading rank gradual, while all known right-aligned
/// dimensions remain available for reconciliation.
fn broadcast_batches(batches: &[IntTuple]) -> Result<IntTuple, ShapeError> {
    match batches {
        [] => unreachable!("a parsed gufunc signature has at least one input"),
        [batch] => return Ok(batch.clone()),
        [left, right] => return broadcast_shapes(left, right),
        _ => {}
    }
    if batches.windows(2).all(|pair| pair[0] == pair[1]) {
        return Ok(batches[0].clone());
    }

    // A named variadic cannot absorb unmatched concrete batch dimensions because their alignment
    // with the unknown middle is ambiguous. Preserve that binary-broadcast constraint before the
    // collective calculation widens variadic ranks to gradual ones.
    for batch in batches {
        if let IntTupleView::Unpacked { middle, .. } = batch.view()
            && !is_gradual_shape_middle(middle)
        {
            for concrete in batches {
                if matches!(concrete.view(), IntTupleView::Concrete(_)) {
                    broadcast_shapes(batch, concrete)?;
                }
            }
        }
    }

    let has_variadic_rank = batches
        .iter()
        .any(|batch| !matches!(batch.view(), IntTupleView::Concrete(_)));
    let suffix_rank = batches
        .iter()
        .map(|batch| known_batch_suffix(batch).len())
        .max()
        .unwrap_or(0);
    let result = (0..suffix_rank)
        .rev()
        .map(|offset| {
            let mut dimensions = batches
                .iter()
                .filter_map(|batch| {
                    let suffix = known_batch_suffix(batch);
                    suffix
                        .len()
                        .checked_sub(offset + 1)
                        .map(|index| suffix[index].clone())
                        .or_else(|| {
                            (!matches!(batch.view(), IntTupleView::Concrete(_))).then_some(Int::Int)
                        })
                })
                .collect::<Vec<_>>();
            dimensions.sort();
            dimensions
                .into_iter()
                .try_fold(Int::Literal(1), |result, dimension| {
                    broadcast_dim(&result, &dimension, suffix_rank - offset - 1)
                })
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(if has_variadic_rank {
        IntTuple::unpacked(
            Vec::new(),
            IntTuple::shapeless().to_shape_arg_type(),
            result,
        )
    } else {
        IntTuple::new(result)
    })
}

/// Evaluate the output shape of a supported single-output gufunc signature.
pub(crate) fn evaluate_gufunc(
    signature: &GufuncSignature,
    operands: &[IntTuple],
) -> Result<IntTuple, ShapeError> {
    if operands.len() != signature.inputs.len() {
        return Err(ShapeError::ShapeComputation {
            message: format!(
                "gufunc: expected {} operands, got {}",
                signature.inputs.len(),
                operands.len()
            ),
        });
    }

    // This is the historical broadcast operation and must retain its exact behavior for gradual
    // and variadic shapes as well as concrete ones.
    if signature.inputs.len() == 2
        && signature.inputs.iter().all(Vec::is_empty)
        && signature.output.is_empty()
    {
        return broadcast_shapes(&operands[0], &operands[1]);
    }

    let operands = operands
        .iter()
        .zip(&signature.inputs)
        .enumerate()
        .map(|(index, (shape, dimensions))| split_operand(shape, dimensions.len(), index))
        .collect::<Result<Vec<_>, _>>()?;

    let mut extents: HashMap<&str, Option<Int>> = HashMap::new();
    let mut literals: HashMap<&str, i64> = HashMap::new();
    for (operand, names) in operands.iter().zip(&signature.inputs) {
        for (name, dimension) in names.iter().zip(&operand.core) {
            let Some(dimension) = dimension else {
                continue;
            };
            if let Int::Literal(value) = dimension
                && let Some(previous) = literals.insert(name, *value)
                && previous != *value
            {
                return Err(ShapeError::ShapeComputation {
                    message: format!(
                        "gufunc: core dimension '{name}' has conflicting extents {previous} and {value}"
                    ),
                });
            }
            extents
                .entry(name)
                .and_modify(|extent| {
                    if extent.as_ref().is_some_and(|known| known != dimension) {
                        *extent = None;
                    }
                })
                .or_insert_with(|| Some(dimension.clone()));
        }
    }

    let output_core = IntTuple::new(
        signature
            .output
            .iter()
            .map(|name| {
                literals
                    .get(name.as_str())
                    .copied()
                    .map(Int::Literal)
                    .or_else(|| extents.get(name.as_str()).cloned().flatten())
                    .unwrap_or(Int::Int)
            })
            .collect(),
    );
    let batches = operands
        .iter()
        .map(|operand| operand.batch.clone())
        .collect::<Vec<_>>();
    Ok(broadcast_batches(&batches)?.concat(&output_core))
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::Arc;

    use pyrefly_python::module::Module;
    use pyrefly_python::module_name::ModuleName;
    use pyrefly_python::module_path::ModulePath;
    use ruff_python_ast::Identifier;
    use ruff_python_ast::name::Name;
    use ruff_text_size::TextRange;
    use ruff_text_size::TextSize;

    use super::*;
    use crate::type_var_tuple::TypeVarTuple;
    use crate::types::Type;

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

    fn signature(spec: &str) -> GufuncSignature {
        let GufuncClassification::Supported(signature) = parse_gufunc_signature(spec) else {
            panic!("expected a supported gufunc signature: {spec}");
        };
        signature
    }

    fn shape(dimensions: &[i64]) -> IntTuple {
        IntTuple::new(dimensions.iter().copied().map(Int::Literal).collect())
    }

    fn symbolic(ty: Type) -> Int {
        Int::Symbolic(Box::new(ty))
    }

    fn gradual_middle() -> Type {
        IntTuple::shapeless().to_shape_arg_type()
    }

    fn named_variadic_middle(name: &str) -> Type {
        Type::TypeVarTuple(TypeVarTuple::new(
            Identifier::new(Name::new(name), TextRange::empty(TextSize::new(0))),
            Module::new(
                ModuleName::from_str("__test__"),
                ModulePath::filesystem(PathBuf::from("__test__")),
                Arc::new("fake module contents".to_owned()),
            ),
            None,
        ))
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

    #[test]
    fn evaluates_scalar_core_and_batched_matrix_multiplication() {
        assert_eq!(
            evaluate_gufunc(&signature("(),()->()"), &[shape(&[2, 1]), shape(&[3])])
                .expect("the batch dimensions should broadcast"),
            shape(&[2, 3])
        );
        assert_eq!(
            evaluate_gufunc(
                &signature("(m,n),(n,p)->(m,p)"),
                &[shape(&[5, 2, 3]), shape(&[1, 3, 4])],
            )
            .expect("the batch and core dimensions should be compatible"),
            shape(&[5, 2, 4])
        );
    }

    #[test]
    fn binary_scalar_core_evaluation_is_exactly_broadcast_shapes() {
        let signature = signature("(),()->()");
        let gradual = IntTuple::shapeless();
        let unpacked = IntTuple::unpacked(
            vec![Int::Literal(2)],
            gradual_middle(),
            vec![Int::Literal(3)],
        );
        let cases = [
            (gradual.clone(), shape(&[2, 3])),
            (shape(&[4, 1]), unpacked.clone()),
            (unpacked.clone(), gradual),
            (unpacked, shape(&[5, 3])),
            (shape(&[2]), shape(&[3])),
        ];
        for (left, right) in cases {
            assert_eq!(
                evaluate_gufunc(&signature, &[left.clone(), right.clone()]),
                broadcast_shapes(&left, &right)
            );
        }
    }

    #[test]
    fn validates_operand_count_and_minimum_concrete_rank() {
        assert_eq!(
            evaluate_gufunc(&signature("(),()->()"), &[shape(&[2])])
                .expect_err("one operand is missing")
                .to_string(),
            "gufunc: expected 2 operands, got 1"
        );
        assert_eq!(
            evaluate_gufunc(
                &signature("(m,n),(n,p)->(m,p)"),
                &[shape(&[3]), shape(&[3, 4])],
            )
            .expect_err("the first operand has insufficient rank")
            .to_string(),
            "gufunc: operand 0 requires at least rank 2, got 1"
        );
    }

    #[test]
    fn core_dimensions_are_exact_and_do_not_broadcast() {
        assert_eq!(
            evaluate_gufunc(&signature("(n),(n)->()"), &[shape(&[1]), shape(&[5])])
                .expect_err("core dimension one must not broadcast")
                .to_string(),
            "gufunc: core dimension 'n' has conflicting extents 1 and 5"
        );
        assert_eq!(
            evaluate_gufunc(&signature("(n,n)->(n)"), &[shape(&[3, 4])])
                .expect_err("repeated core dimensions must agree")
                .to_string(),
            "gufunc: core dimension 'n' has conflicting extents 3 and 4"
        );
    }

    #[test]
    fn reconciles_symbolic_core_dimensions_independently() {
        let n = symbolic(Type::None);
        let other_n = symbolic(Type::Ellipsis);
        let p = symbolic(Type::Materialization);

        assert_eq!(
            evaluate_gufunc(
                &signature("(m,n),(n,p)->(m,p)"),
                &[
                    IntTuple::new(vec![Int::Literal(2), n.clone()]),
                    IntTuple::new(vec![n.clone(), p.clone()]),
                ],
            )
            .expect("equal symbolic dimensions should remain precise"),
            IntTuple::new(vec![Int::Literal(2), p.clone()])
        );
        assert_eq!(
            evaluate_gufunc(
                &signature("(m,n),(n,p)->(m,n,p)"),
                &[
                    IntTuple::new(vec![Int::Literal(2), n]),
                    IntTuple::new(vec![other_n, p.clone()]),
                ],
            )
            .expect("unresolved symbolic equality should widen only its label"),
            IntTuple::new(vec![Int::Literal(2), Int::Int, p])
        );
        assert_eq!(
            evaluate_gufunc(
                &signature("(n),(n)->(n)"),
                &[IntTuple::new(vec![symbolic(Type::None)]), shape(&[7])],
            )
            .expect("a literal should constrain an unresolved symbolic dimension"),
            shape(&[7])
        );
    }

    #[test]
    fn preserves_known_core_suffix_with_gradual_or_variadic_rank() {
        let equation = signature("(m,n),(n,p)->(m,p)");
        assert_eq!(
            evaluate_gufunc(
                &equation,
                &[
                    IntTuple::unpacked(
                        vec![Int::Literal(7)],
                        gradual_middle(),
                        vec![Int::Literal(8), Int::Literal(2), Int::Literal(3)],
                    ),
                    shape(&[3, 4]),
                ],
            )
            .expect("the known variadic suffix contains the full core"),
            IntTuple::unpacked(
                vec![Int::Literal(7)],
                gradual_middle(),
                vec![Int::Literal(8), Int::Literal(2), Int::Literal(4)],
            )
        );
        assert_eq!(
            evaluate_gufunc(
                &equation,
                &[
                    IntTuple::unpacked(
                        vec![Int::Literal(2)],
                        gradual_middle(),
                        vec![Int::Literal(3)],
                    ),
                    shape(&[3, 4]),
                ],
            )
            .expect("an ambiguous core boundary should fall back conservatively"),
            IntTuple::unpacked(
                Vec::new(),
                gradual_middle(),
                vec![Int::Int, Int::Literal(4)],
            )
        );
        assert_eq!(
            evaluate_gufunc(&equation, &[IntTuple::shapeless(), shape(&[3, 4])])
                .expect("a gradual operand should preserve core dimensions known elsewhere"),
            IntTuple::unpacked(
                Vec::new(),
                gradual_middle(),
                vec![Int::Int, Int::Literal(4)],
            )
        );
        assert_eq!(
            evaluate_gufunc(
                &signature("(m,n)->(n)"),
                &[IntTuple::unpacked(
                    Vec::new(),
                    gradual_middle(),
                    vec![Int::Literal(5)],
                )],
            )
            .expect("a short fixed suffix still determines the rightmost core dimension"),
            IntTuple::unpacked(Vec::new(), gradual_middle(), vec![Int::Literal(5)])
        );
    }

    #[test]
    fn arbitrary_arity_batch_broadcasting_is_permutation_invariant() {
        let signature = signature("(),(),()->()");
        let operands = [
            IntTuple::shapeless(),
            IntTuple::unpacked(Vec::new(), gradual_middle(), vec![Int::Literal(3)]),
            shape(&[1, 3]),
        ];
        for permutation in [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ] {
            let permuted = permutation.map(|index| operands[index].clone());
            assert_eq!(
                evaluate_gufunc(&signature, &permuted)
                    .expect("variadic batch fallback should not depend on operand order"),
                IntTuple::unpacked(
                    Vec::new(),
                    gradual_middle(),
                    vec![Int::Int, Int::Literal(3)],
                )
            );
        }

        let fixed_rank = [
            IntTuple::new(vec![Int::Int, Int::Literal(1)]),
            shape(&[1, 5]),
            shape(&[7, 1]),
        ];
        for permutation in [[0, 1, 2], [2, 0, 1], [1, 2, 0]] {
            let permuted = permutation.map(|index| fixed_rank[index].clone());
            assert_eq!(
                evaluate_gufunc(&signature, &permuted)
                    .expect("fixed-rank batches should broadcast exactly"),
                shape(&[7, 5])
            );
        }

        let conflicting = [shape(&[2, 3]), IntTuple::shapeless(), shape(&[4, 3])];
        for permutation in [[0, 1, 2], [2, 0, 1], [1, 2, 0]] {
            let permuted = permutation.map(|index| conflicting[index].clone());
            assert!(
                evaluate_gufunc(&signature, &permuted).is_err(),
                "known incompatible suffixes must conflict in every operand order"
            );
        }

        let forced_suffix = [
            IntTuple::unpacked(
                Vec::new(),
                gradual_middle(),
                vec![Int::Literal(1), Int::Literal(5)],
            ),
            IntTuple::unpacked(
                Vec::new(),
                gradual_middle(),
                vec![Int::Literal(3), Int::Literal(1)],
            ),
            shape(&[4, 3, 5]),
        ];
        assert_eq!(
            evaluate_gufunc(&signature, &forced_suffix)
                .expect("known suffix dimensions should constrain gradual dimensions"),
            IntTuple::unpacked(
                Vec::new(),
                gradual_middle(),
                vec![Int::Literal(4), Int::Literal(3), Int::Literal(5)],
            )
        );

        let identical = IntTuple::unpacked(
            vec![Int::Literal(2)],
            gradual_middle(),
            vec![Int::Literal(3)],
        );
        assert_eq!(
            evaluate_gufunc(
                &signature,
                &[identical.clone(), identical.clone(), identical.clone()]
            )
            .expect("identical variadic batches should retain their full representation"),
            identical
        );
    }

    #[test]
    fn scalar_operand_does_not_hide_named_variadic_batch_ambiguity() {
        let signature = signature("(),(),()->()");
        let operands = [
            IntTuple::unpacked(Vec::new(), named_variadic_middle("Batch"), Vec::new()),
            shape(&[2]),
            shape(&[]),
        ];
        for permutation in [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ] {
            let permuted = permutation.map(|index| operands[index].clone());
            assert!(
                evaluate_gufunc(&signature, &permuted).is_err(),
                "a scalar operand must not hide ambiguous named-variadic alignment"
            );
        }
    }
}
