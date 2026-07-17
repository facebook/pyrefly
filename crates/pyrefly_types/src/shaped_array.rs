/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::cmp::Ordering;
use std::fmt;
use std::fmt::Display;
use std::hash::Hash;
use std::hash::Hasher;

use pyrefly_derive::TypeEq;
use pyrefly_derive::Visit;
use pyrefly_derive::VisitMut;
use pyrefly_util::display::commas_iter;

use crate::class::ClassType;
use crate::dimension::Int;
use crate::dimension::ShapeError;
use crate::dimension::canonicalize;
use crate::dimension::gradual_size;
use crate::dimension::is_gradual_size;
use crate::lit_int::LitInt;
use crate::literal::Lit;
use crate::quantified::QuantifiedKind;
use crate::tuple::Tuple;
use crate::types::Type;

// ============================================================================
// Shaped Array Types
// ============================================================================

/// Whether a shaped-array type was constructed using native (`Tensor[N, M]`) or
/// jaxtyping (`Float[Tensor, "N M"]`) syntax. Controls display rendering and
/// enables diagnostic checks (e.g., mixing both syntaxes in one function).
///
/// Transparent to equality, hashing, and ordering — syntax does not affect
/// type identity. Two shaped-array types with different syntax but identical base
/// class and shape are considered equal.
#[derive(Debug, Clone, Copy, Default)]
#[derive(Visit, VisitMut)]
pub enum ShapedArraySyntax {
    #[default]
    Native,
    Jaxtyping,
}

// Syntax is a display/diagnostic concern, not a type identity concern.
// All trait impls treat every ShapedArraySyntax value as equal.

impl PartialEq for ShapedArraySyntax {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

impl Eq for ShapedArraySyntax {}

impl Hash for ShapedArraySyntax {
    fn hash<H: Hasher>(&self, _state: &mut H) {}
}

impl PartialOrd for ShapedArraySyntax {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ShapedArraySyntax {
    fn cmp(&self, _other: &Self) -> Ordering {
        Ordering::Equal
    }
}

impl crate::equality::TypeEq for ShapedArraySyntax {
    fn type_eq(&self, _other: &Self, _ctx: &mut crate::equality::TypeEqCtx) -> bool {
        true
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(Visit, VisitMut, TypeEq)]
enum ShapedArrayShapeStorage {
    Inline(IntTuple),
    TupleCarrier { index: usize },
}

/// A class instance with shape information.
/// Example: Tensor[[2, 3]] represents a 2x3 tensor
/// Example: Tensor (no brackets) represents a shapeless tensor (`IntTuple`)
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(Visit, VisitMut, TypeEq)]
pub struct ShapedArrayType {
    /// Base shaped-array class (e.g., torch.Tensor)
    pub base_class: ClassType,
    shape: ShapedArrayShapeStorage,
    /// Whether this type was constructed from native or jaxtyping syntax.
    pub syntax: ShapedArraySyntax,
}

impl ShapedArrayType {
    /// Create a shaped-array type with shape information (defaults to Native syntax).
    pub fn new(base_class: ClassType, shape: IntTuple) -> Self {
        Self {
            base_class,
            shape: ShapedArrayShapeStorage::Inline(shape),
            syntax: ShapedArraySyntax::Native,
        }
    }

    /// Create a shapeless shaped-array type (compatible with any shape).
    pub fn shapeless(base_class: ClassType) -> Self {
        Self {
            base_class,
            shape: ShapedArrayShapeStorage::Inline(IntTuple::shapeless()),
            syntax: ShapedArraySyntax::Native,
        }
    }

    /// Set the syntax for this shaped-array type.
    pub fn with_syntax(mut self, syntax: ShapedArraySyntax) -> Self {
        self.syntax = syntax;
        self
    }

    pub fn with_tuple_carrier_shape_arg(mut self, index: usize) -> Self {
        self.shape = ShapedArrayShapeStorage::TupleCarrier { index };
        self
    }

    pub fn to_type(self) -> Type {
        Type::ShapedArray(Box::new(self))
    }

    pub fn tuple_carrier_shape_arg_index(&self) -> Option<usize> {
        match self.shape {
            ShapedArrayShapeStorage::Inline(_) => None,
            ShapedArrayShapeStorage::TupleCarrier { index } => Some(index),
        }
    }

    pub fn set_tuple_carrier_shape_arg(&mut self, index: usize) {
        self.shape = ShapedArrayShapeStorage::TupleCarrier { index };
    }

    pub fn shape(&self) -> IntTuple {
        match &self.shape {
            ShapedArrayShapeStorage::Inline(shape) => shape.clone(),
            ShapedArrayShapeStorage::TupleCarrier { index } => {
                let shape_arg = self
                    .base_class
                    .targs()
                    .as_slice()
                    .get(*index)
                    .expect("shape argument index should point to a class type argument");
                IntTuple::from_shape_arg_type(shape_arg)
                    .or_else(|| tuple_carrier_to_shape(shape_arg))
                    .expect("registered shaped-array shape argument should project to IntTuple")
            }
        }
    }

    pub fn set_shape(&mut self, shape: IntTuple) {
        match &mut self.shape {
            ShapedArrayShapeStorage::Inline(stored_shape) => *stored_shape = shape,
            ShapedArrayShapeStorage::TupleCarrier { index } => {
                let shape_arg = self
                    .base_class
                    .targs_mut()
                    .as_mut()
                    .get_mut(*index)
                    .expect("shape argument index should point to a class type argument");
                *shape_arg = shape.to_shape_arg_type();
            }
        }
    }

    /// Returns rank if shape is concrete, None for variadic/shapeless
    pub fn rank(&self) -> Option<usize> {
        match self.shape().view() {
            IntTupleView::Concrete(dims) => Some(dims.len()),
            IntTupleView::Gradual | IntTupleView::Unpacked { .. } => None,
        }
    }

    /// Returns true if the shaped array has no shape information.
    /// (represented as a gradual `IntTuple`)
    pub fn is_shapeless(&self) -> bool {
        is_shapeless(&self.shape())
    }
}

impl Display for ShapedArrayType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.syntax {
            ShapedArraySyntax::Native => {
                let shape = self.shape();
                if is_shapeless(&shape) {
                    write!(f, "{}", self.base_class.name())
                } else if self.tuple_carrier_shape_arg_index().is_some() {
                    write!(
                        f,
                        "{}[{}]",
                        self.base_class.name(),
                        fmt_tuple_carrier(&shape)
                    )
                } else {
                    write!(f, "{}[{}]", self.base_class.name(), shape)
                }
            }
            ShapedArraySyntax::Jaxtyping => {
                let shape = self.shape();
                write!(
                    f,
                    "Shaped[{}, \"{}\"]",
                    self.base_class.name(),
                    shape.fmt_jaxtyping()
                )
            }
        }
    }
}

/// Shape of a shaped array.
///
/// The storage is deliberately not a `Tuple`: fixed dimensions are always
/// canonical `Int`s, while variadic middles carry the original tuple/type
/// variable shape carrier.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(Visit, VisitMut, TypeEq)]
pub struct IntTuple(IntTupleRepr);

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(Visit, VisitMut, TypeEq)]
enum IntTupleRepr {
    Concrete(Vec<Int>),
    Gradual,
    Unpacked {
        prefix: Vec<Int>,
        middle: Box<Type>,
        suffix: Vec<Int>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntTupleView<'a> {
    Concrete(&'a [Int]),
    Gradual,
    Unpacked {
        prefix: &'a [Int],
        middle: &'a Type,
        suffix: &'a [Int],
    },
}

impl IntTuple {
    pub fn new(dims: Vec<Int>) -> Self {
        Self::from_ints(dims)
    }

    /// Create a concrete shape from dimension types, recovering invalid dimensions to `int`.
    pub fn from_types(dims: Vec<Type>) -> Self {
        Self::from_ints(dims.into_iter().map(type_to_dim_recover).collect())
    }

    fn from_ints(dims: Vec<Int>) -> Self {
        Self(IntTupleRepr::Concrete(
            dims.into_iter().map(canonicalize_int_dim).collect(),
        ))
    }

    /// Rebuild this shape through the canonical constructors.
    pub fn normalize(&self) -> Self {
        match &self.0 {
            IntTupleRepr::Concrete(dims) => Self::from_ints(dims.clone()),
            IntTupleRepr::Gradual => Self::shapeless(),
            IntTupleRepr::Unpacked {
                prefix,
                middle,
                suffix,
            } => Self::unpacked(prefix.clone(), middle.as_ref().clone(), suffix.clone()),
        }
    }

    fn unpacked_from_parts(prefix: Vec<Int>, middle: Type, suffix: Vec<Int>) -> Self {
        if prefix.is_empty() && suffix.is_empty() && is_gradual_shape_middle(&middle) {
            Self::shapeless()
        } else {
            let prefix: Vec<Int> = prefix.into_iter().map(canonicalize_int_dim).collect();
            let suffix: Vec<Int> = suffix.into_iter().map(canonicalize_int_dim).collect();
            Self(IntTupleRepr::Unpacked {
                prefix,
                middle: Box::new(middle),
                suffix,
            })
        }
    }

    pub fn from_tuple(tuple: Tuple) -> Self {
        match tuple {
            Tuple::Concrete(dims) => Self::from_types(dims),
            Tuple::Unpacked(unpacked) => {
                let (prefix, middle, suffix) = *unpacked;
                Self::unpacked_from_types(prefix, middle, suffix)
            }
            Tuple::Unbounded(elt) if elt.is_any() || is_gradual_size(&elt) => Self::shapeless(),
            Tuple::Unbounded(elt) => {
                Self::unpacked(Vec::new(), Type::Tuple(Tuple::Unbounded(elt)), Vec::new())
            }
        }
    }

    pub fn shapeless() -> Self {
        Self(IntTupleRepr::Gradual)
    }

    pub fn is_shapeless(&self) -> bool {
        is_shapeless(self)
    }

    pub fn view(&self) -> IntTupleView<'_> {
        match &self.0 {
            IntTupleRepr::Concrete(dims) => IntTupleView::Concrete(dims),
            IntTupleRepr::Gradual => IntTupleView::Gradual,
            IntTupleRepr::Unpacked {
                prefix,
                middle,
                suffix,
            } => IntTupleView::Unpacked {
                prefix,
                middle,
                suffix,
            },
        }
    }

    /// Project this shape to the ordinary tuple type it denotes.
    pub fn to_tuple_type(&self) -> Type {
        match &self.0 {
            IntTupleRepr::Concrete(dims) => Type::Tuple(Tuple::Concrete(dims_to_types(dims))),
            IntTupleRepr::Gradual => Type::Tuple(Tuple::Unbounded(Box::new(gradual_size()))),
            IntTupleRepr::Unpacked {
                prefix,
                middle,
                suffix,
            } => {
                let middle = match middle.as_ref() {
                    Type::IntTuple(shape) => shape.to_tuple_type(),
                    middle if is_tuple_carrier_shape_middle(middle) => {
                        Type::Tuple(Tuple::Unbounded(Box::new(gradual_size())))
                    }
                    middle => middle.clone(),
                };
                Type::Tuple(Tuple::Unpacked(Box::new((
                    dims_to_types(prefix),
                    middle,
                    dims_to_types(suffix),
                ))))
            }
        }
    }

    pub fn to_tuple(&self) -> Tuple {
        let Type::Tuple(tuple) = self.to_tuple_type() else {
            unreachable!("IntTuple always projects to a tuple")
        };
        tuple
    }

    /// Wrap this shape as the `IntTuple` type argument used to carry a whole
    /// shape (e.g. the `S` in `Tensor[S]`).
    pub fn to_shape_arg_type(&self) -> Type {
        Type::IntTuple(Box::new(self.clone()))
    }

    /// Recover a shape from an `IntTuple` shape argument, canonicalizing it via
    /// `normalize`. Returns `None` for any other type.
    pub fn from_shape_arg_type(arg: &Type) -> Option<Self> {
        match arg {
            Type::IntTuple(shape) => Some(shape.normalize()),
            _ => None,
        }
    }

    /// Create and canonicalize a variadic shape with fixed dimensions around its middle.
    pub fn unpacked(mut prefix: Vec<Int>, middle: Type, mut suffix: Vec<Int>) -> Self {
        if let Type::IntTuple(shape) = &middle {
            match shape.view() {
                IntTupleView::Concrete(dims) => {
                    prefix.extend_from_slice(dims);
                    prefix.extend(suffix);
                    return Self::from_ints(prefix);
                }
                IntTupleView::Gradual => {
                    return Self::unpacked_from_parts(prefix, gradual_shape_middle(), suffix);
                }
                IntTupleView::Unpacked {
                    prefix: inner_prefix,
                    middle: inner_middle,
                    suffix: inner_suffix,
                } => {
                    prefix.extend_from_slice(inner_prefix);
                    let mut combined_suffix = inner_suffix.to_vec();
                    combined_suffix.append(&mut suffix);
                    return Self::unpacked(prefix, inner_middle.clone(), combined_suffix);
                }
            }
        }

        if let Type::Tuple(Tuple::Concrete(dims)) = &middle {
            let dims = dims.iter().map(carrier_element_to_dim_recover);
            prefix.extend(dims);
            prefix.extend(suffix);
            return Self::from_ints(prefix);
        }

        if let Type::Tuple(Tuple::Unpacked(unpacked)) = &middle {
            let (inner_prefix, inner_middle, inner_suffix) = &**unpacked;
            prefix.extend(inner_prefix.iter().map(carrier_element_to_dim_recover));
            let mut combined_suffix: Vec<Int> = inner_suffix
                .iter()
                .map(carrier_element_to_dim_recover)
                .collect();
            combined_suffix.append(&mut suffix);
            return Self::unpacked(prefix, inner_middle.clone(), combined_suffix);
        }

        match middle {
            Type::Tuple(Tuple::Unbounded(elt)) => {
                let dim = unbounded_middle_element_to_dim(&elt);
                if matches!(dim, Int::Int) {
                    Self::unpacked_from_parts(prefix, gradual_shape_middle(), suffix)
                } else {
                    let middle = Type::Tuple(Tuple::Unbounded(Box::new(dim_to_type(&dim))));
                    Self::unpacked_from_parts(prefix, middle, suffix)
                }
            }
            Type::Any(_) => Self::unpacked_from_parts(prefix, gradual_shape_middle(), suffix),
            middle if is_unresolved_shape_middle(&middle) => {
                Self::unpacked_from_parts(prefix, middle, suffix)
            }
            _ => Self::shapeless(),
        }
    }

    /// Create an unpacked shape from a boundary that represents fixed dimensions as `Type`s.
    pub fn unpacked_from_types(prefix: Vec<Type>, middle: Type, suffix: Vec<Type>) -> Self {
        Self::unpacked(
            prefix.into_iter().map(type_to_dim_recover).collect(),
            middle,
            suffix.into_iter().map(type_to_dim_recover).collect(),
        )
    }

    pub fn rank(&self) -> usize {
        match &self.0 {
            IntTupleRepr::Concrete(dims) => dims.len(),
            IntTupleRepr::Gradual | IntTupleRepr::Unpacked { .. } => {
                // For unpacked shapes, rank is unknown at parse time
                // This should not be called for variadic shapes
                panic!("Cannot determine rank of variadic tensor shape")
            }
        }
    }

    pub fn is_empty(&self) -> bool {
        match &self.0 {
            IntTupleRepr::Concrete(dims) => dims.is_empty(),
            IntTupleRepr::Gradual | IntTupleRepr::Unpacked { .. } => false,
        }
    }

    /// Get a slice of dimensions (only valid for concrete shapes)
    pub fn dims_slice(&self) -> &[Int] {
        match &self.0 {
            IntTupleRepr::Concrete(dims) => dims,
            IntTupleRepr::Gradual | IntTupleRepr::Unpacked { .. } => {
                panic!("Cannot get dims_slice for variadic tensor shape")
            }
        }
    }

    /// Get the concrete dims if this is a concrete shape.
    pub fn as_concrete(&self) -> Option<&[Int]> {
        match &self.0 {
            IntTupleRepr::Concrete(dims) => Some(dims),
            IntTupleRepr::Gradual | IntTupleRepr::Unpacked { .. } => None,
        }
    }

    /// Get a mutable reference to concrete dims (for meta-shape operations)
    /// Panics if called on Unpacked shape
    pub fn dims_mut(&mut self) -> &mut Vec<Int> {
        match &mut self.0 {
            IntTupleRepr::Concrete(dims) => dims,
            IntTupleRepr::Gradual | IntTupleRepr::Unpacked { .. } => {
                panic!("Cannot get mutable dims for variadic tensor shape")
            }
        }
    }

    /// Get dims as a Vec for concrete shapes, panics for unpacked
    /// This is used by meta-shape code that doesn't support variadic shapes yet
    pub fn dims(&self) -> &Vec<Int> {
        match &self.0 {
            IntTupleRepr::Concrete(dims) => dims,
            IntTupleRepr::Gradual | IntTupleRepr::Unpacked { .. } => {
                panic!("Meta-shape operations do not yet support variadic tensor shapes")
            }
        }
    }

    /// Check if all dimensions are literal (concrete integers)
    /// Returns false for variadic shapes
    pub fn all_literal(&self) -> bool {
        match &self.0 {
            IntTupleRepr::Concrete(dims) => dims.iter().all(|dim| matches!(dim, Int::Literal(_))),
            IntTupleRepr::Gradual | IntTupleRepr::Unpacked { .. } => false,
        }
    }

    /// Extract literal dimension values if all are literal
    /// Returns None for variadic shapes
    pub fn as_literals(&self) -> Option<Vec<i64>> {
        match &self.0 {
            IntTupleRepr::Concrete(dims) if self.all_literal() => Some(
                dims.iter()
                    .map(|dim| match dim {
                        Int::Literal(n) => *n,
                        _ => unreachable!("all_literal checked every concrete dimension"),
                    })
                    .collect(),
            ),
            _ => None,
        }
    }

    /// Get a dimension by index (only for concrete shapes)
    pub fn get_dim(&self, index: usize) -> Type {
        match &self.0 {
            IntTupleRepr::Concrete(dims) => dim_to_type(dims.get(index).unwrap()),
            IntTupleRepr::Gradual | IntTupleRepr::Unpacked { .. } => {
                panic!("Cannot get dimension by index for variadic tensor shape")
            }
        }
    }

    /// Normalize a dimension index to handle negative indices
    ///
    /// Negative indices count from the end: -1 is the last dimension, -2 is second-to-last, etc.
    /// Returns an error if the index is out of range.
    pub fn normalize_dim(&self, dim: i64) -> Result<usize, ShapeError> {
        // Check for variadic shape first - cannot normalize dims for unpacked shapes
        if !matches!(self.0, IntTupleRepr::Concrete(_)) {
            return Err(ShapeError::InvalidDimension {
                value: dim,
                reason: "Cannot normalize dimension index for variadic tensor shape".to_owned(),
            });
        }

        let rank = self.rank() as i64;

        if rank == 0 {
            return Err(ShapeError::InvalidDimension {
                value: dim,
                reason: "Cannot normalize dimension for scalar tensor (rank 0)".to_owned(),
            });
        }

        let normalized = if dim < 0 { rank + dim } else { dim };

        if normalized < 0 || normalized >= rank {
            return Err(ShapeError::InvalidDimension {
                value: dim,
                reason: format!(
                    "Dimension {} out of range for tensor with rank {} (valid range: {} to {})",
                    dim,
                    rank,
                    -rank,
                    rank - 1
                ),
            });
        }

        Ok(normalized as usize)
    }

    /// Format the shape using jaxtyping syntax (space-separated, no parens for scalar).
    ///
    /// Handles all jaxtyping dimension types:
    /// - `Type::Any` → `_` (anonymous dim)
    /// - `Type::Int(Literal(n))` → `n`
    /// - `Type::Int(Add/Sub)` → `a+b` / `a-b` (no parens, no spaces)
    /// - `Type::Quantified` → dim name
    /// - Unpacked with gradual `IntTuple` middle → `...` (ellipsis)
    /// - Unpacked with TypeVarTuple middle → `*name`
    pub fn fmt_jaxtyping(&self) -> String {
        match &self.0 {
            IntTupleRepr::Concrete(dims) => {
                if dims.is_empty() {
                    String::new() // Scalar: empty string inside quotes
                } else {
                    dims.iter()
                        .map(fmt_jaxtyping_int)
                        .collect::<Vec<_>>()
                        .join(" ")
                }
            }
            IntTupleRepr::Gradual => "...".to_owned(),
            IntTupleRepr::Unpacked {
                prefix,
                middle,
                suffix,
            } => {
                let mut parts: Vec<String> = prefix.iter().map(fmt_jaxtyping_int).collect();

                // Ellipsis: a gradual `IntTuple` middle renders as "..."
                // Named TypeVarTuple renders as "*name"
                if is_gradual_shape_middle(middle) {
                    parts.push("...".to_owned());
                } else {
                    parts.push(format!("*{middle}"));
                }

                parts.extend(suffix.iter().map(fmt_jaxtyping_int));
                parts.join(" ")
            }
        }
    }
}

/// Format an `Int` in jaxtyping syntax (no parens, no spaces around operators).
fn fmt_jaxtyping_int(expr: &Int) -> String {
    match expr {
        Int::Literal(n) => n.to_string(),
        Int::Int => "_".to_owned(),
        Int::Symbolic(ty) => format!("{ty}"),
        Int::Add(left, right) => {
            // After canonicalization, Sub(a,b) becomes Add(Literal(-b), a).
            // Detect this and render as subtraction: Add(-n, x) → x-n
            if let Int::Literal(n) = left.as_ref()
                && *n < 0
            {
                return format!("{}-{}", fmt_jaxtyping_int(right), n.wrapping_neg());
            }
            format!("{}+{}", fmt_jaxtyping_int(left), fmt_jaxtyping_int(right))
        }
        Int::Sub(left, right) => {
            format!("{}-{}", fmt_jaxtyping_int(left), fmt_jaxtyping_int(right))
        }
        // Mul/FloorDiv fall back to default `Int` display (rare in jaxtyping)
        _ => format!("{expr}"),
    }
}

impl Display for IntTuple {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.0 {
            IntTupleRepr::Concrete(dims) => {
                if dims.is_empty() {
                    write!(f, "()") // Scalar tensor: Tensor[()]
                } else {
                    write!(f, "{}", commas_iter(|| dims.iter()))
                }
            }
            IntTupleRepr::Gradual => write!(f, "*IntTuple"),
            IntTupleRepr::Unpacked {
                prefix,
                middle,
                suffix,
            } => {
                let prefix_str = if prefix.is_empty() {
                    "".to_owned()
                } else {
                    format!("{}, ", commas_iter(|| prefix.iter()))
                };
                let suffix_str = if suffix.is_empty() {
                    "".to_owned()
                } else {
                    format!(", {}", commas_iter(|| suffix.iter()))
                };
                write!(
                    f,
                    "{}*{}{}",
                    prefix_str,
                    fmt_unpacked_middle(middle),
                    suffix_str
                )
            }
        }
    }
}

fn fmt_unpacked_middle(middle: &Type) -> String {
    match middle {
        Type::IntTuple(shape) if shape.is_shapeless() => "tuple[int, ...]".to_owned(),
        Type::Tuple(Tuple::Unbounded(elt)) if elt.is_any() => "tuple[int, ...]".to_owned(),
        Type::Tuple(Tuple::Unbounded(elt)) if is_gradual_size(elt) => "tuple[int, ...]".to_owned(),
        middle if is_tuple_carrier_shape_middle(middle) => format!("Elements[{middle}]"),
        _ => format!("{middle}"),
    }
}

fn fmt_tuple_carrier(shape: &IntTuple) -> String {
    match shape.view() {
        IntTupleView::Concrete(dims) => {
            format!("[{}]", commas_iter(|| dims.iter()))
        }
        IntTupleView::Gradual => {
            // No unbounded shape reaches here: the only caller (`Display`) handles
            // the shapeless `IntTuple` case before calling this.
            unreachable!("shapeless IntTuple is handled before fmt_tuple_carrier")
        }
        IntTupleView::Unpacked {
            prefix,
            middle,
            suffix,
        } => {
            if prefix.is_empty() && suffix.is_empty() && is_tuple_carrier_shape_middle(middle) {
                return middle.to_string();
            }
            let mut parts: Vec<String> = prefix.iter().map(|d| d.to_string()).collect();
            parts.push(format!("*{}", fmt_unpacked_middle(middle)));
            parts.extend(suffix.iter().map(|d| d.to_string()));
            format!("[{}]", parts.join(", "))
        }
    }
}

// ============================================================================
// Tuple-carrier conversion
// ============================================================================
//
// A "tuple carrier" is the user-facing spelling of a shape that NumPy-style
// syntax such as `ndarray[[3, 4, 5], DType]` or
// `ndarray[tuple[Literal[3], Literal[4], Literal[5]], DType]` produces, where
// each dimension is written as `Literal[n]` or `Int[x]`. Internally we store
// scalar dimensions as `Type::Int`, while variadic middles keep their carrier
// type. These helpers canonicalize between the two representations so the rest
// of the type checker only ever deals with the internal form.

fn canonicalize_int_dim(dim: Int) -> Int {
    match canonicalize(Type::Int(dim)) {
        Type::Int(dim) => dim,
        _ => unreachable!("canonicalizing a Int dimension should produce a Int"),
    }
}

fn type_to_dim_recover(dim: Type) -> Int {
    type_to_dim(&dim).unwrap_or(Int::Int)
}

fn dim_to_type(dim: &Int) -> Type {
    Type::Int(dim.clone())
}

fn dims_to_types(dims: &[Int]) -> Vec<Type> {
    dims.iter().map(dim_to_type).collect()
}

/// Convert an internal shape dimension into its tuple-carrier element.
/// Literal dimensions become `Literal[n]`; other dimensions remain in their
/// canonical internal representation.
fn dim_to_carrier_element(dim: &Int) -> Type {
    match dim {
        Int::Literal(n) => LitInt::new(*n).to_explicit_type(),
        _ => dim_to_type(dim),
    }
}

fn is_valid_internal_dim(dim: &Type) -> bool {
    match dim {
        Type::Int(expr) => is_valid_internal_int(expr),
        Type::Quantified(q) => q.kind == QuantifiedKind::IntVar,
        Type::TypeVar(tv) => tv.kind() == QuantifiedKind::IntVar,
        Type::Var(_) | Type::Any(_) => true,
        _ => false,
    }
}

fn is_valid_internal_int(expr: &Int) -> bool {
    match expr {
        Int::Literal(_) | Int::Int => true,
        Int::Symbolic(ty) => is_valid_internal_dim(ty),
        Int::Add(left, right)
        | Int::Sub(left, right)
        | Int::Mul(left, right)
        | Int::FloorDiv(left, right)
        | Int::Pow(left, right) => is_valid_internal_int(left) && is_valid_internal_int(right),
    }
}

/// Convert a `Type` into an internal shape dimension, accepting only types that
/// form a valid `Int` dimension. Returns `None` for anything else, so callers
/// fail cleanly instead of treating an unrelated type as a dimension.
pub fn type_to_dim(dim: &Type) -> Option<Int> {
    Int::from_type(dim).filter(is_valid_internal_int)
}

/// Convert a single tuple-carrier element into an internal shape dimension.
///
/// Returns `None` for elements that are not valid dimensions (e.g. non-int
/// literals or arbitrary class types) so that conversion fails cleanly instead
/// of silently treating an unrelated type as a dimension.
fn carrier_element_to_dim(carrier: &Type) -> Option<Int> {
    match carrier {
        // `Literal[n]` (int) -> internal literal dimension.
        Type::Literal(lit) => match &lit.value {
            Lit::Int(i) => i.as_i64().map(Int::Literal),
            _ => None,
        },
        // Dimensions already in internal form pass through unchanged.
        Type::Int(expr) if is_valid_internal_int(expr) => Some(canonicalize_int_dim(expr.clone())),
        Type::Quantified(q) if q.kind == QuantifiedKind::IntVar => {
            Some(type_to_dim_recover(carrier.clone()))
        }
        Type::TypeVar(tv) if tv.kind() == QuantifiedKind::IntVar => {
            Some(type_to_dim_recover(carrier.clone()))
        }
        Type::Var(_) => Some(Int::Int),
        Type::Any(_) => Some(Int::Int),
        Type::ClassType(cls) if cls.is_builtin("int") => Some(Int::Int),
        _ => None,
    }
}

fn carrier_element_to_dim_recover(carrier: &Type) -> Int {
    carrier_element_to_dim(carrier).unwrap_or(Int::Int)
}

/// Convert an `IntTuple` into the equivalent tuple-carrier `Type`.
pub fn shape_to_tuple_carrier(shape: &IntTuple) -> Type {
    match shape.view() {
        IntTupleView::Concrete(dims) => Type::Tuple(Tuple::Concrete(
            dims.iter().map(dim_to_carrier_element).collect(),
        )),
        IntTupleView::Gradual => Type::any_tuple(),
        IntTupleView::Unpacked {
            prefix,
            middle,
            suffix,
        } => {
            let middle = match middle {
                Type::IntTuple(shape) => shape_to_tuple_carrier(shape),
                _ => middle.clone(),
            };
            Type::Tuple(Tuple::Unpacked(Box::new((
                prefix.iter().map(dim_to_carrier_element).collect(),
                middle,
                suffix.iter().map(dim_to_carrier_element).collect(),
            ))))
        }
    }
}

/// Detects a tuple-carrier shape variable occupying the variadic middle of an
/// unpacked shape.
pub fn is_tuple_carrier_shape_middle(ty: &Type) -> bool {
    // An ordinary TypeVar is legal here only as a whole-shape carrier from
    // tuple-carrier syntax, e.g. `Array[S, DType]` -> `Unpacked([], S, [])`.
    // Scalar symbolic dimensions use `Int`/`IntVar` and must not reach
    // this fallback as bare TypeVars.
    matches!(ty, Type::Var(_))
        || matches!(ty, Type::TypeVar(tv) if tv.kind() == QuantifiedKind::TypeVar)
        || matches!(ty, Type::Quantified(q) if q.kind == QuantifiedKind::TypeVar)
}

fn is_unresolved_shape_middle(ty: &Type) -> bool {
    // Tuple-carrier shape variables are scalar TypeVars syntactically, but in an
    // unpacked shape middle they stand for an unresolved shape tuple.
    matches!(ty, Type::Var(_) | Type::TypeVarTuple(_))
        || matches!(ty, Type::Quantified(q) if q.kind == QuantifiedKind::TypeVarTuple)
        || is_tuple_carrier_shape_middle(ty)
}

fn unbounded_middle_element_to_dim(elt: &Type) -> Int {
    if elt.is_any() || matches!(elt, Type::ClassType(cls) if cls.is_builtin("int")) {
        Int::Int
    } else {
        carrier_element_to_dim(elt).unwrap_or(Int::Int)
    }
}

/// Convert a projected tuple-carrier shape back to the class type argument.
///
/// A tuple-carrier `TypeVar` represents the whole shape tuple, so `ndarray[S,
/// DType]` projects to `Unpacked([], S, [])` but must round-trip back to `S`,
/// not `tuple[*S]`.
pub fn shape_to_tuple_carrier_arg(shape: &IntTuple) -> Type {
    match shape.view() {
        IntTupleView::Unpacked {
            prefix,
            middle,
            suffix,
        } => {
            if prefix.is_empty() && suffix.is_empty() && is_tuple_carrier_shape_middle(middle) {
                middle.clone()
            } else {
                shape_to_tuple_carrier(shape)
            }
        }
        _ => shape_to_tuple_carrier(shape),
    }
}

/// Convert a tuple-carrier `Type` into an `IntTuple`.
///
/// Returns `None` when the carrier is not a tuple or contains an element that is
/// not a valid dimension.
///
/// `tuple[T, ...]` (including `tuple[int, ...]` and `tuple[Any, ...]`)
/// intentionally canonicalizes to the shapeless / unknown-rank shape: an
/// unbounded carrier conveys no recoverable per-dimension information.
pub fn tuple_carrier_to_shape(carrier: &Type) -> Option<IntTuple> {
    match carrier {
        Type::Tuple(Tuple::Concrete(elts)) => {
            let dims = elts
                .iter()
                .map(carrier_element_to_dim)
                .collect::<Option<Vec<_>>>()?;
            Some(IntTuple::from_ints(dims))
        }
        Type::Tuple(Tuple::Unpacked(unpacked)) => {
            let (prefix, middle, suffix) = &**unpacked;
            let prefix = prefix
                .iter()
                .map(carrier_element_to_dim)
                .collect::<Option<Vec<_>>>()?;
            let suffix = suffix
                .iter()
                .map(carrier_element_to_dim)
                .collect::<Option<Vec<_>>>()?;
            if matches!(middle, Type::Tuple(Tuple::Unbounded(_))) {
                return Some(IntTuple::unpacked(prefix, gradual_shape_middle(), suffix));
            }
            validate_tuple_carrier_unpacked_middle(middle)?;
            let middle = recover_unbounded_tuple_carrier_middle(middle.clone());
            Some(IntTuple::unpacked(prefix, middle, suffix))
        }
        Type::Tuple(Tuple::Unbounded(_)) => Some(shapeless_shape()),
        _ if is_tuple_carrier_shape_middle(carrier) => {
            Some(IntTuple::unpacked(Vec::new(), carrier.clone(), Vec::new()))
        }
        _ => None,
    }
}

fn recover_unbounded_tuple_carrier_middle(middle: Type) -> Type {
    match middle {
        Type::Tuple(Tuple::Unpacked(unpacked)) => {
            let (prefix, middle, suffix) = *unpacked;
            Type::Tuple(Tuple::Unpacked(Box::new((
                prefix,
                recover_unbounded_tuple_carrier_middle(middle),
                suffix,
            ))))
        }
        Type::Tuple(Tuple::Unbounded(_)) => gradual_shape_middle(),
        middle => middle,
    }
}

fn validate_tuple_carrier_unpacked_middle(middle: &Type) -> Option<()> {
    match middle {
        Type::Tuple(Tuple::Concrete(elts)) => {
            elts.iter()
                .map(carrier_element_to_dim)
                .collect::<Option<Vec<_>>>()?;
            Some(())
        }
        Type::Tuple(Tuple::Unpacked(unpacked)) => {
            let (prefix, middle, suffix) = &**unpacked;
            prefix
                .iter()
                .map(carrier_element_to_dim)
                .collect::<Option<Vec<_>>>()?;
            suffix
                .iter()
                .map(carrier_element_to_dim)
                .collect::<Option<Vec<_>>>()?;
            validate_tuple_carrier_unpacked_middle(middle)
        }
        Type::Tuple(Tuple::Unbounded(_)) => Some(()),
        middle if is_unresolved_shape_middle(middle) => Some(()),
        Type::IntTuple(_) => Some(()),
        _ => None,
    }
}

fn gradual_shape_middle() -> Type {
    IntTuple::shapeless().to_shape_arg_type()
}

fn is_gradual_shape_middle(middle: &Type) -> bool {
    match middle {
        Type::IntTuple(shape) => shape.is_shapeless(),
        Type::Tuple(Tuple::Unbounded(elt)) => elt.is_any() || is_gradual_size(elt),
        _ => false,
    }
}

/// Check if a shape is shapeless: gradual `IntTuple`.
fn is_shapeless(shape: &IntTuple) -> bool {
    matches!(shape.view(), IntTupleView::Gradual)
}

/// Compute the broadcasted shape of two tensor shapes following NumPy/PyTorch broadcasting rules:
/// - Dimensions are aligned from right to left
/// - Each dimension must either match or one of them must be 1
/// - Missing dimensions are treated as 1
///
/// For shapes with variadic middles (Unpacked), the algorithm:
/// 1. Consume concrete suffix dims from both sides, right-to-left, broadcasting each pair.
///    Stop when either side runs out of concrete dims (hits a middle or exhausts its dims).
/// 2. Analyze what remains after suffix consumption:
///    - empty + anything → result is the other side
///    - concrete + unpacked(p, m, []) → shapeless if m is gradual; error if m is TypeVarTuple
///    - unpacked + unpacked → if same TypeVarTuple with no extra suffix, broadcast prefixes;
///      if either is gradual, shapeless; otherwise error
/// 3. Assemble result from step 2 output + broadcast suffix.
pub fn broadcast_shapes(a: &IntTuple, b: &IntTuple) -> Result<IntTuple, ShapeError> {
    match (a.view(), b.view()) {
        (IntTupleView::Concrete(a_dims), IntTupleView::Concrete(b_dims)) => {
            broadcast_concrete(a_dims, b_dims)
        }
        (IntTupleView::Concrete(concrete), IntTupleView::Gradual)
        | (IntTupleView::Gradual, IntTupleView::Concrete(concrete)) => {
            broadcast_concrete_with_unpacked(concrete, &[], &gradual_shape_middle(), &[])
        }
        (
            IntTupleView::Concrete(concrete),
            IntTupleView::Unpacked {
                prefix,
                middle,
                suffix,
            },
        )
        | (
            IntTupleView::Unpacked {
                prefix,
                middle,
                suffix,
            },
            IntTupleView::Concrete(concrete),
        ) => broadcast_concrete_with_unpacked(concrete, prefix, middle, suffix),
        (IntTupleView::Gradual, IntTupleView::Gradual) => Ok(IntTuple::shapeless()),
        (
            IntTupleView::Unpacked {
                prefix,
                middle,
                suffix,
            },
            IntTupleView::Gradual,
        ) => broadcast_unpacked_with_unpacked(
            prefix,
            middle,
            suffix,
            &[],
            &gradual_shape_middle(),
            &[],
        ),
        (
            IntTupleView::Gradual,
            IntTupleView::Unpacked {
                prefix,
                middle,
                suffix,
            },
        ) => broadcast_unpacked_with_unpacked(
            &[],
            &gradual_shape_middle(),
            &[],
            prefix,
            middle,
            suffix,
        ),
        (
            IntTupleView::Unpacked {
                prefix: ap,
                middle: am,
                suffix: a_suf,
            },
            IntTupleView::Unpacked {
                prefix: bp,
                middle: bm,
                suffix: b_suf,
            },
        ) => broadcast_unpacked_with_unpacked(ap, am, a_suf, bp, bm, b_suf),
    }
}

/// Broadcast a Concrete shape with an Unpacked shape.
///
/// Right-aligns concrete dims against the Unpacked's suffix, broadcasting pairwise.
/// After suffix consumption:
/// - If no concrete dims remain: preserve the Unpacked's prefix + middle.
/// - If concrete dims remain and middle is gradual: result middle is gradual `IntTuple`.
/// - If concrete dims remain and middle is TypeVarTuple: error.
fn broadcast_concrete_with_unpacked(
    concrete: &[Int],
    prefix: &[Int],
    middle: &Type,
    suffix: &[Int],
) -> Result<IntTuple, ShapeError> {
    let matched = concrete.len().min(suffix.len());

    // Build result suffix: unmatched suffix dims on the left pass through,
    // then broadcast the matched pairs (right-aligned).
    let mut result_suffix = suffix[..suffix.len() - matched].to_vec();
    for i in 0..matched {
        let c_idx = concrete.len() - matched + i;
        let s_idx = suffix.len() - matched + i;
        result_suffix.push(broadcast_dim(&concrete[c_idx], &suffix[s_idx], s_idx)?);
    }

    // Remaining concrete dims not consumed by suffix matching
    let remaining = &concrete[..concrete.len() - matched];

    if remaining.is_empty() {
        // All concrete dims consumed → preserve prefix + middle
        Ok(IntTuple::unpacked_from_parts(
            prefix.to_vec(),
            middle.clone(),
            result_suffix,
        ))
    } else if is_gradual_shape_middle(middle) {
        // Can't align remaining concrete with gradual shapeless middle.
        Ok(IntTuple::unpacked_from_parts(
            vec![],
            gradual_shape_middle(),
            result_suffix,
        ))
    } else {
        Err(ShapeError::ShapeComputation {
            message: "Cannot broadcast concrete dims with variadic shape: alignment is ambiguous"
                .to_owned(),
        })
    }
}

/// Broadcast two Unpacked shapes.
///
/// Right-aligns suffixes, broadcasting matched pairs. Then analyzes the middles:
/// - Same TypeVarTuple with no extra suffix dims: cancel middles, broadcast prefixes.
/// - Either middle is gradual: result is shapeless + broadcast suffix.
/// - Otherwise: error.
fn broadcast_unpacked_with_unpacked(
    ap: &[Int],
    am: &Type,
    a_suf: &[Int],
    bp: &[Int],
    bm: &Type,
    b_suf: &[Int],
) -> Result<IntTuple, ShapeError> {
    let matched = a_suf.len().min(b_suf.len());

    // Broadcast matched suffix pairs (right-aligned)
    let mut result_suffix = Vec::new();
    for i in 0..matched {
        let a_idx = a_suf.len() - matched + i;
        let b_idx = b_suf.len() - matched + i;
        result_suffix.push(broadcast_dim(&a_suf[a_idx], &b_suf[b_idx], 0)?);
    }

    // Unmatched suffix dims (at most one side has them)
    let a_extra = &a_suf[..a_suf.len() - matched];
    let b_extra = &b_suf[..b_suf.len() - matched];
    let has_extra = !a_extra.is_empty() || !b_extra.is_empty();

    let am_canon = canonicalize(am.clone());
    let bm_canon = canonicalize(bm.clone());

    if !has_extra && am_canon == bm_canon && !is_gradual_shape_middle(am) {
        // Same TypeVarTuple, no extra suffix → cancel middles, broadcast prefixes
        let prefix = broadcast_concrete(ap, bp)?
            .as_concrete()
            .expect("broadcast_concrete returns a concrete shape")
            .to_vec();
        Ok(IntTuple::unpacked_from_parts(
            prefix,
            am.clone(),
            result_suffix,
        ))
    } else if is_gradual_shape_middle(am) || is_gradual_shape_middle(bm) {
        // At least one gradual shapeless middle → can't determine alignment.
        Ok(IntTuple::unpacked_from_parts(
            vec![],
            gradual_shape_middle(),
            result_suffix,
        ))
    } else {
        // Different TypeVarTuples or structural mismatch — degrade to shapeless
        // batch dims rather than producing a hard error. At runtime the middles
        // are often identical (e.g. two Linear.forward calls on the same batch)
        // but the checker can't prove it.
        Ok(IntTuple::unpacked_from_parts(
            vec![],
            gradual_shape_middle(),
            result_suffix,
        ))
    }
}

/// Broadcast two concrete dimension lists following NumPy/PyTorch rules.
/// Returns a Concrete IntTuple.
fn broadcast_concrete(a_dims: &[Int], b_dims: &[Int]) -> Result<IntTuple, ShapeError> {
    let max_rank = a_dims.len().max(b_dims.len());
    let mut result_dims = Vec::with_capacity(max_rank);

    // Iterate from right to left
    for i in 0..max_rank {
        let a_idx = a_dims.len().wrapping_sub(i + 1);
        let b_idx = b_dims.len().wrapping_sub(i + 1);

        let a_dim = if a_idx < a_dims.len() {
            Some(&a_dims[a_idx])
        } else {
            None // Treat as implicit 1
        };

        let b_dim = if b_idx < b_dims.len() {
            Some(&b_dims[b_idx])
        } else {
            None // Treat as implicit 1
        };

        let result_dim = match (a_dim, b_dim) {
            (Some(a_ty), Some(b_ty)) => broadcast_dim(a_ty, b_ty, max_rank - i - 1)?,
            // One shape ran out of dimensions, use the other
            (Some(dim), None) | (None, Some(dim)) => dim.clone(),
            (None, None) => unreachable!(),
        };

        result_dims.push(result_dim);
    }

    // Reverse to get left-to-right order
    result_dims.reverse();
    Ok(IntTuple::from_ints(result_dims))
}

/// Broadcast a single pair of dimensions.
/// Canonicalizes both sides so symbolic expressions that reduce to literals are caught.
fn broadcast_dim(a_ty: &Int, b_ty: &Int, position: usize) -> Result<Int, ShapeError> {
    let a_ty = canonicalize_int_dim(a_ty.clone());
    let b_ty = canonicalize_int_dim(b_ty.clone());
    match (&a_ty, &b_ty) {
        // Equal dimensions (after canonicalization): compatible
        _ if a_ty == b_ty => Ok(a_ty.clone()),
        // Broadcasting with one preserves a gradual runtime dimension.
        (Int::Literal(1), _) => Ok(b_ty.clone()),
        (_, Int::Literal(1)) => Ok(a_ty.clone()),
        // Gradual Int is compatible with anything; prefer the more precise side.
        (Int::Int, _) => Ok(b_ty.clone()),
        (_, Int::Int) => Ok(a_ty.clone()),
        // Different non-broadcastable types: incompatible
        _ => Err(ShapeError::ShapeComputation {
            message: format!(
                "Cannot broadcast dimension {} with dimension {} at position {}",
                dim_to_type(&a_ty),
                dim_to_type(&b_ty),
                position
            ),
        }),
    }
}

// ============================================================================
// Shaped Array Indexing / Slicing
// ============================================================================

/// A single index operation, pre-classified by the type checker.
/// The type checker resolves Expr nodes into these before calling shape functions.
pub enum IndexOp {
    /// Integer index: removes the dimension
    Int,
    /// Slice: replaces dimension with (stop - start) / step.
    /// `start` defaults to 0, `stop` defaults to the dimension size.
    /// `step` defaults to 1 (no stride).
    Slice {
        start: Option<Int>,
        stop: Option<Int>,
        /// Step/stride for the slice. `None` means step=1 (default).
        step: Option<Int>,
    },
    /// Shaped-array advanced operand; all advanced shapes broadcast globally and emit once.
    ShapedArrayIndex(Vec<Int>),
    /// Tuple/list advanced operand; all advanced shapes broadcast globally and emit once.
    Fancy(Int),
    /// None/np.newaxis index: inserts a new dimension of size 1.
    /// Does not consume a shape dimension.
    NewAxis,
}

/// Apply a single integer index — removes first dimension.
/// E.g. `Tensor[10, 20][i]` -> `Tensor[20]`
pub fn index_shape_int(shape: &IntTuple) -> Result<IntTuple, ShapeError> {
    match shape.view() {
        IntTupleView::Concrete(dims) => {
            if dims.is_empty() {
                return Err(ShapeError::ScalarIndex);
            }
            Ok(IntTuple::from_ints(dims[1..].to_vec()))
        }
        IntTupleView::Unpacked {
            prefix,
            middle,
            suffix,
        } if !prefix.is_empty() => Ok(IntTuple::unpacked_from_parts(
            prefix[1..].to_vec(),
            middle.clone(),
            suffix.to_vec(),
        )),
        // First dim is in variadic middle; can't determine result
        IntTupleView::Gradual | IntTupleView::Unpacked { .. } => Ok(shapeless_shape()),
    }
}

/// Apply a single slice to first dimension.
/// E.g. `Tensor[10, 20][2:5]` -> `Tensor[3, 20]`
/// With step: `Tensor[100][::2]` -> `Tensor[50]` (ceil_div(100, 2))
pub fn index_shape_slice(
    shape: &IntTuple,
    start: Option<Int>,
    stop: Option<Int>,
    step: Option<Int>,
) -> Result<IntTuple, ShapeError> {
    match shape.view() {
        IntTupleView::Concrete(dims) => {
            if dims.is_empty() {
                return Err(ShapeError::ScalarIndex);
            }
            let start = adjust_negative(start.unwrap_or(Int::Literal(0)), &dims[0]);
            let stop = adjust_negative(stop.unwrap_or_else(|| dims[0].clone()), &dims[0]);
            let range_dim = sub_dim(stop, start);
            let new_first_dim = apply_step(range_dim, step);
            let mut new_dims = vec![new_first_dim];
            new_dims.extend_from_slice(&dims[1..]);
            Ok(IntTuple::new(new_dims))
        }
        IntTupleView::Unpacked {
            prefix,
            middle,
            suffix,
        } if !prefix.is_empty() => {
            let start = adjust_negative(start.unwrap_or(Int::Literal(0)), &prefix[0]);
            let stop = adjust_negative(stop.unwrap_or_else(|| prefix[0].clone()), &prefix[0]);
            let range_dim = sub_dim(stop, start);
            let new_first_dim = apply_step(range_dim, step);
            let mut new_prefix = vec![new_first_dim];
            new_prefix.extend_from_slice(&prefix[1..]);
            Ok(IntTuple::unpacked_from_parts(
                new_prefix,
                middle.clone(),
                suffix.to_vec(),
            ))
        }
        // Empty prefix: dim0 is hidden in the variadic middle
        IntTupleView::Gradual | IntTupleView::Unpacked { .. } => Ok(shapeless_shape()),
    }
}

/// Apply tensor-as-index — replaces first dim with index tensor's dims.
/// E.g. `Tensor[B, D1, D2][Tensor[T]]` -> `Tensor[T, D1, D2]`
pub fn index_shape_tensor(shape: &IntTuple, idx_dims: &[Int]) -> Result<IntTuple, ShapeError> {
    match shape.view() {
        IntTupleView::Concrete(dims) => {
            if dims.is_empty() {
                return Err(ShapeError::ScalarIndex);
            }
            let mut new_dims = idx_dims.to_vec();
            new_dims.extend_from_slice(&dims[1..]);
            Ok(IntTuple::new(new_dims))
        }
        IntTupleView::Unpacked {
            prefix,
            middle,
            suffix,
        } if !prefix.is_empty() => {
            let mut new_prefix = idx_dims.to_vec();
            new_prefix.extend_from_slice(&prefix[1..]);
            Ok(IntTuple::unpacked_from_parts(
                new_prefix,
                middle.clone(),
                suffix.to_vec(),
            ))
        }
        // First dim is in variadic middle; can't determine result
        IntTupleView::Gradual | IntTupleView::Unpacked { .. } => Ok(shapeless_shape()),
    }
}

/// Count how many shape dimensions a sequence of ops consumes.
/// `NewAxis` ops don't consume a dimension; all others consume one.
fn ops_dims_consumed(ops: &[IndexOp]) -> usize {
    ops.iter()
        .filter(|op| !matches!(op, IndexOp::NewAxis))
        .count()
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum IndexOpGroup {
    Pre,
    Post,
}

enum AdvancedIndexEmission {
    None,
    Front,
    At {
        group: IndexOpGroup,
        op_index: usize,
    },
}

struct AdvancedIndexPlan {
    broadcast_shape: Option<IntTuple>,
    emission: AdvancedIndexEmission,
}

impl AdvancedIndexPlan {
    fn build(
        pre_ops: &[IndexOp],
        post_ops: &[IndexOp],
        has_ellipsis: bool,
    ) -> Result<Self, ShapeError> {
        let mut broadcast_shape = None;
        let mut first_advanced = None;
        let mut separator_since_advanced = false;
        let mut separated = false;

        let mut entered_post = false;
        for (group, op_index, op) in pre_ops
            .iter()
            .enumerate()
            .map(|(op_index, op)| (IndexOpGroup::Pre, op_index, op))
            .chain(
                post_ops
                    .iter()
                    .enumerate()
                    .map(|(op_index, op)| (IndexOpGroup::Post, op_index, op)),
            )
        {
            if group == IndexOpGroup::Post && !entered_post {
                entered_post = true;
                if has_ellipsis && first_advanced.is_some() {
                    separator_since_advanced = true;
                }
            }
            let operand_shape = match op {
                IndexOp::Fancy(dim) => Some(IntTuple::from_ints(vec![dim.clone()])),
                IndexOp::ShapedArrayIndex(dims) => Some(IntTuple::from_ints(dims.clone())),
                IndexOp::Slice { .. } | IndexOp::NewAxis => {
                    if first_advanced.is_some() {
                        separator_since_advanced = true;
                    }
                    None
                }
                // Pyrefly's shared shaped-array kernel treats scalar Int as basic.
                IndexOp::Int => None,
            };
            if let Some(operand_shape) = operand_shape {
                let accumulated = broadcast_shape
                    .take()
                    .unwrap_or_else(|| IntTuple::from_ints(Vec::new()));
                broadcast_shape = Some(broadcast_shapes(&accumulated, &operand_shape)?);
                if first_advanced.is_none() {
                    first_advanced = Some((group, op_index));
                } else if separator_since_advanced {
                    separated = true;
                }
            }
        }

        let emission = match first_advanced {
            None => AdvancedIndexEmission::None,
            Some(_) if separated => AdvancedIndexEmission::Front,
            Some((group, op_index)) => AdvancedIndexEmission::At { group, op_index },
        };
        Ok(Self {
            broadcast_shape,
            emission,
        })
    }

    fn dims(&self) -> &[Int] {
        match &self.broadcast_shape {
            None => &[],
            Some(shape) => shape
                .as_concrete()
                .expect("advanced index operands always broadcast to a concrete-rank shape"),
        }
    }

    fn emits_at(&self, group: IndexOpGroup, op_index: usize) -> bool {
        matches!(
            self.emission,
            AdvancedIndexEmission::At {
                group: emission_group,
                op_index: emission_index,
            } if emission_group == group && emission_index == op_index
        )
    }

    fn emits_at_front(&self) -> bool {
        matches!(self.emission, AdvancedIndexEmission::Front)
    }
}

/// Apply multi-axis indexing with optional ellipsis.
/// `pre_ops` are applied left-to-right from dim 0.
/// `post_ops` are applied from the end (only when `has_ellipsis` is true).
/// Dims between pre and post (the ellipsis range) are preserved.
pub fn index_shape_multi(
    shape: &IntTuple,
    pre_ops: &[IndexOp],
    post_ops: &[IndexOp],
    has_ellipsis: bool,
) -> Result<IntTuple, ShapeError> {
    let pre_consumed = ops_dims_consumed(pre_ops);
    let post_consumed = ops_dims_consumed(post_ops);
    let total_consumed = pre_consumed + post_consumed;
    let shape_view = shape.view();
    if let IntTupleView::Concrete(shape_dims) = &shape_view
        && total_consumed > shape_dims.len()
    {
        return Err(ShapeError::TooManyIndices {
            got: total_consumed,
            max: shape_dims.len(),
        });
    }

    let advanced_plan = AdvancedIndexPlan::build(pre_ops, post_ops, has_ellipsis)?;
    match shape_view {
        IntTupleView::Concrete(shape_dims) => {
            let (pre_result, _) = apply_ops_to_dims(
                pre_ops,
                &shape_dims[..pre_consumed],
                IndexOpGroup::Pre,
                &advanced_plan,
            );

            let post_start = if has_ellipsis {
                shape_dims.len() - post_consumed
            } else {
                pre_consumed
            };
            let post_end = post_start + post_consumed;
            let (post_result, _) = apply_ops_to_dims(
                post_ops,
                &shape_dims[post_start..post_end],
                IndexOpGroup::Post,
                &advanced_plan,
            );

            let mut new_dims = pre_result;
            if has_ellipsis {
                // Preserve ellipsis-covered dims
                new_dims.extend_from_slice(&shape_dims[pre_consumed..post_start]);
                new_dims.extend(post_result);
            } else {
                new_dims.extend(post_result);
                new_dims.extend_from_slice(&shape_dims[post_end..]);
            }
            if advanced_plan.emits_at_front() {
                let mut with_advanced = advanced_plan.dims().to_vec();
                with_advanced.extend(new_dims);
                new_dims = with_advanced;
            }

            Ok(IntTuple::new(new_dims))
        }
        IntTupleView::Gradual => {
            if pre_consumed > 0 || post_consumed > 0 {
                return Ok(shapeless_shape());
            }

            let (pre_result, _) =
                apply_ops_to_dims(pre_ops, &[], IndexOpGroup::Pre, &advanced_plan);
            let (post_result, _) =
                apply_ops_to_dims(post_ops, &[], IndexOpGroup::Post, &advanced_plan);
            Ok(IntTuple::unpacked_from_parts(
                pre_result,
                gradual_shape_middle(),
                post_result,
            ))
        }
        IntTupleView::Unpacked {
            prefix,
            middle,
            suffix,
        } => {
            if pre_consumed > prefix.len() || post_consumed > suffix.len() {
                return Ok(shapeless_shape());
            }

            let (pre_result, _) = apply_ops_to_dims(
                pre_ops,
                &prefix[..pre_consumed],
                IndexOpGroup::Pre,
                &advanced_plan,
            );

            let post_suffix_start = suffix.len() - post_consumed;
            let (post_result, _) = apply_ops_to_dims(
                post_ops,
                &suffix[post_suffix_start..],
                IndexOpGroup::Post,
                &advanced_plan,
            );

            let remaining_prefix = &prefix[pre_consumed..];
            let remaining_suffix = &suffix[..post_suffix_start];

            let mut result_prefix = pre_result;
            result_prefix.extend_from_slice(remaining_prefix);
            if advanced_plan.emits_at_front() {
                let mut with_advanced = advanced_plan.dims().to_vec();
                with_advanced.extend(result_prefix);
                result_prefix = with_advanced;
            }
            let mut result_suffix = remaining_suffix.to_vec();
            result_suffix.extend(post_result);

            Ok(IntTuple::unpacked_from_parts(
                result_prefix,
                middle.clone(),
                result_suffix,
            ))
        }
    }
}

/// Create a shapeless shape (compatible with any shape).
fn shapeless_shape() -> IntTuple {
    IntTuple::shapeless()
}

/// Adjust a negative slice bound by adding dim size (Python negative index semantics).
/// E.g. -1 on dim N becomes N + (-1) = N - 1.
/// Also handles symbolic negation: -1 * X (from unary `-` on a Dim/Int expression)
/// becomes dim_size + (-1 * X) = dim_size - X.
fn adjust_negative(bound: Int, dim_size: &Int) -> Int {
    let is_negative = match &bound {
        // Literal negative: -1, -2, etc.
        Int::Literal(v) => *v < 0,
        // Symbolic negation: (-1 * X), (-2 * X), etc. from unary negation
        Int::Mul(left, _) if let Int::Literal(v) = left.as_ref() => *v < 0,
        _ => false,
    };
    if is_negative {
        Int::Add(Box::new(dim_size.clone()), Box::new(bound))
    } else {
        bound
    }
}

/// Compute stop - start, simplifying x - 0 to x.
fn sub_dim(stop: Int, start: Int) -> Int {
    match &start {
        Int::Literal(0) => stop,
        _ => Int::Sub(Box::new(stop), Box::new(start)),
    }
}

/// Apply step (stride) to a range dimension: ceil_div(range, step).
/// step=None or step=Literal(1) is identity. For literal range and step,
/// computes the exact integer ceiling division. For symbolic steps (Int,
/// Quantified), builds a symbolic ceil_div expression.
fn apply_step(range_dim: Int, step: Option<Int>) -> Int {
    let step = match step {
        None => return range_dim,
        Some(s) => s,
    };
    match &step {
        // Literal step: exact arithmetic
        Int::Literal(1) => range_dim,
        Int::Literal(s) if *s > 1 => {
            let s = *s;
            if let Int::Literal(n) = &range_dim {
                Int::Literal((*n + s - 1) / s)
            } else {
                // Symbolic range, literal step: ceil_div(range, step)
                let numerator = Int::Add(Box::new(range_dim), Box::new(Int::Literal(s - 1)));
                Int::FloorDiv(Box::new(numerator), Box::new(Int::Literal(s)))
            }
        }
        Int::Literal(s) if *s <= 0 => {
            // Negative or zero step: degenerate, return unknown
            Int::Int
        }
        // Symbolic step (Int var, Quantified): build ceil_div(range, step) symbolically
        _ => {
            // ceil_div(range, step) = (range + step - 1) // step
            let step_minus_1 = Int::Sub(Box::new(step.clone()), Box::new(Int::Literal(1)));
            let numerator = Int::Add(Box::new(range_dim), Box::new(step_minus_1));
            Int::FloorDiv(Box::new(numerator), Box::new(step))
        }
    }
}

/// Apply a basic consuming operation to a known dimension.
fn apply_index_op(op: &IndexOp, dim: &Int) -> Option<Int> {
    match op {
        IndexOp::Int => None,
        IndexOp::Slice { start, stop, step } => {
            let start = adjust_negative(start.clone().unwrap_or(Int::Literal(0)), dim);
            let stop = adjust_negative(stop.clone().unwrap_or_else(|| dim.clone()), dim);
            let range_dim = sub_dim(stop, start);
            Some(apply_step(range_dim, step.clone()))
        }
        IndexOp::ShapedArrayIndex(_) | IndexOp::Fancy(_) => {
            unreachable!(
                "advanced-index dispatch invariant violated: apply_ops_to_dims must consume advanced operations"
            )
        }
        IndexOp::NewAxis => unreachable!("NewAxis handled by apply_ops_to_dims"),
    }
}

/// Apply a sequence of `IndexOp`s to a slice of dimensions.
/// `NewAxis` ops insert a dim of size 1 without consuming a shape dimension.
/// Advanced ops consume one dimension and emit only where instructed by the
/// operation-wide advanced-index plan.
/// Returns (result_dims, number_of_shape_dims_consumed).
fn apply_ops_to_dims(
    ops: &[IndexOp],
    dims: &[Int],
    group: IndexOpGroup,
    advanced_plan: &AdvancedIndexPlan,
) -> (Vec<Int>, usize) {
    let mut new_dims = Vec::new();
    let mut dim_idx = 0;
    for (op_index, op) in ops.iter().enumerate() {
        match op {
            IndexOp::NewAxis => {
                new_dims.push(Int::Literal(1));
            }
            IndexOp::ShapedArrayIndex(_) | IndexOp::Fancy(_) => {
                dims.get(dim_idx)
                    .expect("rank checks must provide one dimension per consuming index operation");
                if advanced_plan.emits_at(group, op_index) {
                    new_dims.extend_from_slice(advanced_plan.dims());
                }
                dim_idx += 1;
            }
            _ => {
                let dim = dims
                    .get(dim_idx)
                    .expect("rank checks must provide one dimension per consuming index operation");
                if let Some(new_dim) = apply_index_op(op, dim) {
                    new_dims.push(new_dim);
                }
                dim_idx += 1;
            }
        }
    }
    (new_dims, dim_idx)
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::Arc;

    use pyrefly_python::module::Module;
    use pyrefly_python::module_name::ModuleName;
    use pyrefly_python::module_path::ModulePath;
    use pyrefly_python::nesting_context::NestingContext;
    use ruff_python_ast::Identifier;
    use ruff_python_ast::name::Name;
    use ruff_text_size::TextRange;
    use ruff_text_size::TextSize;

    use crate::class::Class;
    use crate::class::ClassDefIndex;
    use crate::class::ClassType;
    use crate::dimension::Int;
    use crate::dimension::ShapeError;
    use crate::dimension::gradual_size;
    use crate::lit_int::LitInt;
    use crate::literal::Lit;
    use crate::literal::LitStyle;
    use crate::literal::Literal;
    use crate::quantified::AnchorIndex;
    use crate::quantified::Quantified;
    use crate::quantified::QuantifiedIdentity;
    use crate::quantified::QuantifiedKind;
    use crate::quantified::QuantifiedOrigin;
    use crate::shaped_array::IndexOp;
    use crate::shaped_array::IntTuple;
    use crate::shaped_array::IntTupleRepr;
    use crate::shaped_array::IntTupleView;
    use crate::shaped_array::ShapedArrayType;
    use crate::shaped_array::broadcast_dim;
    use crate::shaped_array::broadcast_shapes;
    use crate::shaped_array::gradual_shape_middle;
    use crate::shaped_array::index_shape_multi;
    use crate::shaped_array::is_tuple_carrier_shape_middle;
    use crate::shaped_array::shape_to_tuple_carrier;
    use crate::shaped_array::shape_to_tuple_carrier_arg;
    use crate::shaped_array::tuple_carrier_to_shape;
    use crate::tuple::Tuple;
    use crate::type_var::PreInferenceVariance;
    use crate::type_var::Restriction;
    use crate::type_var::TypeVar;
    use crate::type_var_tuple::TypeVarTuple;
    use crate::types::AnyStyle;
    use crate::types::TArgs;
    use crate::types::TParams;
    use crate::types::Type;
    use crate::types::Var;

    /// Internal literal dimension `n` (`Type::Int(Int::Literal(n))`).
    fn size(n: i64) -> Type {
        Type::Int(Int::Literal(n))
    }

    fn dim(n: i64) -> Int {
        Int::Literal(n)
    }

    /// User-facing `Literal[n]` carrier element.
    fn literal(n: i64) -> Type {
        LitInt::new(n).to_explicit_type()
    }

    fn concrete_carrier(elts: Vec<Type>) -> Type {
        Type::Tuple(Tuple::Concrete(elts))
    }

    fn fake_module(module: &str) -> Module {
        Module::new(
            ModuleName::from_str(module),
            ModulePath::filesystem(PathBuf::from(module)),
            Arc::new("fake module contents".to_owned()),
        )
    }

    fn fake_class_type(module: &str, name: &str) -> ClassType {
        let module = fake_module(module);
        ClassType::new(
            Class::new(
                ClassDefIndex(0),
                Identifier::new(Name::new(name), TextRange::empty(TextSize::new(0))),
                NestingContext::toplevel(),
                module,
                None,
            ),
            TArgs::default(),
        )
    }

    fn fake_type_var(name: &str, kind: QuantifiedKind) -> TypeVar {
        TypeVar::new_with_kind(
            Identifier::new(Name::new(name), TextRange::empty(TextSize::new(0))),
            fake_module("__test__"),
            kind,
            Restriction::Unrestricted,
            None,
            PreInferenceVariance::Invariant,
        )
    }

    fn fake_type_var_tuple(name: &str) -> TypeVarTuple {
        TypeVarTuple::new(
            Identifier::new(Name::new(name), TextRange::empty(TextSize::new(0))),
            fake_module("__test__"),
            None,
        )
    }

    fn fake_tparam(name: &str, kind: QuantifiedKind) -> Quantified {
        Quantified::new(
            QuantifiedIdentity::new(
                ModuleName::from_str("__test__"),
                AnchorIndex::first(TextRange::default()),
                QuantifiedOrigin::Pep695,
            ),
            Name::new(name),
            kind,
            None,
            Restriction::Unrestricted,
            PreInferenceVariance::Invariant,
        )
    }

    fn scalar_symbol(name: &str) -> Int {
        Int::from_type(&Type::TypeVar(fake_type_var(name, QuantifiedKind::IntVar)))
            .expect("IntVar should construct a symbolic dimension")
    }

    fn shape_carrier(name: &str) -> Type {
        Type::TypeVar(fake_type_var(name, QuantifiedKind::TypeVar))
    }

    fn registered_array_shape_arg(shape_arg: Type) -> ShapedArrayType {
        let shape_param = fake_tparam("Shape", QuantifiedKind::TypeVar);
        let class = fake_class_type("arrays", "Array").class_object().clone();
        ShapedArrayType::new(
            ClassType::new(
                class,
                TArgs::new(Arc::new(TParams::new(vec![shape_param])), vec![shape_arg]),
            ),
            IntTuple::shapeless(),
        )
        .with_tuple_carrier_shape_arg(0)
    }

    fn registered_array_shape_arg_at(
        shape_arg_index: usize,
        shape_args: Vec<Type>,
    ) -> ShapedArrayType {
        let tparams = (0..shape_args.len())
            .map(|i| fake_tparam(&format!("Shape{i}"), QuantifiedKind::TypeVar))
            .collect();
        let class = fake_class_type("arrays", "Array").class_object().clone();
        ShapedArrayType::new(
            ClassType::new(
                class,
                TArgs::new(Arc::new(TParams::new(tparams)), shape_args),
            ),
            IntTuple::shapeless(),
        )
        .with_tuple_carrier_shape_arg(shape_arg_index)
    }

    #[test]
    fn concrete_shape_to_tuple_carrier() {
        let shape = IntTuple::from_types(vec![size(3), size(4), size(5)]);
        assert_eq!(
            shape_to_tuple_carrier(&shape),
            concrete_carrier(vec![literal(3), literal(4), literal(5)])
        );
    }

    #[test]
    fn grouped_tensor_indices_use_multi_index_dispatch() {
        let index_shape = vec![dim(2), dim(3)];
        let ops = [
            IndexOp::ShapedArrayIndex(index_shape.clone()),
            IndexOp::ShapedArrayIndex(index_shape),
        ];
        let middle = gradual_shape_middle();
        for (source_kind, shape, expected) in [
            (
                "concrete",
                IntTuple::from_types(vec![size(10), size(20), size(30)]),
                IntTuple::from_types(vec![size(2), size(3), size(30)]),
            ),
            ("gradual", IntTuple::shapeless(), IntTuple::shapeless()),
            (
                "unpacked",
                IntTuple::unpacked(vec![dim(10), dim(20)], middle.clone(), vec![dim(30)]),
                IntTuple::unpacked(vec![dim(2), dim(3)], middle.clone(), vec![dim(30)]),
            ),
        ] {
            assert_eq!(
                index_shape_multi(&shape, &ops, &[], false)
                    .unwrap_or_else(|e| panic!("{source_kind} source shape: {e:?}")),
                expected,
                "{source_kind} source shape",
            );
        }
    }

    #[test]
    fn advanced_indices_broadcast_once_across_all_operands() {
        let source = IntTuple::new(vec![dim(10), dim(20), dim(30), dim(40)]);
        assert_eq!(
            index_shape_multi(
                &source,
                &[
                    IndexOp::ShapedArrayIndex(vec![dim(3)]),
                    IndexOp::ShapedArrayIndex(vec![dim(2), dim(1)]),
                ],
                &[],
                false,
            )
            .unwrap(),
            IntTuple::new(vec![dim(2), dim(3), dim(30), dim(40)])
        );

        let symbolic = scalar_symbol("N");
        assert_eq!(
            index_shape_multi(
                &source,
                &[
                    IndexOp::ShapedArrayIndex(vec![symbolic.clone(), dim(1)]),
                    IndexOp::ShapedArrayIndex(vec![dim(1), dim(3)]),
                    IndexOp::ShapedArrayIndex(vec![symbolic.clone(), dim(3)]),
                ],
                &[],
                false,
            )
            .unwrap(),
            IntTuple::new(vec![symbolic, dim(3), dim(40)])
        );

        assert_eq!(
            index_shape_multi(
                &source,
                &[
                    IndexOp::Fancy(dim(3)),
                    IndexOp::ShapedArrayIndex(vec![dim(2), dim(1)]),
                ],
                &[],
                false,
            )
            .unwrap(),
            IntTuple::new(vec![dim(2), dim(3), dim(30), dim(40)])
        );
        assert_eq!(
            index_shape_multi(
                &source,
                &[
                    IndexOp::Fancy(Int::Int),
                    IndexOp::ShapedArrayIndex(vec![dim(2), dim(1)]),
                ],
                &[],
                false,
            )
            .unwrap(),
            IntTuple::new(vec![dim(2), Int::Int, dim(30), dim(40)])
        );
        assert_eq!(
            index_shape_multi(
                &source,
                &[IndexOp::Fancy(dim(2)), IndexOp::Fancy(dim(1))],
                &[],
                false,
            )
            .unwrap(),
            IntTuple::new(vec![dim(2), dim(30), dim(40)])
        );

        assert_eq!(
            index_shape_multi(
                &source,
                &[
                    IndexOp::ShapedArrayIndex(vec![]),
                    IndexOp::ShapedArrayIndex(vec![]),
                ],
                &[],
                false,
            )
            .unwrap(),
            IntTuple::new(vec![dim(30), dim(40)])
        );
    }

    #[test]
    fn gradual_dimension_broadcast_with_one_preserves_gradual() {
        assert_eq!(
            broadcast_shapes(&IntTuple::new(vec![Int::Int]), &IntTuple::new(vec![dim(1)]),)
                .unwrap(),
            IntTuple::new(vec![Int::Int])
        );
        assert_eq!(
            broadcast_shapes(&IntTuple::new(vec![Int::Int]), &IntTuple::new(vec![dim(2)]),)
                .unwrap(),
            IntTuple::new(vec![dim(2)])
        );
    }

    #[test]
    fn advanced_index_placement_uses_global_separators() {
        let full_slice = || IndexOp::Slice {
            start: None,
            stop: None,
            step: None,
        };
        for (case, source, pre_ops, post_ops, has_ellipsis, expected) in [
            (
                "slice separator",
                IntTuple::new(vec![dim(10), dim(20), dim(30), dim(40), dim(50)]),
                vec![
                    full_slice(),
                    IndexOp::Fancy(dim(3)),
                    full_slice(),
                    IndexOp::ShapedArrayIndex(vec![dim(2), dim(1)]),
                ],
                vec![],
                false,
                IntTuple::new(vec![dim(2), dim(3), dim(10), dim(30), dim(50)]),
            ),
            (
                "new-axis separator",
                IntTuple::new(vec![dim(10), dim(20), dim(30), dim(40)]),
                vec![
                    full_slice(),
                    IndexOp::Fancy(dim(3)),
                    IndexOp::NewAxis,
                    IndexOp::ShapedArrayIndex(vec![dim(2), dim(1)]),
                ],
                vec![],
                false,
                IntTuple::new(vec![dim(2), dim(3), dim(10), dim(1), dim(40)]),
            ),
            (
                "integer is transparent between advanced operands",
                IntTuple::new(vec![dim(10), dim(20), dim(30), dim(40), dim(50)]),
                vec![
                    full_slice(),
                    IndexOp::Fancy(dim(3)),
                    IndexOp::Int,
                    IndexOp::ShapedArrayIndex(vec![dim(2), dim(1)]),
                ],
                vec![],
                false,
                IntTuple::new(vec![dim(10), dim(2), dim(3), dim(50)]),
            ),
            (
                "leading integer before slice and advanced operand",
                IntTuple::new(vec![dim(10), dim(20), dim(30), dim(40)]),
                vec![IndexOp::Int, full_slice(), IndexOp::Fancy(dim(3))],
                vec![],
                false,
                IntTuple::new(vec![dim(20), dim(3), dim(40)]),
            ),
            (
                "trailing integer does not extend the advanced subspace",
                IntTuple::new(vec![dim(10), dim(20), dim(30), dim(40)]),
                vec![
                    full_slice(),
                    IndexOp::Fancy(dim(3)),
                    full_slice(),
                    IndexOp::Int,
                ],
                vec![],
                false,
                IntTuple::new(vec![dim(10), dim(3), dim(30)]),
            ),
            (
                "integer-only indexing stays basic",
                IntTuple::new(vec![dim(10), dim(20)]),
                vec![IndexOp::Int],
                vec![],
                false,
                IntTuple::new(vec![dim(20)]),
            ),
            (
                "zero-width ellipsis separator",
                IntTuple::new(vec![dim(10), dim(20), dim(30)]),
                vec![full_slice(), IndexOp::Fancy(dim(3))],
                vec![IndexOp::ShapedArrayIndex(vec![dim(2), dim(1)])],
                true,
                IntTuple::new(vec![dim(2), dim(3), dim(10)]),
            ),
            (
                "positive-width ellipsis separator",
                IntTuple::new(vec![dim(10), dim(20), dim(30), dim(40), dim(50)]),
                vec![full_slice(), IndexOp::Fancy(dim(3))],
                vec![IndexOp::ShapedArrayIndex(vec![dim(2), dim(1)])],
                true,
                IntTuple::new(vec![dim(2), dim(3), dim(10), dim(30), dim(40)]),
            ),
        ] {
            assert_eq!(
                index_shape_multi(&source, &pre_ops, &post_ops, has_ellipsis).unwrap(),
                expected,
                "{case}",
            );
        }
    }

    #[test]
    fn advanced_index_errors_precede_gradual_fallback_but_not_rank_errors() {
        let incompatible = [
            IndexOp::Fancy(dim(2)),
            IndexOp::ShapedArrayIndex(vec![dim(3)]),
        ];
        assert!(matches!(
            index_shape_multi(
                &IntTuple::new(vec![dim(10), dim(20)]),
                &incompatible,
                &[],
                false,
            ),
            Err(ShapeError::ShapeComputation { .. })
        ));
        match index_shape_multi(&IntTuple::new(vec![dim(10)]), &incompatible, &[], false) {
            Err(ShapeError::TooManyIndices { got, max }) => assert_eq!((got, max), (2, 1)),
            result => panic!("expected rank error before broadcast, got {result:?}"),
        }
        match index_shape_multi(&IntTuple::new(vec![]), &incompatible, &[], false) {
            Err(ShapeError::TooManyIndices { got, max }) => assert_eq!((got, max), (2, 0)),
            result => panic!("expected scalar rank error before broadcast, got {result:?}"),
        }
        assert!(matches!(
            super::index_shape_tensor(&IntTuple::new(vec![]), &[dim(2)]),
            Err(ShapeError::ScalarIndex)
        ));

        for source in [
            IntTuple::shapeless(),
            IntTuple::unpacked_from_parts(
                vec![],
                Type::Quantified(Box::new(fake_tparam("Ts", QuantifiedKind::TypeVarTuple))),
                vec![],
            ),
        ] {
            assert!(matches!(
                index_shape_multi(&source, &incompatible, &[], false),
                Err(ShapeError::ShapeComputation { .. })
            ));
        }
    }

    #[test]
    fn advanced_indices_preserve_known_unpacked_ends() {
        let middle = Type::Quantified(Box::new(fake_tparam("Ts", QuantifiedKind::TypeVarTuple)));
        let source = IntTuple::unpacked_from_parts(
            vec![dim(10), dim(20)],
            middle.clone(),
            vec![dim(30), dim(40)],
        );
        assert_eq!(
            index_shape_multi(
                &source,
                &[IndexOp::ShapedArrayIndex(vec![dim(2)])],
                &[],
                false,
            )
            .unwrap(),
            IntTuple::unpacked_from_parts(
                vec![dim(2), dim(20)],
                middle.clone(),
                vec![dim(30), dim(40)],
            )
        );
        assert_eq!(
            index_shape_multi(
                &source,
                &[
                    IndexOp::ShapedArrayIndex(vec![dim(2), dim(1)]),
                    IndexOp::Int,
                ],
                &[IndexOp::ShapedArrayIndex(vec![dim(3)])],
                true,
            )
            .unwrap(),
            IntTuple::unpacked_from_parts(vec![dim(2), dim(3)], middle.clone(), vec![dim(30)],)
        );
        assert_eq!(
            index_shape_multi(
                &source,
                &[
                    IndexOp::Slice {
                        start: None,
                        stop: None,
                        step: None,
                    },
                    IndexOp::Fancy(dim(3)),
                ],
                &[IndexOp::ShapedArrayIndex(vec![dim(2), dim(1)])],
                true,
            )
            .unwrap(),
            IntTuple::unpacked_from_parts(
                vec![dim(2), dim(3), dim(10)],
                middle.clone(),
                vec![dim(30)],
            )
        );
        assert_eq!(
            index_shape_multi(
                &source,
                &[],
                &[IndexOp::ShapedArrayIndex(vec![dim(3)])],
                true,
            )
            .unwrap(),
            IntTuple::unpacked_from_parts(
                vec![dim(10), dim(20)],
                middle.clone(),
                vec![dim(30), dim(3)],
            )
        );

        assert_eq!(
            index_shape_multi(
                &source,
                &[
                    IndexOp::ShapedArrayIndex(vec![dim(2)]),
                    IndexOp::ShapedArrayIndex(vec![dim(2)]),
                    IndexOp::ShapedArrayIndex(vec![dim(2)]),
                ],
                &[],
                false,
            )
            .unwrap(),
            IntTuple::shapeless()
        );
    }

    #[test]
    fn fancy_index_payload_preserves_output_shape() {
        let shape = IntTuple::new(vec![dim(10), dim(20)]);
        for (index_dim, expected) in [
            (Int::Literal(3), IntTuple::new(vec![dim(3), dim(20)])),
            (Int::Int, IntTuple::new(vec![Int::Int, dim(20)])),
        ] {
            assert_eq!(
                index_shape_multi(&shape, &[IndexOp::Fancy(index_dim)], &[], false).unwrap(),
                expected
            );
        }
    }

    #[test]
    fn prewrapped_invalid_slice_int_recovers_to_gradual() {
        let ordinary = Type::Quantified(Box::new(fake_tparam("T", QuantifiedKind::TypeVar)));
        let invalid = Type::Int(Int::add(size(1), ordinary));

        assert_eq!(super::type_to_dim(&invalid), None);
    }

    #[test]
    fn valid_slice_int_trees_survive_recursive_recovery() {
        let symbolic_var = Type::Int(Int::Symbolic(Box::new(Type::Var(Var::ZERO))));
        assert_eq!(
            super::type_to_dim(&symbolic_var),
            Some(Int::Symbolic(Box::new(Type::Var(Var::ZERO))))
        );

        let nested = Type::Int(Int::Symbolic(Box::new(Type::Int(Int::Symbolic(Box::new(
            Type::Var(Var::ZERO),
        ))))));
        assert_eq!(
            super::type_to_dim(&nested),
            Some(Int::Symbolic(Box::new(Type::Int(Int::Symbolic(Box::new(
                Type::Var(Var::ZERO)
            ))))))
        );
    }

    #[test]
    fn tensor_indexing_is_native_across_shape_kinds() {
        let symbolic = Int::Symbolic(Box::new(Type::Var(Var::ZERO)));
        let index_dims = vec![
            Int::Add(Box::new(dim(1)), Box::new(dim(1))),
            Int::Add(Box::new(symbolic.clone()), Box::new(dim(0))),
        ];
        let expected_index_dims = vec![dim(2), symbolic];

        let concrete = IntTuple::new(vec![dim(10), dim(20)]);
        let mut concrete_expected = expected_index_dims.clone();
        concrete_expected.push(dim(20));
        assert_eq!(
            super::index_shape_tensor(&concrete, &index_dims).unwrap(),
            IntTuple::new(concrete_expected)
        );

        assert_eq!(
            super::index_shape_tensor(&IntTuple::shapeless(), &index_dims).unwrap(),
            IntTuple::shapeless()
        );

        let middle = Type::Quantified(Box::new(fake_tparam("Ts", QuantifiedKind::TypeVarTuple)));
        let unpacked =
            IntTuple::unpacked_from_parts(vec![dim(10), dim(20)], middle.clone(), vec![dim(30)]);
        let mut unpacked_prefix = expected_index_dims;
        unpacked_prefix.push(dim(20));
        assert_eq!(
            super::index_shape_tensor(&unpacked, &index_dims).unwrap(),
            IntTuple::unpacked_from_parts(unpacked_prefix, middle, vec![dim(30)])
        );
    }

    #[test]
    fn slice_steps_and_negative_forms_remain_distinct() {
        let shape = IntTuple::new(vec![dim(10), dim(20)]);
        assert_eq!(
            super::index_shape_slice(&shape, None, None, None).unwrap(),
            shape
        );
        assert_eq!(
            super::index_shape_slice(&shape, None, None, Some(dim(3))).unwrap(),
            IntTuple::new(vec![dim(4), dim(20)])
        );
        for step in [dim(0), dim(-1)] {
            assert_eq!(
                super::index_shape_slice(&shape, None, None, Some(step)).unwrap(),
                IntTuple::new(vec![Int::Int, dim(20)])
            );
        }

        let symbolic = Int::Symbolic(Box::new(Type::Var(Var::ZERO)));
        let symbolic_step = Int::FloorDiv(
            Box::new(Int::Add(
                Box::new(dim(10)),
                Box::new(Int::Sub(Box::new(symbolic.clone()), Box::new(dim(1)))),
            )),
            Box::new(symbolic.clone()),
        );
        assert_eq!(
            super::index_shape_slice(&shape, None, None, Some(symbolic.clone())).unwrap(),
            IntTuple::new(vec![symbolic_step, dim(20)])
        );

        let raw_subtraction = Int::Sub(Box::new(dim(0)), Box::new(symbolic.clone()));
        let unary_negation = Int::Mul(Box::new(dim(-1)), Box::new(symbolic));
        assert_eq!(
            super::adjust_negative(raw_subtraction.clone(), &dim(10)),
            raw_subtraction
        );
        assert_eq!(
            super::adjust_negative(unary_negation.clone(), &dim(10)),
            Int::Add(Box::new(dim(10)), Box::new(unary_negation))
        );
    }

    #[test]
    fn multi_indexing_preserves_ellipsis_and_unpacked_middle() {
        let concrete = IntTuple::new(vec![dim(10), dim(20), dim(30), dim(40)]);
        let pre_ops = [
            IndexOp::Slice {
                start: Some(dim(1)),
                stop: Some(dim(9)),
                step: Some(dim(2)),
            },
            IndexOp::NewAxis,
        ];
        assert_eq!(
            index_shape_multi(&concrete, &pre_ops, &[IndexOp::Int], true).unwrap(),
            IntTuple::new(vec![dim(4), dim(1), dim(20), dim(30)])
        );

        match index_shape_multi(
            &IntTuple::new(vec![dim(10), dim(20)]),
            &[IndexOp::Int, IndexOp::NewAxis, IndexOp::Int, IndexOp::Int],
            &[],
            false,
        ) {
            Err(crate::dimension::ShapeError::TooManyIndices { got, max }) => {
                assert_eq!((got, max), (3, 2));
            }
            result => panic!("expected exact TooManyIndices error, got {result:?}"),
        }

        let middle = Type::Quantified(Box::new(fake_tparam("Ts", QuantifiedKind::TypeVarTuple)));
        let unpacked = IntTuple::unpacked_from_parts(
            vec![dim(10), dim(20)],
            middle.clone(),
            vec![dim(30), dim(40)],
        );
        let canonicalized_stop = Int::Add(Box::new(dim(4)), Box::new(dim(6)));
        let slice = IndexOp::Slice {
            start: None,
            stop: Some(canonicalized_stop),
            step: Some(dim(2)),
        };
        assert_eq!(
            index_shape_multi(&unpacked, &[slice], &[IndexOp::Int], true).unwrap(),
            IntTuple::unpacked_from_parts(vec![dim(5), dim(20)], middle, vec![dim(30)],)
        );
    }

    #[test]
    fn shapeless_projects_to_gradual_int_tuple() {
        let shape = IntTuple::shapeless();
        assert!(shape.is_shapeless());
        assert_eq!(
            shape.to_tuple_type(),
            Type::Tuple(Tuple::Unbounded(Box::new(gradual_size())))
        );
        assert_eq!(shape.to_tuple(), Tuple::Unbounded(Box::new(gradual_size())));
    }

    #[test]
    fn int_tuple_view_borrows_shape_structure() {
        let concrete = IntTuple::from_types(vec![size(2), size(3)]);
        match concrete.view() {
            IntTupleView::Concrete(dims) => assert_eq!(dims, &[dim(2), dim(3)]),
            _ => panic!("expected concrete shape view"),
        }

        let shapeless = IntTuple::shapeless();
        assert!(shapeless.is_shapeless());
        assert!(matches!(shapeless.view(), IntTupleView::Gradual));

        let middle = Type::Var(Var::ZERO);
        let unpacked = IntTuple::unpacked(vec![dim(1)], middle.clone(), vec![dim(4)]);
        match unpacked.view() {
            IntTupleView::Unpacked {
                prefix,
                middle: view_middle,
                suffix,
            } => {
                assert_eq!(prefix, &[dim(1)]);
                assert_eq!(view_middle, &middle);
                assert_eq!(suffix, &[dim(4)]);
            }
            _ => panic!("expected unpacked shape view"),
        }
    }

    #[test]
    fn from_tuple_wraps_valid_unbounded_shape_middle() {
        let elt = literal(5);
        let shape = IntTuple::from_tuple(Tuple::Unbounded(Box::new(elt.clone())));

        assert_eq!(
            shape,
            IntTuple::unpacked(
                Vec::new(),
                Type::Tuple(Tuple::Unbounded(Box::new(size(5)))),
                Vec::new(),
            )
        );
        assert_eq!(
            shape.to_tuple_type(),
            Type::Tuple(Tuple::Unpacked(Box::new((
                Vec::new(),
                Type::Tuple(Tuple::Unbounded(Box::new(size(5)))),
                Vec::new(),
            ))))
        );
    }

    #[test]
    fn unpacked_nested_shapeless_inttuple_middle_flattens() {
        assert_eq!(
            IntTuple::unpacked(
                Vec::new(),
                IntTuple::shapeless().to_shape_arg_type(),
                Vec::new(),
            ),
            IntTuple::shapeless()
        );
    }

    #[test]
    fn unpacked_nested_inttuple_affixes_canonicalize_in_order() {
        let middle = shape_carrier("Shape");
        let inner = IntTuple(IntTupleRepr::Unpacked {
            prefix: vec![Int::Add(Box::new(dim(1)), Box::new(dim(1)))],
            middle: Box::new(middle.clone()),
            suffix: vec![Int::Add(Box::new(dim(1)), Box::new(dim(2)))],
        });
        let outer = IntTuple(IntTupleRepr::Unpacked {
            prefix: vec![Int::Add(Box::new(dim(0)), Box::new(dim(1)))],
            middle: Box::new(inner.to_shape_arg_type()),
            suffix: vec![Int::Add(Box::new(dim(2)), Box::new(dim(2)))],
        });
        let expected = IntTuple::unpacked(vec![dim(1), dim(2)], middle, vec![dim(3), dim(4)]);
        let normalized = outer.normalize();

        assert_eq!(normalized, expected);
        assert_eq!(normalized.normalize(), normalized);
    }

    #[test]
    fn unpacked_from_types_recovers_invalid_boundary_affixes() {
        let middle = shape_carrier("Shape");
        let invalid_kind = Type::Quantified(Box::new(fake_tparam("T", QuantifiedKind::TypeVar)));
        let invalid_symbolic = Type::Int(Int::Symbolic(Box::new(bool_literal())));

        assert_eq!(
            IntTuple::unpacked_from_types(
                vec![
                    Type::Int(Int::Add(Box::new(dim(1)), Box::new(dim(2)))),
                    invalid_kind,
                ],
                middle.clone(),
                vec![invalid_symbolic],
            ),
            IntTuple::unpacked(vec![dim(3), Int::Int], middle, vec![Int::Int],)
        );
    }

    #[test]
    fn unpacked_invalid_unbounded_middle_recovers_to_gradual() {
        let elt = Type::ClassType(fake_class_type("torch", "Materialization"));
        let shape = IntTuple::from_tuple(Tuple::Unbounded(Box::new(elt)));

        assert_eq!(shape, IntTuple::shapeless());
    }

    #[test]
    fn from_tuple_valid_unbounded_middle_projects_as_tuple_carrier() {
        let shape = IntTuple::from_tuple(Tuple::Unbounded(Box::new(size(5))));

        assert_eq!(
            shape,
            IntTuple::unpacked(
                Vec::new(),
                Type::Tuple(Tuple::Unbounded(Box::new(size(5)))),
                Vec::new(),
            )
        );
    }

    #[test]
    fn broadcast_accepts_raw_gradual_tuple_middle() {
        let unnormalized = IntTuple(IntTupleRepr::Unpacked {
            prefix: vec![dim(2)],
            middle: Box::new(Type::Tuple(Tuple::Unbounded(Box::new(gradual_size())))),
            suffix: vec![dim(3)],
        });
        let concrete = IntTuple::from_types(vec![size(4), size(3)]);

        assert_eq!(
            broadcast_shapes(&concrete, &unnormalized).unwrap(),
            IntTuple::unpacked(Vec::new(), gradual_shape_middle(), vec![dim(3)],)
        );
    }

    #[test]
    fn broadcast_missing_leading_dimensions_in_both_orders() {
        let shorter = IntTuple::new(vec![dim(3)]);
        let longer = IntTuple::new(vec![dim(2), dim(3)]);

        for (left, right) in [(&shorter, &longer), (&longer, &shorter)] {
            assert_eq!(broadcast_shapes(left, right).unwrap(), longer);
        }
    }

    #[test]
    fn broadcast_literal_one_with_symbolic_dimensions() {
        let n = scalar_symbol("N");
        let left = IntTuple::new(vec![dim(1), n.clone()]);
        let right = IntTuple::new(vec![n.clone(), dim(1)]);

        assert_eq!(
            broadcast_shapes(&left, &right).unwrap(),
            IntTuple::new(vec![n.clone(), n])
        );
    }

    #[test]
    fn broadcast_gradual_dimension_with_known_non_one_in_both_orders() {
        let gradual = IntTuple::new(vec![Int::Int]);
        let known = IntTuple::new(vec![dim(7)]);

        for (left, right) in [(&gradual, &known), (&known, &gradual)] {
            assert_eq!(broadcast_shapes(left, right).unwrap(), known);
        }
    }

    #[test]
    fn broadcast_whole_gradual_shape_dispatch_arms() {
        let gradual = IntTuple::shapeless();
        let concrete = IntTuple::new(vec![dim(2), dim(3)]);
        let unpacked = IntTuple::unpacked(vec![dim(2)], shape_carrier("Shape"), vec![dim(3)]);

        for (case, left, right) in [
            ("gradual and concrete", &gradual, &concrete),
            ("concrete and gradual", &concrete, &gradual),
            ("gradual and gradual", &gradual, &gradual),
            ("gradual and unpacked", &gradual, &unpacked),
            ("unpacked and gradual", &unpacked, &gradual),
        ] {
            assert_eq!(broadcast_shapes(left, right).unwrap(), gradual, "{case}");
        }
    }

    #[test]
    fn broadcast_canonicalizes_arithmetic_dimensions() {
        let n = scalar_symbol("N");
        let cases = [
            (Int::Add(Box::new(dim(2)), Box::new(dim(3))), dim(5), dim(5)),
            (
                Int::Add(Box::new(n.clone()), Box::new(dim(0))),
                n.clone(),
                n,
            ),
        ];

        for (left, right, expected) in cases {
            assert_eq!(broadcast_dim(&left, &right, 0).unwrap(), expected);
            assert_eq!(broadcast_dim(&right, &left, 0).unwrap(), expected);
        }
    }

    #[test]
    fn broadcast_fixed_rank_mismatch_reports_absolute_position() {
        let left = IntTuple::new(vec![dim(2), dim(3), dim(4)]);
        let right = IntTuple::new(vec![dim(5), dim(4)]);

        assert_eq!(
            broadcast_shapes(&left, &right).unwrap_err().to_string(),
            "Cannot broadcast dimension Int[3] with dimension Int[5] at position 1"
        );
    }

    #[test]
    fn broadcast_concrete_shorter_than_unpacked_suffix_preserves_leading_suffix() {
        let concrete = IntTuple::new(vec![dim(4)]);
        let middle = shape_carrier("Shape");
        let unpacked = IntTuple::unpacked(vec![dim(2)], middle.clone(), vec![dim(3), dim(1)]);
        let expected = IntTuple::unpacked(vec![dim(2)], middle, vec![dim(3), dim(4)]);

        for (left, right) in [(&concrete, &unpacked), (&unpacked, &concrete)] {
            assert_eq!(broadcast_shapes(left, right).unwrap(), expected);
        }
    }

    #[test]
    fn broadcast_concrete_with_whole_shape_carrier_is_ambiguous() {
        let concrete = IntTuple::new(vec![dim(2), dim(3)]);
        let unpacked = IntTuple::unpacked(vec![dim(9)], shape_carrier("Shape"), vec![dim(3)]);

        for (left, right) in [(&concrete, &unpacked), (&unpacked, &concrete)] {
            assert_eq!(
                broadcast_shapes(left, right).unwrap_err().to_string(),
                "Cannot broadcast concrete dims with variadic shape: alignment is ambiguous"
            );
        }
    }

    #[test]
    fn broadcast_unpacked_same_middle_combines_prefixes_and_suffixes() {
        let middle = shape_carrier("Shape");
        let left = IntTuple::unpacked(vec![dim(1), dim(3)], middle.clone(), vec![dim(1), dim(5)]);
        let right = IntTuple::unpacked(vec![dim(2), dim(1)], middle.clone(), vec![dim(4), dim(1)]);

        assert_eq!(
            broadcast_shapes(&left, &right).unwrap(),
            IntTuple::unpacked(vec![dim(2), dim(3)], middle, vec![dim(4), dim(5)],)
        );
    }

    #[test]
    fn broadcast_unpacked_different_middles_degrade_to_gradual() {
        let left = IntTuple::unpacked(
            vec![dim(2)],
            shape_carrier("LeftShape"),
            vec![dim(1), dim(5)],
        );
        let right = IntTuple::unpacked(
            vec![dim(4)],
            shape_carrier("RightShape"),
            vec![dim(3), dim(1)],
        );

        assert_eq!(
            broadcast_shapes(&left, &right).unwrap(),
            IntTuple::unpacked(Vec::new(), gradual_shape_middle(), vec![dim(3), dim(5)],)
        );
    }

    #[test]
    fn broadcast_gradual_middle_absorbs_unmatched_concrete_dimensions() {
        let concrete = IntTuple::new(vec![dim(8), dim(6), dim(2), dim(3)]);
        let unpacked =
            IntTuple::unpacked(vec![dim(7)], gradual_shape_middle(), vec![dim(1), dim(3)]);
        let expected = IntTuple::unpacked(Vec::new(), gradual_shape_middle(), vec![dim(2), dim(3)]);

        for (left, right) in [(&concrete, &unpacked), (&unpacked, &concrete)] {
            assert_eq!(broadcast_shapes(left, right).unwrap(), expected);
        }
    }

    #[test]
    fn concrete_projects_to_int_dims_not_literals() {
        let n = Type::Quantified(Box::new(Quantified::new(
            QuantifiedIdentity::new(
                ModuleName::from_str("__test__"),
                AnchorIndex::first(TextRange::default()),
                QuantifiedOrigin::Pep695,
            ),
            Name::new("N"),
            QuantifiedKind::IntVar,
            None,
            Restriction::Unrestricted,
            PreInferenceVariance::Invariant,
        )));
        let projected = IntTuple::from_types(vec![size(2), n.clone()]).to_tuple_type();

        assert_eq!(
            projected,
            Type::Tuple(Tuple::Concrete(vec![
                size(2),
                Type::Int(Int::Symbolic(Box::new(n))),
            ]))
        );
        assert_eq!(projected.to_string(), "tuple[Int[2], Int[N]]");
    }

    #[test]
    fn unpacked_projection_preserves_shape() {
        let projected =
            IntTuple::unpacked(vec![dim(2)], Type::any_tuple(), vec![dim(3)]).to_tuple_type();

        assert_eq!(
            projected,
            Type::Tuple(Tuple::Unpacked(Box::new((
                vec![size(2)],
                Type::Tuple(Tuple::Unbounded(Box::new(gradual_size()))),
                vec![size(3)],
            ))))
        );
    }

    #[test]
    fn unpacked_typevartuple_middle_projects_as_variadic() {
        let s = Type::Quantified(Box::new(Quantified::new(
            QuantifiedIdentity::new(
                ModuleName::from_str("__test__"),
                AnchorIndex::first(TextRange::default()),
                QuantifiedOrigin::Pep695,
            ),
            Name::new("S"),
            QuantifiedKind::TypeVarTuple,
            None,
            Restriction::Unrestricted,
            PreInferenceVariance::Invariant,
        )));
        let projected = IntTuple::unpacked(vec![dim(2)], s.clone(), vec![dim(3)]).to_tuple_type();

        assert_eq!(
            projected,
            Type::Tuple(Tuple::Unpacked(Box::new(
                (vec![size(2)], s, vec![size(3)],)
            )))
        );
    }

    #[test]
    fn affixed_tuple_carrier_middle_projects_to_gradual_tuple_boundary() {
        let middle = Type::Var(Var::ZERO);
        let shape = IntTuple::unpacked(vec![dim(1)], middle.clone(), vec![dim(2)]);

        assert_eq!(
            shape.to_tuple_type(),
            Type::Tuple(Tuple::Unpacked(Box::new((
                vec![size(1)],
                Type::Tuple(Tuple::Unbounded(Box::new(gradual_size()))),
                vec![size(2)],
            ))))
        );
        assert_eq!(
            IntTuple::from_tuple(shape.to_tuple()),
            IntTuple::unpacked(vec![dim(1)], gradual_shape_middle(), vec![dim(2)])
        );
        assert_eq!(
            shape_to_tuple_carrier_arg(&shape),
            shape_to_tuple_carrier(&shape)
        );

        let whole_shape = IntTuple::unpacked(Vec::new(), middle.clone(), Vec::new());
        assert_eq!(shape_to_tuple_carrier_arg(&whole_shape), middle);
    }

    #[test]
    fn affixed_tuple_carrier_middle_displays_as_elements_unpack() {
        let middle = Type::Quantified(Box::new(Quantified::new(
            QuantifiedIdentity::new(
                ModuleName::from_str("__test__"),
                AnchorIndex::first(TextRange::default()),
                QuantifiedOrigin::Pep695,
            ),
            Name::new("S"),
            QuantifiedKind::TypeVar,
            None,
            Restriction::Unrestricted,
            PreInferenceVariance::Invariant,
        )));

        assert_eq!(
            IntTuple::unpacked(vec![dim(1)], middle, vec![dim(2)]).to_string(),
            "1, *Elements[S], 2"
        );
        assert_eq!(IntTuple::shapeless().to_string(), "*IntTuple");
    }

    #[test]
    fn literal_carrier_to_concrete_shape() {
        let carrier = concrete_carrier(vec![literal(3), literal(4), literal(5)]);
        assert_eq!(
            tuple_carrier_to_shape(&carrier),
            Some(IntTuple::from_types(vec![size(3), size(4), size(5)]))
        );
    }

    #[test]
    fn concrete_round_trip_both_directions() {
        let shape = IntTuple::from_types(vec![size(2), size(3)]);
        let carrier = shape_to_tuple_carrier(&shape);
        assert_eq!(tuple_carrier_to_shape(&carrier), Some(shape.clone()));

        let carrier = concrete_carrier(vec![literal(2), literal(3)]);
        let shape = tuple_carrier_to_shape(&carrier).unwrap();
        assert_eq!(shape_to_tuple_carrier(&shape), carrier);
    }

    #[test]
    fn explicit_any_internal_dimension_becomes_gradual_int_carrier() {
        let shape = IntTuple::from_types(vec![Type::Any(AnyStyle::Explicit)]);
        assert_eq!(
            shape_to_tuple_carrier(&shape),
            concrete_carrier(vec![gradual_size()])
        );
    }

    #[test]
    fn error_any_internal_dimension_becomes_gradual_int_carrier() {
        let shape = IntTuple::from_types(vec![Type::Any(AnyStyle::Error)]);
        assert_eq!(
            shape_to_tuple_carrier(&shape),
            concrete_carrier(vec![gradual_size()])
        );
    }

    #[test]
    fn symbolic_internal_dimension_round_trips_through_size_carrier() {
        let var = Type::Var(Var::ZERO);
        let shape = IntTuple::from_types(vec![var.clone()]);
        let carrier = concrete_carrier(vec![Type::Int(Int::Symbolic(Box::new(var)))]);

        assert_eq!(shape_to_tuple_carrier(&shape), carrier);
        assert_eq!(tuple_carrier_to_shape(&carrier), Some(shape));
    }

    #[test]
    fn raw_internal_intvar_carrier_elements_pass_through() {
        let quantified = Type::Quantified(Box::new(fake_tparam("N", QuantifiedKind::IntVar)));
        let dims = vec![size(8), Type::Any(AnyStyle::Explicit), quantified];

        assert_eq!(
            tuple_carrier_to_shape(&concrete_carrier(dims.clone())),
            Some(IntTuple::from_types(dims))
        );
    }

    #[test]
    fn int_carriers_with_internal_operands_pass_through() {
        let quantified = Type::Quantified(Box::new(fake_tparam("N", QuantifiedKind::IntVar)));
        let symint = Type::Int(Int::add(quantified.clone(), size(1)));

        assert_eq!(
            tuple_carrier_to_shape(&concrete_carrier(vec![symint.clone()])),
            Some(IntTuple::from_types(vec![symint.clone()]))
        );
    }

    #[test]
    fn raw_typevar_carrier_projects_to_variadic_shape() {
        let carrier = Type::Quantified(Box::new(Quantified::new(
            QuantifiedIdentity::new(
                ModuleName::from_str("__test__"),
                AnchorIndex::first(TextRange::default()),
                QuantifiedOrigin::Pep695,
            ),
            Name::new("Shape"),
            QuantifiedKind::TypeVar,
            None,
            Restriction::Unrestricted,
            PreInferenceVariance::Invariant,
        )));
        let shape = IntTuple::unpacked(Vec::new(), carrier.clone(), Vec::new());

        assert_eq!(tuple_carrier_to_shape(&carrier), Some(shape.clone()));
        assert_eq!(shape_to_tuple_carrier_arg(&shape), carrier);

        let carrier = Type::Var(Var::ZERO);
        let shape = IntTuple::unpacked(Vec::new(), carrier.clone(), Vec::new());
        assert_eq!(shape_to_tuple_carrier_arg(&shape), carrier);
    }

    #[test]
    fn shape_arg_type_is_first_class_int_tuple() {
        let shape = IntTuple::from_types(vec![size(2), size(3)]);
        let shape_arg = shape.to_shape_arg_type();
        assert_eq!(shape_arg, Type::IntTuple(Box::new(shape.clone())));
        assert_eq!(IntTuple::from_shape_arg_type(&shape_arg), Some(shape));
    }

    #[test]
    fn normalize_rebuilds_raw_representations_and_preserves_middle() {
        let raw_concrete = IntTuple(IntTupleRepr::Concrete(vec![Int::add(size(1), size(2))]));
        let concrete = IntTuple::from_types(vec![size(3)]);
        assert_ne!(raw_concrete, concrete);
        assert_eq!(raw_concrete.normalize(), concrete);
        assert_eq!(
            IntTuple::from_shape_arg_type(&Type::IntTuple(Box::new(raw_concrete))),
            Some(concrete.clone())
        );

        let gradual = IntTuple(IntTupleRepr::Gradual).normalize();
        assert_eq!(gradual, IntTuple::shapeless());

        let middle = Type::Var(Var::ZERO);
        let raw_unpacked = IntTuple(IntTupleRepr::Unpacked {
            prefix: vec![Int::add(size(1), size(2))],
            middle: Box::new(middle.clone()),
            suffix: vec![Int::add(size(3), size(4))],
        });
        let unpacked = raw_unpacked.normalize();
        assert_eq!(
            unpacked,
            IntTuple::unpacked(vec![dim(3)], middle.clone(), vec![dim(7)])
        );
        assert!(matches!(
            unpacked.view(),
            IntTupleView::Unpacked { middle: stored, .. } if stored == &middle
        ));

        for normalized in [concrete, gradual, unpacked] {
            assert_eq!(normalized.normalize(), normalized);
        }
    }

    #[test]
    fn registered_shape_projects_from_first_class_shape_arg() {
        let projected = IntTuple::from_types(vec![size(2), size(3)]);
        let tensor = registered_array_shape_arg(projected.to_shape_arg_type());

        assert_eq!(tensor.shape(), projected);
    }

    #[test]
    fn registered_shape_projects_from_legacy_tuple_carrier() {
        let projected = IntTuple::from_types(vec![size(6)]);
        let tensor = registered_array_shape_arg(concrete_carrier(vec![literal(6)]));

        assert_eq!(tensor.shape(), projected);
    }

    #[test]
    fn set_shape_updates_registered_shape_arg() {
        let old_shape = IntTuple::from_types(vec![size(2)]);
        let new_shape = IntTuple::from_types(vec![size(4), size(5)]);
        let mut tensor = registered_array_shape_arg(old_shape.to_shape_arg_type());

        tensor.set_shape(new_shape.clone());

        assert_eq!(tensor.shape(), new_shape.clone());
        assert_eq!(
            tensor.base_class.targs().as_slice()[0],
            new_shape.to_shape_arg_type()
        );
    }

    #[test]
    fn tuple_carrier_shape_arg_index_participates_in_identity() {
        let shape = IntTuple::from_types(vec![size(2)]);
        let shape_args = vec![shape.to_shape_arg_type(), shape.to_shape_arg_type()];
        let first_arg_shape = registered_array_shape_arg_at(0, shape_args.clone());
        let second_arg_shape = registered_array_shape_arg_at(1, shape_args);

        assert_eq!(first_arg_shape.shape(), second_arg_shape.shape());
        assert_ne!(first_arg_shape, second_arg_shape);
    }

    #[test]
    fn registered_shape_display_uses_projected_shape_arg() {
        let projected = IntTuple::from_types(vec![size(2), size(3)]);
        let tensor = registered_array_shape_arg(projected.to_shape_arg_type());
        assert_eq!(tensor.to_string(), "Array[[2, 3]]");

        let tensor = registered_array_shape_arg(IntTuple::shapeless().to_shape_arg_type());
        assert_eq!(tensor.to_string(), "Array");
    }

    #[test]
    #[should_panic(expected = "registered shaped-array shape argument should project to IntTuple")]
    fn registered_shape_with_invalid_carrier_panics() {
        let tensor = registered_array_shape_arg(Type::None);

        let _ = tensor.shape();
    }

    #[test]
    fn unpacked_first_class_int_tuple_middle_flattens() {
        let middle = IntTuple::from_types(vec![size(3), size(4)]).to_shape_arg_type();
        assert_eq!(
            IntTuple::unpacked(vec![dim(2)], middle, vec![dim(5)]),
            IntTuple::from_types(vec![size(2), size(3), size(4), size(5)])
        );
    }

    #[test]
    fn from_shape_arg_type_normalizes_raw_nested_int_tuple() {
        let inner = IntTuple::from_types(vec![size(2), size(3)]);
        let nested = IntTuple(IntTupleRepr::Unpacked {
            prefix: Vec::new(),
            middle: Box::new(inner.to_shape_arg_type()),
            suffix: Vec::new(),
        })
        .to_shape_arg_type();
        assert_eq!(IntTuple::from_shape_arg_type(&nested), Some(inner));
    }

    #[test]
    fn finite_tuple_unpack_flattens() {
        let middle = Type::Tuple(Tuple::Concrete(vec![
            size(2),
            LitInt::new(3).to_explicit_type(),
        ]));
        assert_eq!(
            IntTuple::unpacked(vec![dim(1)], middle, vec![dim(4)]),
            IntTuple::from_types(vec![size(1), size(2), size(3), size(4)])
        );
    }

    #[test]
    fn invalid_concrete_tuple_middle_recovers_with_gradual_dims() {
        let middle = Type::Tuple(Tuple::Concrete(vec![
            literal(2),
            Type::ClassType(fake_class_type("builtins", "str")),
            size(3),
            bool_literal(),
            Type::Quantified(Box::new(fake_tparam("T", QuantifiedKind::TypeVar))),
            Type::Quantified(Box::new(fake_tparam("P", QuantifiedKind::ParamSpec))),
            Type::Quantified(Box::new(fake_tparam("Ts", QuantifiedKind::TypeVarTuple))),
        ]));
        let shape = IntTuple::unpacked(vec![dim(1)], middle, vec![dim(4)]);
        let expected = IntTuple::from_types(vec![
            size(1),
            size(2),
            gradual_size(),
            size(3),
            gradual_size(),
            gradual_size(),
            gradual_size(),
            gradual_size(),
            size(4),
        ]);

        assert!(matches!(shape.view(), IntTupleView::Concrete(_)));
        assert_eq!(shape, expected);
        assert_eq!(shape.to_tuple_type(), expected.to_tuple_type());
        assert_eq!(IntTuple::from_tuple(shape.to_tuple()), shape);
    }

    #[test]
    fn invalid_non_tuple_middle_recovers_as_shapeless() {
        let middle = Type::ClassType(fake_class_type("builtins", "str"));
        let shape = IntTuple::unpacked(vec![dim(1)], middle, vec![dim(2)]);

        assert_eq!(shape, IntTuple::shapeless());
        assert_eq!(
            shape.to_tuple_type(),
            Type::Tuple(Tuple::Unbounded(Box::new(gradual_size())))
        );
        assert_eq!(IntTuple::from_tuple(shape.to_tuple()), shape);
    }

    #[test]
    fn any_middle_recovers_with_gradual_unbounded_middle() {
        let shapeless = IntTuple::unpacked(Vec::new(), Type::Any(AnyStyle::Implicit), Vec::new());
        assert_eq!(shapeless, IntTuple::shapeless());

        let shape = IntTuple::unpacked(vec![dim(1)], Type::Any(AnyStyle::Implicit), vec![dim(2)]);

        assert_eq!(
            shape,
            IntTuple::unpacked(
                vec![dim(1)],
                IntTuple::shapeless().to_shape_arg_type(),
                vec![dim(2)],
            )
        );
        assert_eq!(IntTuple::from_tuple(shape.to_tuple()), shape);
    }

    #[test]
    fn invalid_unbounded_tuple_middle_element_recovers_to_gradual() {
        let middle = Type::Tuple(Tuple::Unbounded(Box::new(Type::ClassType(
            fake_class_type("builtins", "str"),
        ))));
        let shape = IntTuple::unpacked(vec![dim(1)], middle, vec![dim(2)]);

        assert_eq!(
            shape,
            IntTuple::unpacked(
                vec![dim(1)],
                IntTuple::shapeless().to_shape_arg_type(),
                vec![dim(2)],
            )
        );
        assert_eq!(IntTuple::from_tuple(shape.to_tuple()), shape);
    }

    #[test]
    fn invalid_quantified_unbounded_middle_elements_recover_to_gradual() {
        let invalid_elements = [
            Type::Quantified(Box::new(fake_tparam("T", QuantifiedKind::TypeVar))),
            Type::Quantified(Box::new(fake_tparam("P", QuantifiedKind::ParamSpec))),
            Type::Quantified(Box::new(fake_tparam("Ts", QuantifiedKind::TypeVarTuple))),
            Type::TypeVarTuple(fake_type_var_tuple("Ts")),
        ];
        for elt in invalid_elements {
            let middle = Type::Tuple(Tuple::Unbounded(Box::new(elt)));
            let shape = IntTuple::unpacked(vec![dim(1)], middle, vec![dim(2)]);

            assert_eq!(
                shape,
                IntTuple::unpacked(
                    vec![dim(1)],
                    IntTuple::shapeless().to_shape_arg_type(),
                    vec![dim(2)],
                )
            );
            assert_eq!(IntTuple::from_tuple(shape.to_tuple()), shape);
        }
    }

    #[test]
    fn valid_unbounded_middle_elements_are_canonicalized() {
        let quantified = Type::Quantified(Box::new(fake_tparam("N", QuantifiedKind::IntVar)));
        let type_var = Type::TypeVar(fake_type_var("N", QuantifiedKind::IntVar));
        let int_type = Type::ClassType(fake_class_type("builtins", "int"));
        for (elt, expected) in [
            (literal(5), size(5)),
            (Type::Any(AnyStyle::Explicit), gradual_size()),
            (int_type, gradual_size()),
            (
                quantified.clone(),
                Type::Int(Int::Symbolic(Box::new(quantified))),
            ),
            (
                type_var.clone(),
                Type::Int(Int::Symbolic(Box::new(type_var))),
            ),
            (
                Type::Int(Int::add(size(1), size(2))),
                Type::Int(Int::Literal(3)),
            ),
        ] {
            let middle = Type::Tuple(Tuple::Unbounded(Box::new(elt)));
            let shape = IntTuple::unpacked(vec![dim(1)], middle, vec![dim(2)]);

            assert_eq!(
                shape,
                IntTuple::unpacked(
                    vec![dim(1)],
                    Type::Tuple(Tuple::Unbounded(Box::new(expected))),
                    vec![dim(2)],
                )
            );
            assert_eq!(IntTuple::from_tuple(shape.to_tuple()), shape);
        }
    }

    #[test]
    fn ordinary_var_unbounded_middle_element_recovers_to_gradual() {
        let middle = Type::Tuple(Tuple::Unbounded(Box::new(Type::Var(Var::ZERO))));
        let shape = IntTuple::unpacked(vec![dim(1)], middle, vec![dim(2)]);

        assert_eq!(
            shape,
            IntTuple::unpacked(
                vec![dim(1)],
                IntTuple::shapeless().to_shape_arg_type(),
                vec![dim(2)],
            )
        );
        assert_eq!(IntTuple::from_tuple(shape.to_tuple()), shape);
    }

    #[test]
    fn invalid_quantified_middle_kinds_recover_as_shapeless() {
        for kind in [QuantifiedKind::IntVar, QuantifiedKind::ParamSpec] {
            let middle = Type::Quantified(Box::new(fake_tparam("Invalid", kind)));
            let shape = IntTuple::unpacked(vec![dim(1)], middle, vec![dim(2)]);

            assert_eq!(shape, IntTuple::shapeless());
            assert_eq!(IntTuple::from_tuple(shape.to_tuple()), shape);
        }
    }

    #[test]
    fn scalar_typevar_middle_is_preserved_as_tuple_carrier() {
        let quantified = Type::Quantified(Box::new(fake_tparam("Shape", QuantifiedKind::TypeVar)));
        let whole_shape = IntTuple::unpacked(Vec::new(), quantified.clone(), Vec::new());
        assert_eq!(
            whole_shape,
            IntTuple::unpacked(Vec::new(), quantified.clone(), Vec::new(),)
        );

        let affixed = IntTuple::unpacked(vec![dim(1)], quantified.clone(), vec![dim(2)]);
        assert_eq!(
            affixed,
            IntTuple::unpacked(vec![dim(1)], quantified, vec![dim(2)],)
        );

        let direct = Type::TypeVar(fake_type_var("Shape", QuantifiedKind::TypeVar));
        let whole_shape = IntTuple::unpacked(Vec::new(), direct.clone(), Vec::new());
        assert_eq!(
            whole_shape,
            IntTuple::unpacked(Vec::new(), direct.clone(), Vec::new(),)
        );
        let affixed = IntTuple::unpacked(vec![dim(1)], direct.clone(), vec![dim(2)]);
        assert_eq!(
            affixed,
            IntTuple::unpacked(vec![dim(1)], direct, vec![dim(2)],)
        );
    }

    #[test]
    fn true_unresolved_variadic_middles_are_preserved() {
        for middle in [
            Type::Quantified(Box::new(fake_tparam("Shape", QuantifiedKind::TypeVarTuple))),
            Type::TypeVarTuple(fake_type_var_tuple("Shape")),
            Type::Var(Var::ZERO),
        ] {
            let shape = IntTuple::unpacked(vec![dim(1)], middle.clone(), vec![dim(2)]);

            assert_eq!(
                shape,
                IntTuple::unpacked(vec![dim(1)], middle, vec![dim(2)],)
            );
            if is_tuple_carrier_shape_middle(match shape.view() {
                IntTupleView::Unpacked { middle, .. } => middle,
                _ => unreachable!("test constructs unpacked shapes"),
            }) {
                assert_eq!(
                    IntTuple::from_tuple(shape.to_tuple()),
                    IntTuple::unpacked(vec![dim(1)], gradual_shape_middle(), vec![dim(2)])
                );
            } else {
                assert_eq!(IntTuple::from_tuple(shape.to_tuple()), shape);
            }
        }
    }

    #[test]
    fn unpacked_carrier_round_trip() {
        // tuple[Literal[2], *Ts, Literal[3]] <-> Unpacked([2], Ts, [3]).
        let middle = Type::Var(Var::ZERO);
        let shape = IntTuple::unpacked(vec![dim(2)], middle.clone(), vec![dim(3)]);
        let carrier = shape_to_tuple_carrier(&shape);
        assert_eq!(
            carrier,
            Type::Tuple(Tuple::Unpacked(Box::new((
                vec![literal(2)],
                middle,
                vec![literal(3)],
            ))))
        );
        assert_eq!(tuple_carrier_to_shape(&carrier), Some(shape));
    }

    #[test]
    fn unbounded_carriers_canonicalize_to_shapeless() {
        // Unbounded carriers have no recoverable rank or per-dimension values,
        // regardless of their element type.
        let any_unbounded = Type::any_tuple();
        let internal_unbounded = Type::Tuple(Tuple::Unbounded(Box::new(size(5))));
        let int_unbounded = Type::Tuple(Tuple::Unbounded(Box::new(Type::ClassType(
            fake_class_type("builtins", "int"),
        ))));
        let shapeless = IntTuple::shapeless();
        assert_eq!(
            tuple_carrier_to_shape(&any_unbounded),
            Some(shapeless.clone())
        );
        assert_eq!(
            tuple_carrier_to_shape(&internal_unbounded),
            Some(shapeless.clone())
        );
        assert_eq!(tuple_carrier_to_shape(&int_unbounded), Some(shapeless));
    }

    #[test]
    fn unpacked_tuple_carrier_middle_is_validated_strictly() {
        let valid = Type::Tuple(Tuple::Unpacked(Box::new((
            vec![literal(1)],
            Type::Tuple(Tuple::Concrete(vec![literal(3)])),
            vec![literal(2)],
        ))));
        assert_eq!(
            tuple_carrier_to_shape(&valid),
            Some(IntTuple::from_types(vec![size(1), size(3), size(2)]))
        );

        let invalid = Type::Tuple(Tuple::Unpacked(Box::new((
            vec![literal(1)],
            Type::Tuple(Tuple::Concrete(vec![Type::ClassType(fake_class_type(
                "builtins", "str",
            ))])),
            vec![literal(2)],
        ))));
        assert_eq!(tuple_carrier_to_shape(&invalid), None);
    }

    #[test]
    fn nested_concrete_tuple_carrier_middle_is_recursively_flattened() {
        let carrier = Type::Tuple(Tuple::Unpacked(Box::new((
            vec![literal(1)],
            Type::Tuple(Tuple::Unpacked(Box::new((
                vec![literal(2)],
                Type::Tuple(Tuple::Concrete(vec![literal(3)])),
                vec![literal(4)],
            )))),
            vec![literal(5)],
        ))));

        assert_eq!(
            tuple_carrier_to_shape(&carrier),
            Some(IntTuple::from_types(vec![
                size(1),
                size(2),
                size(3),
                size(4),
                size(5),
            ]))
        );
    }

    #[test]
    fn nested_unbounded_tuple_carrier_middle_recovers_to_gradual() {
        let carrier = Type::Tuple(Tuple::Unpacked(Box::new((
            vec![literal(1)],
            Type::Tuple(Tuple::Unpacked(Box::new((
                vec![literal(2)],
                Type::Tuple(Tuple::Unbounded(Box::new(literal(3)))),
                vec![literal(4)],
            )))),
            vec![literal(5)],
        ))));

        assert_eq!(
            tuple_carrier_to_shape(&carrier),
            Some(IntTuple::unpacked(
                vec![dim(1), dim(2)],
                gradual_shape_middle(),
                vec![dim(4), dim(5)],
            ))
        );
    }

    #[test]
    fn unpacked_tuple_carrier_unbounded_middle_recovers_to_gradual() {
        let carrier = Type::Tuple(Tuple::Unpacked(Box::new((
            vec![literal(1)],
            Type::Tuple(Tuple::Unbounded(Box::new(literal(5)))),
            vec![literal(2)],
        ))));

        assert_eq!(
            tuple_carrier_to_shape(&carrier),
            Some(IntTuple::unpacked(
                vec![dim(1)],
                gradual_shape_middle(),
                vec![dim(2)],
            ))
        );

        let invalid_prefix = Type::Tuple(Tuple::Unpacked(Box::new((
            vec![Type::ClassType(fake_class_type("builtins", "str"))],
            Type::Tuple(Tuple::Unbounded(Box::new(literal(5)))),
            vec![literal(2)],
        ))));
        assert_eq!(tuple_carrier_to_shape(&invalid_prefix), None);
    }

    #[test]
    fn bare_var_internal_tuple_middle_scalar_position_recovers_to_gradual_dim() {
        let middle = Type::Tuple(Tuple::Concrete(vec![Type::Var(Var::ZERO)]));
        let shape = IntTuple::unpacked(vec![dim(1)], middle, vec![dim(2)]);

        assert_eq!(
            shape,
            IntTuple::from_types(vec![size(1), gradual_size(), size(2)])
        );
    }

    #[test]
    fn bare_var_tuple_carrier_scalar_position_recovers_to_gradual_dim() {
        assert_eq!(
            tuple_carrier_to_shape(&concrete_carrier(vec![Type::Var(Var::ZERO)])),
            Some(IntTuple::from_types(vec![gradual_size()]))
        );
    }

    #[test]
    fn unsupported_carrier_elements_fail() {
        // A non-int literal element is not a valid dimension.
        let carrier = concrete_carrier(vec![LitInt::new(0).to_explicit_type(), bool_literal()]);
        assert_eq!(tuple_carrier_to_shape(&carrier), None);
        // A non-tuple carrier is not convertible at all.
        assert_eq!(tuple_carrier_to_shape(&literal(3)), None);
    }

    #[test]
    fn invalid_quantified_carrier_elements_fail_front_door() {
        for elt in [
            Type::Quantified(Box::new(fake_tparam("T", QuantifiedKind::TypeVar))),
            Type::Quantified(Box::new(fake_tparam("P", QuantifiedKind::ParamSpec))),
            Type::Quantified(Box::new(fake_tparam("Ts", QuantifiedKind::TypeVarTuple))),
            Type::TypeVar(fake_type_var("T", QuantifiedKind::TypeVar)),
            Type::TypeVarTuple(fake_type_var_tuple("Ts")),
        ] {
            assert_eq!(tuple_carrier_to_shape(&concrete_carrier(vec![elt])), None);
        }
    }

    #[test]
    fn unsupported_int_operands_fail() {
        let invalid_int = Type::Int(Int::Symbolic(Box::new(literal(1))));
        assert_eq!(
            tuple_carrier_to_shape(&concrete_carrier(vec![invalid_int.clone()])),
            None
        );
    }

    fn bool_literal() -> Type {
        Type::Literal(Box::new(Literal {
            value: Lit::Bool(true),
            style: LitStyle::Explicit,
        }))
    }
}
