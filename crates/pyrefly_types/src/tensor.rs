/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::fmt;
use std::fmt::Display;

use pyrefly_derive::TypeEq;
use pyrefly_derive::Visit;
use pyrefly_derive::VisitMut;
use pyrefly_util::display::commas_iter;

use crate::class::ClassType;
pub use crate::dimension::ShapeError;
pub use crate::dimension::SizeExpr;
use crate::dimension::canonicalize;
pub use crate::dimension::contains_var_in_type;
use crate::tuple::Tuple;
use crate::types::Type;

// ============================================================================
// Tensor Types
// ============================================================================

/// A tensor type with shape information
/// Example: Tensor[2, 3] represents a 2x3 tensor
/// Example: Tensor (no brackets) represents a shapeless tensor (Unpacked with tuple[Unknown, ...])
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(Visit, VisitMut, TypeEq)]
pub struct TensorType {
    /// Base tensor class (e.g., torch.Tensor)
    pub base_class: ClassType,
    /// Shape dimensions. Shapeless tensors use Unpacked([], tuple[Unknown, ...], []).
    pub shape: TensorShape,
}

impl TensorType {
    /// Create tensor type with shape information
    pub fn new(base_class: ClassType, shape: TensorShape) -> Self {
        Self { base_class, shape }
    }

    /// Create shapeless tensor type (compatible with any shape)
    /// Represented as Unpacked([], tuple[Unknown, ...], [])
    pub fn shapeless(base_class: ClassType) -> Self {
        Self {
            base_class,
            shape: TensorShape::Unpacked(Box::new((vec![], Type::any_tuple(), vec![]))),
        }
    }

    pub fn to_type(self) -> Type {
        Type::Tensor(Box::new(self))
    }

    /// Returns rank if shape is concrete, None for variadic/shapeless
    pub fn rank(&self) -> Option<usize> {
        match &self.shape {
            TensorShape::Concrete(dims) => Some(dims.len()),
            TensorShape::Unpacked(_) => None,
        }
    }

    /// Returns true if tensor has no shape information
    /// (represented as Unpacked([], tuple[Unknown/Any, ...], []))
    pub fn is_shapeless(&self) -> bool {
        is_shapeless(&self.shape)
    }
}

impl Display for TensorType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_shapeless() {
            write!(f, "{}", self.base_class.name())
        } else {
            write!(f, "{}[{}]", self.base_class.name(), self.shape)
        }
    }
}

/// Shape of a tensor
/// Similar to Tuple, supports unpacked TypeVarTuple for variadic shapes
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(Visit, VisitMut, TypeEq)]
pub enum TensorShape {
    /// Concrete shape: Tensor[2, 3, 4]
    /// List of dimensions, where each dimension is a Type
    /// Can be Type::Size(SizeExpr::Literal(n)), Type::Var(...), Type::Quantified(...), or Type::Any(...)
    Concrete(Vec<Type>),
    /// Variadic shape with unpacked TypeVarTuple: Tensor[2, *Shape, 4]
    /// Stores (prefix dims, middle TypeVarTuple, suffix dims)
    Unpacked(Box<(Vec<Type>, Type, Vec<Type>)>),
}

impl TensorShape {
    pub fn new(dims: Vec<SizeExpr>) -> Self {
        Self::Concrete(
            dims.into_iter()
                .map(|d| canonicalize(Type::Size(d)))
                .collect(),
        )
    }

    /// Create from Vec<Type> directly (for when dims are already wrapped)
    /// Automatically normalizes dimensions to canonical form:
    /// - Canonicalizes SizeExpr expressions (e.g., 2+3 -> 5, N+0 -> N)
    /// - Leaves Quantified, Var, and Any as-is (already canonical)
    pub fn from_types(dims: Vec<Type>) -> Self {
        Self::Concrete(dims.into_iter().map(canonicalize).collect())
    }

    /// Create variadic shape with unpacked TypeVarTuple: Tensor[2, *Shape, 4]
    pub fn unpacked(prefix: Vec<Type>, middle: Type, suffix: Vec<Type>) -> Self {
        // Canonicalize all dimensions
        let prefix: Vec<Type> = prefix.into_iter().map(canonicalize).collect();
        let suffix: Vec<Type> = suffix.into_iter().map(canonicalize).collect();

        Self::Unpacked(Box::new((prefix, middle, suffix)))
    }

    pub fn rank(&self) -> usize {
        match self {
            Self::Concrete(dims) => dims.len(),
            Self::Unpacked(_) => {
                // For unpacked shapes, rank is unknown at parse time
                // This should not be called for variadic shapes
                panic!("Cannot determine rank of variadic tensor shape")
            }
        }
    }

    pub fn is_empty(&self) -> bool {
        match self {
            Self::Concrete(dims) => dims.is_empty(),
            Self::Unpacked(_) => false, // Variadic shapes are never empty
        }
    }

    /// Get a slice of dimensions (only valid for concrete shapes)
    pub fn dims_slice(&self) -> &[Type] {
        match self {
            Self::Concrete(dims) => dims,
            Self::Unpacked(_) => {
                panic!("Cannot get dims_slice for variadic tensor shape")
            }
        }
    }

    /// Get the concrete dims if this is a Concrete shape
    pub fn as_concrete(&self) -> Option<&Vec<Type>> {
        match self {
            Self::Concrete(dims) => Some(dims),
            Self::Unpacked(_) => None,
        }
    }

    /// Get a mutable reference to concrete dims (for meta-shape operations)
    /// Panics if called on Unpacked shape
    pub fn dims_mut(&mut self) -> &mut Vec<Type> {
        match self {
            Self::Concrete(dims) => dims,
            Self::Unpacked(_) => {
                panic!("Cannot get mutable dims for variadic tensor shape")
            }
        }
    }

    /// Get dims as a Vec for concrete shapes, panics for unpacked
    /// This is used by meta-shape code that doesn't support variadic shapes yet
    pub fn dims(&self) -> &Vec<Type> {
        match self {
            Self::Concrete(dims) => dims,
            Self::Unpacked(_) => {
                panic!("Meta-shape operations do not yet support variadic tensor shapes")
            }
        }
    }

    /// Check if this is a variadic shape with unpacked TypeVarTuple
    pub fn is_unpacked(&self) -> bool {
        matches!(self, Self::Unpacked(_))
    }

    /// Check if all dimensions are literal (concrete integers)
    /// Returns false for variadic shapes
    pub fn all_literal(&self) -> bool {
        match self {
            Self::Concrete(dims) => dims
                .iter()
                .all(|ty| matches!(ty, Type::Size(SizeExpr::Literal(_)))),
            Self::Unpacked(_) => false,
        }
    }

    /// Extract literal dimension values if all are literal
    /// Returns None for variadic shapes
    pub fn as_literals(&self) -> Option<Vec<i64>> {
        match self {
            Self::Concrete(dims) if self.all_literal() => Some(
                dims.iter()
                    .filter_map(|ty| {
                        if let Type::Size(SizeExpr::Literal(n)) = ty {
                            Some(*n)
                        } else {
                            None
                        }
                    })
                    .collect(),
            ),
            _ => None,
        }
    }

    /// Get a dimension by index (only for concrete shapes)
    pub fn get_dim(&self, index: usize) -> Type {
        match self {
            Self::Concrete(dims) => dims.get(index).unwrap().clone(),
            Self::Unpacked(_) => {
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
        if let Self::Unpacked(_) = self {
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
}

impl Display for TensorShape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Concrete(dims) => {
                if dims.is_empty() {
                    write!(f, "()") // Scalar tensor: Tensor[()]
                } else {
                    write!(f, "{}", commas_iter(|| dims.iter()))
                }
            }
            Self::Unpacked(box (prefix, middle, suffix)) => {
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
                write!(f, "{}*{}{}", prefix_str, middle, suffix_str)
            }
        }
    }
}

/// Check if a shape is shapeless: Unpacked([], tuple[Any,...], [])
fn is_shapeless(shape: &TensorShape) -> bool {
    matches!(
        shape,
        TensorShape::Unpacked(box (prefix, Type::Tuple(Tuple::Unbounded(box Type::Any(_))), suffix))
            if prefix.is_empty() && suffix.is_empty()
    )
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
///    - concrete + unpacked(p, m, []) → shapeless if m is unbounded; error if m is TypeVarTuple
///    - unpacked + unpacked → if same TypeVarTuple with no extra suffix, broadcast prefixes;
///      if either is unbounded, shapeless; otherwise error
/// 3. Assemble result from step 2 output + broadcast suffix.
pub fn broadcast_shapes(a: &TensorShape, b: &TensorShape) -> Result<TensorShape, ShapeError> {
    match (a, b) {
        (TensorShape::Concrete(a_dims), TensorShape::Concrete(b_dims)) => {
            broadcast_concrete(a_dims, b_dims)
        }
        (TensorShape::Concrete(concrete), TensorShape::Unpacked(box (prefix, middle, suffix)))
        | (TensorShape::Unpacked(box (prefix, middle, suffix)), TensorShape::Concrete(concrete)) => {
            broadcast_concrete_with_unpacked(concrete, prefix, middle, suffix)
        }
        (
            TensorShape::Unpacked(box (ap, am, a_suf)),
            TensorShape::Unpacked(box (bp, bm, b_suf)),
        ) => broadcast_unpacked_with_unpacked(ap, am, a_suf, bp, bm, b_suf),
    }
}

/// Check if a middle type is an unbounded tuple (e.g., tuple[int, ...])
fn is_unbounded_middle(middle: &Type) -> bool {
    matches!(middle, Type::Tuple(Tuple::Unbounded(_)))
}

/// Broadcast a Concrete shape with an Unpacked shape.
///
/// Right-aligns concrete dims against the Unpacked's suffix, broadcasting pairwise.
/// After suffix consumption:
/// - If no concrete dims remain: preserve the Unpacked's prefix + middle.
/// - If concrete dims remain and middle is unbounded: result middle is tuple[Any, ...].
/// - If concrete dims remain and middle is TypeVarTuple: error.
fn broadcast_concrete_with_unpacked(
    concrete: &[Type],
    prefix: &[Type],
    middle: &Type,
    suffix: &[Type],
) -> Result<TensorShape, ShapeError> {
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
        Ok(TensorShape::Unpacked(Box::new((
            prefix.to_vec(),
            middle.clone(),
            result_suffix,
        ))))
    } else if is_unbounded_middle(middle) {
        // Can't align remaining concrete with unbounded middle
        Ok(TensorShape::Unpacked(Box::new((
            vec![],
            Type::any_tuple(),
            result_suffix,
        ))))
    } else {
        Err(ShapeError::InvalidDimension {
            value: 0,
            reason: "Cannot broadcast concrete dims with variadic shape: alignment is ambiguous"
                .to_owned(),
        })
    }
}

/// Broadcast two Unpacked shapes.
///
/// Right-aligns suffixes, broadcasting matched pairs. Then analyzes the middles:
/// - Same TypeVarTuple with no extra suffix dims: cancel middles, broadcast prefixes.
/// - Either middle is unbounded: result is shapeless + broadcast suffix.
/// - Otherwise: error.
fn broadcast_unpacked_with_unpacked(
    ap: &[Type],
    am: &Type,
    a_suf: &[Type],
    bp: &[Type],
    bm: &Type,
    b_suf: &[Type],
) -> Result<TensorShape, ShapeError> {
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

    if !has_extra && am_canon == bm_canon && !is_unbounded_middle(am) {
        // Same TypeVarTuple, no extra suffix → cancel middles, broadcast prefixes
        let prefix = match broadcast_concrete(ap, bp)? {
            TensorShape::Concrete(dims) => dims,
            _ => unreachable!(),
        };
        Ok(TensorShape::Unpacked(Box::new((
            prefix,
            am.clone(),
            result_suffix,
        ))))
    } else if is_unbounded_middle(am) || is_unbounded_middle(bm) {
        // At least one unbounded middle → can't determine alignment
        Ok(TensorShape::Unpacked(Box::new((
            vec![],
            Type::any_tuple(),
            result_suffix,
        ))))
    } else {
        // Different TypeVarTuples or structural mismatch
        Err(ShapeError::InvalidDimension {
            value: 0,
            reason: format!(
                "Cannot broadcast variadic shapes: incompatible middles *{} vs *{}",
                am, bm
            ),
        })
    }
}

/// Broadcast two concrete dimension lists following NumPy/PyTorch rules.
/// Returns a Concrete TensorShape.
fn broadcast_concrete(a_dims: &[Type], b_dims: &[Type]) -> Result<TensorShape, ShapeError> {
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
    Ok(TensorShape::from_types(result_dims))
}

/// Broadcast a single pair of dimensions.
/// Canonicalizes both sides so symbolic expressions that reduce to literals are caught.
fn broadcast_dim(a_ty: &Type, b_ty: &Type, position: usize) -> Result<Type, ShapeError> {
    let a_ty = canonicalize(a_ty.clone());
    let b_ty = canonicalize(b_ty.clone());
    match (&a_ty, &b_ty) {
        // Any is compatible with anything; prefer the non-Any side
        (Type::Any(_), _) => Ok(b_ty.clone()),
        (_, Type::Any(_)) => Ok(a_ty.clone()),
        // Equal dimensions (after canonicalization): compatible
        _ if a_ty == b_ty => Ok(a_ty.clone()),
        // Size(1) broadcasts to anything
        (Type::Size(SizeExpr::Literal(1)), _) => Ok(b_ty.clone()),
        (_, Type::Size(SizeExpr::Literal(1))) => Ok(a_ty.clone()),
        // Different non-broadcastable types: incompatible
        _ => Err(ShapeError::InvalidDimension {
            value: 0,
            reason: format!(
                "Cannot broadcast dimension {} with dimension {} at position {}",
                a_ty, b_ty, position
            ),
        }),
    }
}

// ============================================================================
// Tensor Indexing / Slicing
// ============================================================================

/// A single index operation, pre-classified by the type checker.
/// The type checker resolves Expr nodes into these before calling shape functions.
pub enum IndexOp {
    /// Integer index: removes the dimension
    Int,
    /// Slice: replaces dimension with (stop - start).
    /// `start` defaults to 0, `stop` defaults to the dimension size.
    Slice {
        start: Option<Type>,
        stop: Option<Type>,
    },
    /// Tensor index: replaces dimension with the index tensor's dims
    TensorIndex(Vec<Type>),
    /// Tuple/list fancy index: dimension becomes known size or unknown.
    /// `Some(n)` for concrete tuple of length n, `None` for list/unknown.
    Fancy(Option<i64>),
}

/// Apply a single integer index — removes first dimension.
/// E.g. `Tensor[10, 20][i]` -> `Tensor[20]`
pub fn index_shape_int(shape: &TensorShape) -> Result<TensorShape, ShapeError> {
    match shape {
        TensorShape::Concrete(dims) => {
            if dims.is_empty() {
                return Err(ShapeError::ScalarIndex);
            }
            Ok(TensorShape::Concrete(dims[1..].to_vec()))
        }
        TensorShape::Unpacked(box (prefix, middle, suffix)) if !prefix.is_empty() => {
            Ok(TensorShape::Unpacked(Box::new((
                prefix[1..].to_vec(),
                middle.clone(),
                suffix.clone(),
            ))))
        }
        // First dim is in variadic middle; can't determine result
        TensorShape::Unpacked(_) => Ok(shapeless_shape()),
    }
}

/// Apply a single slice to first dimension.
/// E.g. `Tensor[10, 20][2:5]` -> `Tensor[3, 20]`
pub fn index_shape_slice(
    shape: &TensorShape,
    start: Option<Type>,
    stop: Option<Type>,
) -> Result<TensorShape, ShapeError> {
    match shape {
        TensorShape::Concrete(dims) => {
            if dims.is_empty() {
                return Err(ShapeError::ScalarIndex);
            }
            let start = adjust_negative(
                start.unwrap_or_else(|| Type::Size(SizeExpr::Literal(0))),
                &dims[0],
            );
            let stop = adjust_negative(stop.unwrap_or_else(|| dims[0].clone()), &dims[0]);
            let new_first_dim = sub_dim(stop, start);
            let mut new_dims = vec![new_first_dim];
            new_dims.extend_from_slice(&dims[1..]);
            Ok(TensorShape::from_types(new_dims))
        }
        TensorShape::Unpacked(box (prefix, middle, suffix)) if !prefix.is_empty() => {
            let start = adjust_negative(
                start.unwrap_or_else(|| Type::Size(SizeExpr::Literal(0))),
                &prefix[0],
            );
            let stop = adjust_negative(stop.unwrap_or_else(|| prefix[0].clone()), &prefix[0]);
            let new_first_dim = sub_dim(stop, start);
            let mut new_prefix = vec![new_first_dim];
            new_prefix.extend_from_slice(&prefix[1..]);
            Ok(TensorShape::Unpacked(Box::new((
                new_prefix,
                middle.clone(),
                suffix.clone(),
            ))))
        }
        // Empty prefix: dim0 is hidden in the variadic middle
        TensorShape::Unpacked(_) => Ok(shapeless_shape()),
    }
}

/// Apply tensor-as-index — replaces first dim with index tensor's dims.
/// E.g. `Tensor[B, D1, D2][Tensor[T]]` -> `Tensor[T, D1, D2]`
pub fn index_shape_tensor(
    shape: &TensorShape,
    idx_dims: &[Type],
) -> Result<TensorShape, ShapeError> {
    match shape {
        TensorShape::Concrete(dims) => {
            if dims.is_empty() {
                return Err(ShapeError::ScalarIndex);
            }
            let mut new_dims = idx_dims.to_vec();
            new_dims.extend_from_slice(&dims[1..]);
            Ok(TensorShape::from_types(new_dims))
        }
        TensorShape::Unpacked(box (prefix, middle, suffix)) if !prefix.is_empty() => {
            let mut new_prefix = idx_dims.to_vec();
            new_prefix.extend_from_slice(&prefix[1..]);
            Ok(TensorShape::Unpacked(Box::new((
                new_prefix,
                middle.clone(),
                suffix.clone(),
            ))))
        }
        // First dim is in variadic middle; can't determine result
        TensorShape::Unpacked(_) => Ok(shapeless_shape()),
    }
}

/// Apply multi-axis indexing with optional ellipsis.
/// `pre_ops` are applied left-to-right from dim 0.
/// `post_ops` are applied from the end (only when `has_ellipsis` is true).
/// Dims between pre and post (the ellipsis range) are preserved.
pub fn index_shape_multi(
    shape: &TensorShape,
    pre_ops: &[IndexOp],
    post_ops: &[IndexOp],
    has_ellipsis: bool,
) -> Result<TensorShape, ShapeError> {
    match shape {
        TensorShape::Concrete(shape_dims) => {
            let non_ellipsis_count = pre_ops.len() + post_ops.len();
            if non_ellipsis_count > shape_dims.len() {
                return Err(ShapeError::TooManyIndices {
                    got: non_ellipsis_count,
                    max: shape_dims.len(),
                });
            }
            let ellipsis_dims = if has_ellipsis {
                shape_dims.len() - non_ellipsis_count
            } else {
                0
            };

            let pre_dims = &shape_dims[..pre_ops.len()];
            let post_start = pre_ops.len() + ellipsis_dims;
            let post_dims = &shape_dims[post_start..];
            let ellipsis_preserved = &shape_dims[pre_ops.len()..post_start];

            let pre_result = apply_ops_to_dims(pre_ops, pre_dims)?;
            let post_result = apply_ops_to_dims(post_ops, post_dims)?;

            let mut new_dims = pre_result;
            new_dims.extend_from_slice(ellipsis_preserved);
            new_dims.extend(post_result);
            // Add remaining unindexed dims (no ellipsis case)
            if !has_ellipsis {
                for dim in shape_dims.iter().skip(pre_ops.len()) {
                    new_dims.push(dim.clone());
                }
            }

            Ok(TensorShape::from_types(new_dims))
        }
        TensorShape::Unpacked(box (prefix, middle, suffix)) => {
            // Pre-ellipsis indices consume prefix left-to-right.
            // Post-ellipsis indices consume suffix from the right.
            // Ellipsis (or unindexed middle) covers remaining prefix + middle + remaining suffix.
            if pre_ops.len() > prefix.len() || post_ops.len() > suffix.len() {
                return Ok(shapeless_shape());
            }

            let pre_dims = &prefix[..pre_ops.len()];
            let pre_result = apply_ops_to_dims(pre_ops, pre_dims)?;

            let post_suffix_start = suffix.len() - post_ops.len();
            let post_dims = &suffix[post_suffix_start..];
            let post_result = apply_ops_to_dims(post_ops, post_dims)?;

            let remaining_prefix = &prefix[pre_ops.len()..];
            let remaining_suffix = &suffix[..post_suffix_start];

            let mut result_prefix = pre_result;
            result_prefix.extend_from_slice(remaining_prefix);
            let mut result_suffix = remaining_suffix.to_vec();
            result_suffix.extend(post_result);

            Ok(TensorShape::Unpacked(Box::new((
                result_prefix,
                middle.clone(),
                result_suffix,
            ))))
        }
    }
}

/// Create a shapeless shape (compatible with any shape).
fn shapeless_shape() -> TensorShape {
    TensorShape::Unpacked(Box::new((vec![], Type::any_tuple(), vec![])))
}

/// Adjust a negative literal slice bound by adding dim size.
/// E.g. -1 on dim N becomes N + (-1) = N - 1.
fn adjust_negative(bound: Type, dim_size: &Type) -> Type {
    if let Type::Size(SizeExpr::Literal(v)) = &bound
        && *v < 0
    {
        return Type::Size(SizeExpr::Add(Box::new(dim_size.clone()), Box::new(bound)));
    }
    bound
}

/// Compute stop - start, simplifying x - 0 to x.
fn sub_dim(stop: Type, start: Type) -> Type {
    match &start {
        Type::Size(SizeExpr::Literal(0)) => stop,
        _ => Type::Size(SizeExpr::Sub(Box::new(stop), Box::new(start))),
    }
}

/// Apply a single `IndexOp` to a known dimension.
/// Returns `Some(new_dim)` for ops that keep the dim, `None` for `Int` (dim removed).
fn apply_index_op(op: &IndexOp, dim: &Type) -> Option<Type> {
    match op {
        IndexOp::Int => None,
        IndexOp::Slice { start, stop } => {
            let start = adjust_negative(
                start
                    .clone()
                    .unwrap_or_else(|| Type::Size(SizeExpr::Literal(0))),
                dim,
            );
            let stop = adjust_negative(stop.clone().unwrap_or_else(|| dim.clone()), dim);
            Some(sub_dim(stop, start))
        }
        IndexOp::TensorIndex(idx_dims) => {
            // Multi-axis tensor indexing: this case shouldn't appear in apply_index_op
            // since tensor indexing replaces dims entirely. Treat as fancy.
            if idx_dims.is_empty() {
                None
            } else {
                // Return the first dim of the index tensor; the rest are handled
                // at a higher level. For multi-axis, this degrades to unknown.
                Some(Type::any_implicit())
            }
        }
        IndexOp::Fancy(Some(n)) => Some(Type::Size(SizeExpr::Literal(*n))),
        IndexOp::Fancy(None) => Some(Type::any_implicit()),
    }
}

/// Apply a sequence of `IndexOp`s to a corresponding slice of dimensions.
/// Returns the resulting dims (with int-indexed dims removed).
fn apply_ops_to_dims(ops: &[IndexOp], dims: &[Type]) -> Result<Vec<Type>, ShapeError> {
    let mut new_dims = Vec::new();
    for (op, dim) in ops.iter().zip(dims.iter()) {
        if let Some(new_dim) = apply_index_op(op, dim) {
            new_dims.push(new_dim);
        }
    }
    Ok(new_dims)
}
