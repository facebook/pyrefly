/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Shape algebra for indexing shaped arrays.

use crate::dimension::Int;
use crate::dimension::ShapeError;
use crate::shaped_array::IntTuple;
use crate::shaped_array::IntTupleView;
use crate::shaped_array::broadcast_shapes;
use crate::shaped_array::gradual_shape_middle;

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
    use ruff_python_ast::Identifier;
    use ruff_python_ast::name::Name;
    use ruff_text_size::TextRange;
    use ruff_text_size::TextSize;

    use super::*;
    use crate::quantified::AnchorIndex;
    use crate::quantified::Quantified;
    use crate::quantified::QuantifiedIdentity;
    use crate::quantified::QuantifiedKind;
    use crate::quantified::QuantifiedOrigin;
    use crate::type_var::PreInferenceVariance;
    use crate::type_var::Restriction;
    use crate::type_var::TypeVar;
    use crate::types::Type;
    use crate::types::Var;

    fn size(value: i64) -> Type {
        Type::Int(Int::Literal(value))
    }

    fn dim(value: i64) -> Int {
        Int::Literal(value)
    }

    fn fake_module(module: &str) -> Module {
        Module::new(
            ModuleName::from_str(module),
            ModulePath::filesystem(PathBuf::from(module)),
            Arc::new("fake module contents".to_owned()),
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

    fn scalar_symbol(name: &str) -> Int {
        Int::from_type(&Type::TypeVar(fake_type_var(name, QuantifiedKind::IntVar)))
            .expect("IntVar should construct a symbolic dimension")
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
            index_shape_tensor(&IntTuple::new(vec![]), &[dim(2)]),
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
            index_shape_tensor(&concrete, &index_dims).unwrap(),
            IntTuple::new(concrete_expected)
        );

        assert_eq!(
            index_shape_tensor(&IntTuple::shapeless(), &index_dims).unwrap(),
            IntTuple::shapeless()
        );

        let middle = Type::Quantified(Box::new(fake_tparam("Ts", QuantifiedKind::TypeVarTuple)));
        let unpacked =
            IntTuple::unpacked_from_parts(vec![dim(10), dim(20)], middle.clone(), vec![dim(30)]);
        let mut unpacked_prefix = expected_index_dims;
        unpacked_prefix.push(dim(20));
        assert_eq!(
            index_shape_tensor(&unpacked, &index_dims).unwrap(),
            IntTuple::unpacked_from_parts(unpacked_prefix, middle, vec![dim(30)])
        );
    }

    #[test]
    fn slice_steps_and_negative_forms_remain_distinct() {
        let shape = IntTuple::new(vec![dim(10), dim(20)]);
        assert_eq!(index_shape_slice(&shape, None, None, None).unwrap(), shape);
        assert_eq!(
            index_shape_slice(&shape, None, None, Some(dim(3))).unwrap(),
            IntTuple::new(vec![dim(4), dim(20)])
        );
        for step in [dim(0), dim(-1)] {
            assert_eq!(
                index_shape_slice(&shape, None, None, Some(step)).unwrap(),
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
            index_shape_slice(&shape, None, None, Some(symbolic.clone())).unwrap(),
            IntTuple::new(vec![symbolic_step, dim(20)])
        );

        let raw_subtraction = Int::Sub(Box::new(dim(0)), Box::new(symbolic.clone()));
        let unary_negation = Int::Mul(Box::new(dim(-1)), Box::new(symbolic));
        assert_eq!(
            adjust_negative(raw_subtraction.clone(), &dim(10)),
            raw_subtraction
        );
        assert_eq!(
            adjust_negative(unary_negation.clone(), &dim(10)),
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
}
