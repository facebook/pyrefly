# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import shape_extensions.dsl as dsl
from shape_extensions import (
    gufunc_broadcast,
    Int,
    IntTuple,
    IntTuples,
    type_shape_dsl_function,
)

@type_shape_dsl_function
def int_min(a: Int, b: Int) -> Int:
    if a == b:
        return a
    if dsl.is_concrete_int(a) and dsl.is_concrete_int(b):
        if a < b:
            return a
        return b
    return dsl.Int.gradual()

@type_shape_dsl_function
def diag_extent(n: Int, k: int) -> Int:
    # Non-literal Flag arguments become gradual before the DSL body is evaluated.
    if k < 0:
        return n - k
    return n + k

@type_shape_dsl_function
def matmul_shape(left: IntTuple, right: IntTuple) -> IntTuple:
    if len(left) == 0 or len(right) == 0:
        return dsl.Invalid("matmul expects at least 1-D arrays")
    operands = dsl.IntTuples((left, right))
    if len(right) == 1:
        spec = "(n),(n)->()"
        return gufunc_broadcast(spec, operands)
    if len(left) == 1:
        spec = "(n),(n,p)->(p)"
        return gufunc_broadcast(spec, operands)
    spec = "(m,n),(n,p)->(m,p)"
    return gufunc_broadcast(spec, operands)

@type_shape_dsl_function
def matvec_shape(left: IntTuple, right: IntTuple) -> IntTuple:
    spec = "(m,n),(n)->(m)"
    operands = dsl.IntTuples((left, right))
    return gufunc_broadcast(spec, operands)

@type_shape_dsl_function
def vecdot_shape(left: IntTuple, right: IntTuple) -> IntTuple:
    spec = "(n),(n)->()"
    operands = dsl.IntTuples((left, right))
    return gufunc_broadcast(spec, operands)

@type_shape_dsl_function
def vecmat_shape(left: IntTuple, right: IntTuple) -> IntTuple:
    spec = "(n),(n,m)->(m)"
    operands = dsl.IntTuples((left, right))
    return gufunc_broadcast(spec, operands)

@type_shape_dsl_function
def reduce_shape(
    shape: IntTuple,
    axis: int | tuple[int, ...] | None,
    keepdims: bool,
) -> IntTuple:
    if axis is None:
        axes = range(len(shape))
    elif dsl.is_int_value(axis):
        axes = (axis,)
    else:
        axes = axis
    # The DSL does not support unary negation of a Flag integer.
    if any(item < 0 - len(shape) or item >= len(shape) for item in axes):
        return dsl.Invalid("axis out of bounds")
    normalized = tuple(item + len(shape) if item < 0 else item for item in axes)
    if any(normalized.count(item) > 1 for item in normalized):
        return dsl.Invalid("duplicate axis")
    if keepdims:
        return dsl.IntTuple(
            (1 if index in normalized else shape[index] for index in range(len(shape)))
        )
    return dsl.IntTuple(
        (shape[index] for index in range(len(shape)) if index not in normalized)
    )

@type_shape_dsl_function
def stack_shape(shapes: IntTuples, axis: int) -> IntTuple:
    if len(shapes) == 0:
        return dsl.Invalid("stack expects a non-empty sequence of arrays")
    first = shapes[0]
    output_rank = len(first) + 1
    # Unary minus is not supported by the type-level DSL.
    if axis < 0 - output_rank or axis >= output_rank:
        return dsl.Invalid("stack axis out of range")
    if axis < 0:
        norm_axis = axis + output_rank
    else:
        norm_axis = axis + 0
    rank_mismatches = dsl.IntTuple(
        (0 if len(shape) == len(first) else 1 for shape in shapes)
    )
    if any(mismatch == 1 for mismatch in rank_mismatches):
        return dsl.Invalid("stack expects all arrays to have the same rank")
    # Equal ranks are established above, so indexing every member by an index of
    # `first` is in bounds.
    mismatches = dsl.IntTuple(
        (
            1 if any(shape[index] != first[index] for index in range(len(first))) else 0
            for shape in shapes
        )
    )
    if any(mismatch == 1 for mismatch in mismatches):
        return dsl.Invalid("stack expects all arrays to have the same shape")
    return dsl.IntTuple(
        (
            len(shapes)
            if index == norm_axis
            else first[index if index < norm_axis else index - 1]
            for index in range(output_rank)
        )
    )

@type_shape_dsl_function
def expand_dims_shape(shape: IntTuple, axis: int) -> IntTuple:
    output_rank = len(shape) + 1
    # Unary minus is not supported by the type-level DSL.
    if axis < 0 - output_rank or axis >= output_rank:
        return dsl.Invalid("expand_dims axis out of bounds")
    if axis < 0:
        norm_axis = axis + output_rank
    else:
        norm_axis = axis + 0
    return dsl.IntTuple(
        (
            1
            if index == norm_axis
            else shape[index if index < norm_axis else index - 1]
            for index in range(output_rank)
        )
    )

@type_shape_dsl_function
def swapaxes_shape(shape: IntTuple, axis1: int, axis2: int) -> IntTuple:
    rank = len(shape)
    if axis1 < 0 - rank or axis1 >= rank or axis2 < 0 - rank or axis2 >= rank:
        return dsl.Invalid("swapaxes axis out of bounds")
    if axis1 < 0:
        first = axis1 + rank
    else:
        first = axis1 + 0
    if axis2 < 0:
        second = axis2 + rank
    else:
        second = axis2 + 0
    return dsl.IntTuple(
        (
            shape[second]
            if index == first
            else shape[first]
            if index == second
            else shape[index]
            for index in range(rank)
        )
    )

@type_shape_dsl_function
def reverse_shape(shape: IntTuple) -> IntTuple:
    rank = len(shape)
    return dsl.IntTuple((shape[rank - index - 1] for index in range(rank)))

@type_shape_dsl_function
def transpose_shape(shape: IntTuple, axes: IntTuple) -> IntTuple:
    if len(axes) != len(shape):
        return dsl.Invalid("transpose axes must match the array rank")
    if any(axis < 0 - len(shape) or axis >= len(shape) for axis in axes):
        return dsl.Invalid("transpose axis out of bounds")
    normalized = dsl.IntTuple(
        (axis + len(shape) if axis < 0 else axis for axis in axes)
    )
    duplicates = dsl.IntTuple(
        (
            1
            if any(normalized[index] == normalized[other] for other in range(index))
            else 0
            for index in range(len(shape))
        )
    )
    if any(duplicate == 1 for duplicate in duplicates):
        return dsl.Invalid("transpose axes must be unique")
    return dsl.IntTuple((shape[axis] for axis in normalized))

@type_shape_dsl_function
def nonzero_shapes(shape: IntTuple) -> IntTuples:
    if len(shape) == 0:
        return dsl.Invalid("nonzero requires at least one dimension")
    return dsl.IntTuples(
        (dsl.IntTuple((dsl.Int.gradual(),)) for _ in range(len(shape)))
    )

@type_shape_dsl_function
def squeeze_shape(shape: IntTuple, axis: int | tuple[int, ...] | None) -> IntTuple:
    if axis is None:
        return dsl.IntTuple((extent for extent in shape if extent != 1))
    if dsl.is_int_value(axis):
        if axis == -2:
            tail = shape[-2:]
            if len(tail) < 2:
                return dsl.Invalid("squeeze axis out of bounds")
            extent = shape[-2]
            if extent == 1:
                return dsl.concat(shape[:-2], shape[-1:])
            if dsl.is_concrete_int(extent):
                return dsl.Invalid("squeeze axis must have length 1")
        if axis == 0 or axis == -1:
            if len(shape) == 0:
                return shape
        axes = (axis,)
    else:
        axes = axis
    if any(item < 0 - len(shape) or item >= len(shape) for item in axes):
        return dsl.Invalid("squeeze axis out of bounds")
    normalized = tuple(item + len(shape) if item < 0 else item for item in axes)
    if any(normalized.count(item) > 1 for item in normalized):
        return dsl.Invalid("squeeze axes must be unique")
    if any(shape[item] != 1 for item in normalized):
        return dsl.Invalid("squeeze axis must have length 1")
    return dsl.IntTuple(
        (shape[index] for index in range(len(shape)) if index not in normalized)
    )
