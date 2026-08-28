# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import shape_extensions.dsl as dsl
from shape_extensions import Int, IntTuple, type_shape_dsl_function

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
    if len(left) != 2 or len(right) != 2:
        return dsl.Invalid("matmul expects 2-D arrays")
    left_inner = left[1]
    right_inner = right[0]
    if (
        dsl.is_concrete_int(left_inner)
        and dsl.is_concrete_int(right_inner)
        and left_inner != right_inner
    ):
        return dsl.Invalid("matmul inner dimensions must match")
    return dsl.IntTuple((left[0], right[1]))

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
