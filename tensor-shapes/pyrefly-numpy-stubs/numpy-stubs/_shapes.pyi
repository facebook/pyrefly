# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import shape_extensions.dsl as dsl
from shape_extensions import Int, type_shape_dsl_function
from shape_extensions.dsl import Error, shape_dsl_function, ShapedArray, symint

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

@shape_dsl_function
def matmul_2d_ir(a: ShapedArray, b: ShapedArray) -> ShapedArray:
    if len(a.shape) != 2 or len(b.shape) != 2:
        raise Error("matmul expects 2-D arrays")
    if (
        isinstance(a.shape[1], int)
        and isinstance(b.shape[0], int)
        and a.shape[1] != b.shape[0]
    ):
        raise Error("matmul inner dimensions must match")
    return ShapedArray(shape=[a.shape[0], b.shape[1]])

@shape_dsl_function
def normalize_axis(rank: int, axis: int) -> int:
    if axis < 0:
        return axis + rank
    return axis

@shape_dsl_function
def count_axis(axes: list[int], axis: int) -> int:
    return len([candidate for candidate in axes if candidate == axis])

@shape_dsl_function
def reduce_shape(
    shape: list[int | symint],
    axis: int | list[int] | None,
    keepdims: bool,
) -> list[int | symint]:
    if axis == None:
        if keepdims:
            return [1 for _ in range(len(shape))]
        return []
    axes = axis if isinstance(axis, list) else [axis]
    normalized = [normalize_axis(len(shape), axis) for axis in axes]
    out_of_bounds = [axis for axis in normalized if axis < 0 or axis > len(shape) - 1]
    if len(out_of_bounds) > 0:
        raise Error("axis out of bounds")
    duplicate_axes = [axis for axis in normalized if count_axis(normalized, axis) > 1]
    if len(duplicate_axes) > 0:
        raise Error("duplicate axis")
    return [
        1 if i in normalized else dim
        for i, dim in enumerate(shape)
        if keepdims or not (i in normalized)
    ]

@shape_dsl_function
def reduce_ir(
    a: ShapedArray,
    axis: int | list[int] | None = None,
    keepdims: bool = False,
) -> ShapedArray:
    return ShapedArray(shape=reduce_shape(a.shape, axis, keepdims))
