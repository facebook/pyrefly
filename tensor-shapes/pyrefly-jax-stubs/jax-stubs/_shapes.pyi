# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import shape_extensions.dsl as dsl
from shape_extensions import IntTuple, type_shape_dsl_function

@type_shape_dsl_function
def matmul_shape(left: IntTuple, right: IntTuple) -> IntTuple:
    # TODO(stroxler): Model batched matmul, which broadcasts the leading
    # dimensions. Batched calls are common in JAX, so an unmodeled rank yields a
    # gradual shape rather than an error: rejecting valid code would be worse
    # than not checking it.
    if len(left) > 2 or len(right) > 2 or len(left) == 0 or len(right) == 0:
        return dsl.IntTuple.gradual()
    # The contracted dimension is the last of the left operand and the first of
    # the right, whatever the ranks. A 1-D operand contributes no axis to the
    # result, so `(N, M) @ (M,)` is `(N,)` and `(M,) @ (M,)` is a scalar.
    left_inner = left[len(left) - 1]
    right_inner = right[0]
    if (
        dsl.is_concrete_int(left_inner)
        and dsl.is_concrete_int(right_inner)
        and left_inner != right_inner
    ):
        return dsl.Invalid("matmul inner dimensions must match")
    if len(left) == 1 and len(right) == 1:
        return dsl.IntTuple(())
    if len(left) == 1:
        return dsl.IntTuple((right[1],))
    if len(right) == 1:
        return dsl.IntTuple((left[0],))
    return dsl.IntTuple((left[0], right[1]))

@type_shape_dsl_function
def reverse_shape(shape: IntTuple) -> IntTuple:
    # The default transpose: every axis in reverse order, at any rank.
    return dsl.IntTuple(shape[len(shape) - index - 1] for index in range(len(shape)))

@type_shape_dsl_function
def permute_shape(shape: IntTuple, axes: int | tuple[int, ...] | None) -> IntTuple:
    # `None` means the default reversal, which `reverse_shape` already covers, so
    # it never reaches here. The arm exists because the DSL recognizes only
    # `int | tuple[int, ...] | None` as an integer-or-tuple parameter domain.
    if axes is None:
        return dsl.IntTuple.gradual()
    if dsl.is_int_value(axes):
        return dsl.Invalid("transpose axes must be a sequence")
    if len(axes) != len(shape):
        return dsl.Invalid("transpose axes must cover every dimension")
    # The DSL does not support unary negation of a Flag integer.
    if any(item < 0 - len(shape) or item >= len(shape) for item in axes):
        return dsl.Invalid("axis out of bounds")
    normalized = tuple(item + len(shape) if item < 0 else item for item in axes)
    if any(normalized.count(item) > 1 for item in normalized):
        return dsl.Invalid("duplicate axis")
    return dsl.IntTuple(shape[index] for index in normalized)

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
def reshape_shape(shape: IntTuple, newshape: int | tuple[int, ...] | None) -> IntTuple:
    # `None` is not a legal argument to `reshape`. The arm exists because an
    # `int | tuple[int, ...]` parameter cannot be iterated after narrowing with
    # `is_int_value` alone -- the DSL function silently evaluates to `Unknown`.
    # Leading with the `None` check is what makes the narrowing work. The Torch
    # stubs use the same workaround in `conv_shape`.
    if newshape is None:
        return dsl.Invalid("reshape requires a shape")
    if dsl.is_int_value(newshape):
        dims = (newshape,)
    else:
        dims = newshape
    # The DSL does not support unary negation of a Flag integer, so the
    # placeholder dimension is spelled `0 - 1` throughout.
    if any(dim < 0 - 1 for dim in dims):
        return dsl.Invalid("reshape sizes must be -1 or non-negative")
    inferred = tuple(dim for dim in dims if dim == 0 - 1)
    if len(inferred) > 1:
        return dsl.Invalid("reshape accepts at most one -1")
    # TODO(stroxler): Infer the placeholder dimension, and reject a reshape that
    # changes the number of elements. Both need the product of the dimensions,
    # which the type-level DSL does not expose yet; the in-flight Torch
    # migration adds `dsl.prod` as an intrinsic.
    if len(inferred) == 1:
        return dsl.IntTuple.gradual()
    return dsl.IntTuple(dim for dim in dims)
