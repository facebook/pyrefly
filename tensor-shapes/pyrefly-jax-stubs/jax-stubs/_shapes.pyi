# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import shape_extensions.dsl as dsl
from shape_extensions import (
    broadcast,
    gufunc_broadcast,
    Int,
    IntTuple,
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
    known_shape = dsl.IntTuple((dim for dim in dims if dim != 0 - 1))
    known = dsl.prod(known_shape)
    total = dsl.prod(shape)
    if len(inferred) == 0:
        if dsl.is_concrete_int(total) and dsl.is_concrete_int(known) and total != known:
            return dsl.Invalid("reshape target element count does not match the input")
        return dsl.IntTuple(dim for dim in dims)
    if dsl.is_concrete_int(known):
        if known == 0:
            return dsl.Invalid("could not infer size for dimension -1")
        if dsl.is_concrete_int(total) and total % known != 0:
            return dsl.Invalid("could not infer size for dimension -1")
    return dsl.IntTuple((total // known if dim == 0 - 1 else dim for dim in dims))

@type_shape_dsl_function
def fft_n_shape(shape: IntTuple, n: int, dim: int) -> IntTuple:
    rank = len(shape)
    if rank == 0:
        return dsl.Invalid("FFT requires at least 1-D array")
    if n < 0:
        return dsl.Invalid("n must be non-negative")
    if dim < 0:
        axis = dim + rank
    else:
        axis = dim + 0
    if axis < 0 or axis >= rank:
        return dsl.Invalid("FFT axis out of bounds")
    extent = n + 0
    return dsl.concat(
        dsl.concat(shape[:axis], dsl.IntTuple((extent,))), shape[axis + 1 :]
    )

@type_shape_dsl_function
def rfft_shape(shape: IntTuple, dim: int) -> IntTuple:
    rank = len(shape)
    if rank == 0:
        return dsl.Invalid("FFT requires at least 1-D array")
    if dim < 0:
        axis = dim + rank
    else:
        axis = dim + 0
    if axis < 0 or axis >= rank:
        return dsl.Invalid("FFT axis out of bounds")
    extent = shape[axis] // 2 + 1
    return dsl.concat(
        dsl.concat(shape[:axis], dsl.IntTuple((extent,))), shape[axis + 1 :]
    )

@type_shape_dsl_function
def rfft_n_shape(shape: IntTuple, n: int, dim: int) -> IntTuple:
    rank = len(shape)
    if rank == 0:
        return dsl.Invalid("FFT requires at least 1-D array")
    if n < 0:
        return dsl.Invalid("n must be non-negative")
    if dim < 0:
        axis = dim + rank
    else:
        axis = dim + 0
    if axis < 0 or axis >= rank:
        return dsl.Invalid("FFT axis out of bounds")
    extent = n // 2 + 1
    return dsl.concat(
        dsl.concat(shape[:axis], dsl.IntTuple((extent,))), shape[axis + 1 :]
    )

@type_shape_dsl_function
def irfft_shape(shape: IntTuple, dim: int) -> IntTuple:
    rank = len(shape)
    if rank == 0:
        return dsl.Invalid("FFT requires at least 1-D array")
    if dim < 0:
        axis = dim + rank
    else:
        axis = dim + 0
    if axis < 0 or axis >= rank:
        return dsl.Invalid("FFT axis out of bounds")
    extent = 2 * (shape[axis] - 1)
    return dsl.concat(
        dsl.concat(shape[:axis], dsl.IntTuple((extent,))), shape[axis + 1 :]
    )

@type_shape_dsl_function
def irfft_n_shape(shape: IntTuple, n: int, dim: int) -> IntTuple:
    rank = len(shape)
    if rank == 0:
        return dsl.Invalid("FFT requires at least 1-D array")
    if n < 0:
        return dsl.Invalid("n must be non-negative")
    if dim < 0:
        axis = dim + rank
    else:
        axis = dim + 0
    if axis < 0 or axis >= rank:
        return dsl.Invalid("FFT axis out of bounds")
    extent = n + 0
    return dsl.concat(
        dsl.concat(shape[:axis], dsl.IntTuple((extent,))), shape[axis + 1 :]
    )

@type_shape_dsl_function
def fftfreq_shape(n: int) -> IntTuple:
    if n < 0:
        return dsl.Invalid("n must be non-negative")
    extent = n + 0
    return dsl.IntTuple((extent,))

@type_shape_dsl_function
def rfftfreq_shape(n: int) -> IntTuple:
    if n < 0:
        return dsl.Invalid("n must be non-negative")
    return dsl.IntTuple((n // 2 + 1,))

@type_shape_dsl_function
def lax_broadcast(left: IntTuple, right: IntTuple) -> IntTuple:
    if len(left) != 0 and len(right) != 0 and len(left) != len(right):
        return dsl.Invalid("arrays must have the same number of dimensions")
    return broadcast(left, right)

@type_shape_dsl_function
def symmetric_product_shape(a_shape: IntTuple, c_shape: IntTuple) -> IntTuple:
    if len(a_shape) < 2 or len(c_shape) < 2:
        return dsl.Invalid("symmetric_product requires at least 2-D arrays")
    if len(a_shape) != len(c_shape):
        return dsl.Invalid("arrays must have the same number of batch dimensions")
    operands = dsl.IntTuples((a_shape, c_shape))
    spec = "(m,n),(m,m)->(m,m)"
    return gufunc_broadcast(spec, operands)

@type_shape_dsl_function
def triangular_solve_shape(
    a_shape: IntTuple, b_shape: IntTuple, left_side: bool
) -> IntTuple:
    if len(a_shape) < 2 or len(b_shape) < 2:
        return dsl.Invalid("triangular_solve requires at least 2-D arrays")
    if len(a_shape) != len(b_shape):
        return dsl.Invalid("arrays must have the same number of batch dimensions")
    operands = dsl.IntTuples((a_shape, b_shape))
    if left_side:
        spec = "(m,m),(m,n)->(m,n)"
    else:
        spec = "(n,n),(m,n)->(m,n)"
    return gufunc_broadcast(spec, operands)

@type_shape_dsl_function
def cholesky_update_shape(r_shape: IntTuple, w_shape: IntTuple) -> IntTuple:
    if len(r_shape) < 2 or len(w_shape) < 1:
        return dsl.Invalid(
            "cholesky_update requires at least 2-D matrix and 1-D vector"
        )
    if len(r_shape) - 2 != len(w_shape) - 1:
        return dsl.Invalid("arrays must have the same number of batch dimensions")
    operands = dsl.IntTuples((r_shape, w_shape))
    spec = "(n,n),(n)->(n,n)"
    return gufunc_broadcast(spec, operands)

@type_shape_dsl_function
def householder_product_shape(a_shape: IntTuple, taus_shape: IntTuple) -> IntTuple:
    if len(a_shape) < 2 or len(taus_shape) < 1:
        return dsl.Invalid(
            "householder_product requires at least 2-D matrix and 1-D taus"
        )
    if len(a_shape) - 2 != len(taus_shape) - 1:
        return dsl.Invalid("arrays must have the same number of batch dimensions")
    operands = dsl.IntTuples((a_shape, taus_shape))
    spec = "(m,n),(k)->(m,n)"
    return gufunc_broadcast(spec, operands)

@type_shape_dsl_function
def ormqr_shape(a_shape: IntTuple, taus_shape: IntTuple, c_shape: IntTuple) -> IntTuple:
    if len(a_shape) < 2 or len(taus_shape) < 1 or len(c_shape) < 2:
        return dsl.Invalid("ormqr requires at least 2-D arrays")
    batch_c = c_shape[: len(c_shape) - 2]
    m = c_shape[len(c_shape) - 2]
    n = c_shape[len(c_shape) - 1]
    if len(batch_c) == 0:
        return dsl.IntTuple((m, n))
    return dsl.concat(batch_c, dsl.IntTuple((m, n)))

@type_shape_dsl_function
def tridiagonal_solve_shape(
    dl_shape: IntTuple,
    d_shape: IntTuple,
    du_shape: IntTuple,
    b_shape: IntTuple,
) -> IntTuple:
    if len(dl_shape) < 1 or len(d_shape) < 1 or len(du_shape) < 1 or len(b_shape) < 2:
        return dsl.Invalid(
            "tridiagonal_solve requires at least 1-D diagonals and 2-D b"
        )
    b_rank = len(b_shape) - 2
    if (
        len(dl_shape) - 1 != b_rank
        or len(d_shape) - 1 != b_rank
        or len(du_shape) - 1 != b_rank
    ):
        return dsl.Invalid("arrays must have the same number of batch dimensions")
    operands = dsl.IntTuples((dl_shape, d_shape, du_shape, b_shape))
    spec = "(n),(n),(n),(n,k)->(n,k)"
    return gufunc_broadcast(spec, operands)

@type_shape_dsl_function
def hessenberg_taus_shape(a_shape: IntTuple) -> IntTuple:
    if len(a_shape) < 2:
        return dsl.Invalid("hessenberg requires at least 2-D array")
    n1 = a_shape[len(a_shape) - 2]
    n2 = a_shape[len(a_shape) - 1]
    if n1 != n2:
        return dsl.Invalid("hessenberg requires a square matrix")
    batch = a_shape[: len(a_shape) - 2]
    return dsl.concat(batch, dsl.IntTuple((n1 - 1,)))

@type_shape_dsl_function
def tridiagonal_d_shape(a_shape: IntTuple) -> IntTuple:
    if len(a_shape) < 2:
        return dsl.Invalid("tridiagonal requires at least 2-D array")
    n1 = a_shape[len(a_shape) - 2]
    n2 = a_shape[len(a_shape) - 1]
    if n1 != n2:
        return dsl.Invalid("tridiagonal requires a square matrix")
    batch = a_shape[: len(a_shape) - 2]
    return dsl.concat(batch, dsl.IntTuple((n1,)))

@type_shape_dsl_function
def tridiagonal_diag_minus_one_shape(a_shape: IntTuple) -> IntTuple:
    if len(a_shape) < 2:
        return dsl.Invalid("tridiagonal requires at least 2-D array")
    n1 = a_shape[len(a_shape) - 2]
    n2 = a_shape[len(a_shape) - 1]
    if n1 != n2:
        return dsl.Invalid("tridiagonal requires a square matrix")
    batch = a_shape[: len(a_shape) - 2]
    return dsl.concat(batch, dsl.IntTuple((n1 - 1,)))
