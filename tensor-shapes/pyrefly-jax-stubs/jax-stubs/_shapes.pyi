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

@type_shape_dsl_function
def einsum_shape(spec: str, shapes: IntTuples) -> IntTuple:
    return dsl.einsum(spec, shapes)

@type_shape_dsl_function
def dot_shape(left: IntTuple, right: IntTuple) -> IntTuple:
    if len(left) == 0:
        return right
    if len(right) == 0:
        return left
    if len(left) == 1 and len(right) == 1:
        if left[0] != right[0]:
            return dsl.Invalid("dot dimensions must match")
        return dsl.IntTuple(())
    if len(right) == 1:
        if left[len(left) - 1] != right[0]:
            return dsl.Invalid("dot inner dimensions must match")
        return left[: len(left) - 1]
    if len(left) == 1:
        if left[0] != right[len(right) - 2]:
            return dsl.Invalid("dot inner dimensions must match")
        return dsl.concat(
            right[: len(right) - 2], dsl.IntTuple((right[len(right) - 1],))
        )
    if left[len(left) - 1] != right[len(right) - 2]:
        return dsl.Invalid("dot inner dimensions must match")
    return dsl.concat(
        left[: len(left) - 1],
        dsl.concat(right[: len(right) - 2], dsl.IntTuple((right[len(right) - 1],))),
    )

@type_shape_dsl_function
def inner_shape(left: IntTuple, right: IntTuple) -> IntTuple:
    if len(left) == 0:
        return right
    if len(right) == 0:
        return left
    if left[len(left) - 1] != right[len(right) - 1]:
        return dsl.Invalid("inner dimensions must match")
    if len(left) == 1 and len(right) == 1:
        return dsl.IntTuple(())
    return dsl.concat(left[: len(left) - 1], right[: len(right) - 1])

@type_shape_dsl_function
def kron_shape(a_shape: IntTuple, b_shape: IntTuple) -> IntTuple:
    if len(a_shape) == 0:
        return b_shape
    if len(b_shape) == 0:
        return a_shape
    if len(a_shape) >= len(b_shape):
        diff = len(a_shape) - len(b_shape)
        return dsl.IntTuple(
            a_shape[i] * (1 if i < diff else b_shape[i - diff])
            for i in range(len(a_shape))
        )
    diff_pos = len(b_shape) - len(a_shape)
    return dsl.IntTuple(
        (1 if i < diff_pos else a_shape[i - diff_pos]) * b_shape[i]
        for i in range(len(b_shape))
    )

@type_shape_dsl_function
def matvec_shape(left: IntTuple, right: IntTuple) -> IntTuple:
    if len(left) < 2 or len(right) < 1:
        return dsl.Invalid("matvec requires at least 2-D matrix and 1-D vector")
    operands = dsl.IntTuples((left, right))
    spec = "(m,n),(n)->(m)"
    return gufunc_broadcast(spec, operands)

@type_shape_dsl_function
def vecmat_shape(left: IntTuple, right: IntTuple) -> IntTuple:
    if len(left) < 1 or len(right) < 2:
        return dsl.Invalid("vecmat requires at least 1-D vector and 2-D matrix")
    operands = dsl.IntTuples((left, right))
    spec = "(n),(n,m)->(m)"
    return gufunc_broadcast(spec, operands)

@type_shape_dsl_function
def tensordot_shape(left: IntTuple, right: IntTuple, dims: int) -> IntTuple:
    if dims < 0:
        return dsl.Invalid("tensordot dims must be non-negative")
    if dims > len(left) or dims > len(right):
        return dsl.Invalid("tensordot dims exceeds input rank")
    if any(left[len(left) - dims + i] != right[i] for i in range(dims)):
        return dsl.Invalid("tensordot contracted dimensions must match")
    return dsl.concat(left[: len(left) - dims], right[dims:])

@type_shape_dsl_function
def diagonal_shape(shape: IntTuple, offset: int, axis1: int, axis2: int) -> IntTuple:
    rank = len(shape)
    if rank < 2:
        return dsl.Invalid("diagonal requires at least 2-D array")

    if axis1 < 0:
        norm_axis1 = axis1 + rank
    else:
        norm_axis1 = axis1 + 0
    if norm_axis1 < 0 or norm_axis1 >= rank:
        return dsl.Invalid("axis1 out of bounds")

    if axis2 < 0:
        norm_axis2 = axis2 + rank
    else:
        norm_axis2 = axis2 + 0
    if norm_axis2 < 0 or norm_axis2 >= rank:
        return dsl.Invalid("axis2 out of bounds")

    if norm_axis1 == norm_axis2:
        return dsl.Invalid("axis1 and axis2 cannot be the same")

    d1 = shape[norm_axis1]
    d2 = shape[norm_axis2]

    zero_tuple = dsl.IntTuple((0,))
    zero = zero_tuple[0]
    offset_tuple = dsl.IntTuple((offset + 0,))
    offset_dim = offset_tuple[0]

    remaining = dsl.IntTuple(
        (shape[i] for i in range(rank) if i != norm_axis1 and i != norm_axis2)
    )

    if offset == 0:
        if d1 == d2:
            return dsl.concat(remaining, dsl.IntTuple((d1,)))
        if dsl.is_concrete_int(d1) and dsl.is_concrete_int(d2):
            if d1 < d2:
                return dsl.concat(remaining, dsl.IntTuple((d1,)))
            return dsl.concat(remaining, dsl.IntTuple((d2,)))
        return dsl.concat(remaining, dsl.IntTuple((dsl.Int.gradual(),)))

    if offset > 0:
        limit = d2 - offset_dim
        if d1 == limit:
            return dsl.concat(remaining, dsl.IntTuple((d1,)))
        if dsl.is_concrete_int(d1) and dsl.is_concrete_int(limit):
            if limit < zero:
                return dsl.concat(remaining, dsl.IntTuple((zero,)))
            if d1 < limit:
                return dsl.concat(remaining, dsl.IntTuple((d1,)))
            return dsl.concat(remaining, dsl.IntTuple((limit,)))
        return dsl.concat(remaining, dsl.IntTuple((dsl.Int.gradual(),)))

    limit = d1 + offset_dim
    if limit == d2:
        return dsl.concat(remaining, dsl.IntTuple((d2,)))
    if dsl.is_concrete_int(limit) and dsl.is_concrete_int(d2):
        if limit < zero:
            return dsl.concat(remaining, dsl.IntTuple((zero,)))
        if limit < d2:
            return dsl.concat(remaining, dsl.IntTuple((limit,)))
        return dsl.concat(remaining, dsl.IntTuple((d2,)))
    return dsl.concat(remaining, dsl.IntTuple((dsl.Int.gradual(),)))

@type_shape_dsl_function
def trace_shape(shape: IntTuple, offset: int, axis1: int, axis2: int) -> IntTuple:
    rank = len(shape)
    if rank < 2:
        return dsl.Invalid("trace requires at least 2-D array")

    if axis1 < 0:
        norm_axis1 = axis1 + rank
    else:
        norm_axis1 = axis1 + 0
    if norm_axis1 < 0 or norm_axis1 >= rank:
        return dsl.Invalid("axis1 out of bounds")

    if axis2 < 0:
        norm_axis2 = axis2 + rank
    else:
        norm_axis2 = axis2 + 0
    if norm_axis2 < 0 or norm_axis2 >= rank:
        return dsl.Invalid("axis2 out of bounds")

    if norm_axis1 == norm_axis2:
        return dsl.Invalid("axis1 and axis2 cannot be the same")

    if rank == 2:
        return dsl.IntTuple(())

    return dsl.IntTuple(
        (shape[i] for i in range(rank) if i != norm_axis1 and i != norm_axis2)
    )

@type_shape_dsl_function
def cross_axes_shape(
    a_shape: IntTuple,
    b_shape: IntTuple,
    axisa: int,
    axisb: int,
    axisc: int,
) -> IntTuple:
    rank_a = len(a_shape)
    rank_b = len(b_shape)
    if rank_a == 0 or rank_b == 0:
        return dsl.Invalid("cross requires at least 1-D arrays")

    if axisa < 0:
        norm_axisa = axisa + rank_a
    else:
        norm_axisa = axisa + 0
    if norm_axisa < 0 or norm_axisa >= rank_a:
        return dsl.Invalid("axisa out of bounds")

    if axisb < 0:
        norm_axisb = axisb + rank_b
    else:
        norm_axisb = axisb + 0
    if norm_axisb < 0 or norm_axisb >= rank_b:
        return dsl.Invalid("axisb out of bounds")

    dim_a = a_shape[norm_axisa]
    dim_b = b_shape[norm_axisb]
    if dsl.is_concrete_int(dim_a) and dim_a != 2 and dim_a != 3:
        return dsl.Invalid("Dimension must be either 2 or 3 for cross product")
    if dsl.is_concrete_int(dim_b) and dim_b != 2 and dim_b != 3:
        return dsl.Invalid("Dimension must be either 2 or 3 for cross product")

    batch_a = dsl.concat(a_shape[:norm_axisa], a_shape[norm_axisa + 1 :])
    batch_b = dsl.concat(b_shape[:norm_axisb], b_shape[norm_axisb + 1 :])
    len_a = len(batch_a)
    len_b = len(batch_b)
    if len_a >= len_b:
        diff_a = len_a - len_b
        batch = dsl.IntTuple(
            (
                (batch_b[i - diff_a] if batch_a[i] == 1 else batch_a[i])
                if i >= diff_a
                else batch_a[i]
            )
            for i in range(len_a)
        )
    else:
        diff_b = len_b - len_a
        batch = dsl.IntTuple(
            (
                (batch_a[i - diff_b] if batch_b[i] == 1 else batch_b[i])
                if i >= diff_b
                else batch_b[i]
            )
            for i in range(len_b)
        )

    if dim_a == 2 and dim_b == 2:
        return batch

    if dim_a == 3 or dim_b == 3:
        out_rank = len(batch) + 1
        if axisc < 0:
            norm_axisc = axisc + out_rank
        else:
            norm_axisc = axisc + 0
        if norm_axisc < 0 or norm_axisc >= out_rank:
            return dsl.Invalid("axisc out of bounds")
        return dsl.concat(
            dsl.concat(batch[:norm_axisc], dsl.IntTuple((3,))),
            batch[norm_axisc:],
        )

    return dsl.IntTuple.gradual()

@type_shape_dsl_function
def cross_axis_shape(
    a_shape: IntTuple,
    b_shape: IntTuple,
    axis: int,
) -> IntTuple:
    return cross_axes_shape(a_shape, b_shape, axis, axis, axis)

@type_shape_dsl_function
def expand_dims_shape(shape: IntTuple, axis: int) -> IntTuple:
    out_rank = len(shape) + 1
    if axis < 0 - out_rank or axis >= out_rank:
        return dsl.Invalid("axis out of bounds")
    if axis < 0:
        norm_axis = axis + out_rank
    else:
        norm_axis = axis + 0
    return dsl.concat(
        dsl.concat(shape[:norm_axis], dsl.IntTuple((1,))),
        shape[norm_axis:],
    )

@type_shape_dsl_function
def squeeze_shape(shape: IntTuple, axis: int | tuple[int, ...] | None) -> IntTuple:
    if axis is None:
        return dsl.IntTuple(
            (shape[index] for index in range(len(shape)) if shape[index] != 1)
        )
    if dsl.is_int_value(axis):
        if len(shape) == 0:
            if axis == 0 or axis == -1:
                return shape
            return dsl.Invalid("squeeze axis out of range")
        if axis < 0 - len(shape) or axis >= len(shape):
            return dsl.Invalid("squeeze axis out of range")
        if axis < 0:
            norm_axis = axis + len(shape)
        else:
            norm_axis = axis + 0
        dim_val = shape[norm_axis]
        if dsl.is_concrete_int(dim_val) and dim_val != 1:
            return dsl.Invalid(
                "cannot select an axis to squeeze out which has size not equal to one"
            )
        return dsl.concat(shape[:norm_axis], shape[norm_axis + 1 :])
    return dsl.IntTuple.gradual()

@type_shape_dsl_function
def concatenate_shape(shapes: IntTuples, axis: int) -> IntTuple:
    if len(shapes) == 0:
        return dsl.Invalid("need at least one array to concatenate")
    first = shapes[0]
    rank = len(first)
    if rank == 0:
        return dsl.Invalid("zero-dimensional arrays cannot be concatenated")
    if axis < 0 - rank or axis >= rank:
        return dsl.Invalid("axis out of bounds")
    if axis < 0:
        norm_axis = axis + rank
    else:
        norm_axis = axis + 0
    ranks = dsl.IntTuple((0 if len(shape) == rank else 1 for shape in shapes))
    if any(r == 1 for r in ranks):
        return dsl.Invalid("all input arrays must have the same number of dimensions")
    mismatches = dsl.IntTuple(
        (
            1
            if any(
                shape[index] != first[index]
                for index in range(rank)
                if index != norm_axis
            )
            else 0
            for shape in shapes
        )
    )
    if any(m == 1 for m in mismatches):
        return dsl.Invalid(
            "all input array dimensions for the concatenation axis must match exactly"
        )
    return dsl.IntTuple(
        (
            dsl.sum(dsl.IntTuple((shape[norm_axis] for shape in shapes)))
            if index == norm_axis
            else first[index]
            for index in range(rank)
        )
    )

@type_shape_dsl_function
def stack_shape(shapes: IntTuples, axis: int) -> IntTuple:
    if len(shapes) == 0:
        return dsl.Invalid("need at least one array to stack")
    first = shapes[0]
    out_rank = len(first) + 1
    if axis < 0 - out_rank or axis >= out_rank:
        return dsl.Invalid("axis out of bounds")
    if axis < 0:
        norm_axis = axis + out_rank
    else:
        norm_axis = axis + 0
    ranks = dsl.IntTuple((0 if len(shape) == len(first) else 1 for shape in shapes))
    if any(r == 1 for r in ranks):
        return dsl.Invalid("all input arrays must have the same number of dimensions")
    mismatches = dsl.IntTuple(
        (
            1 if any(shape[index] != first[index] for index in range(len(first))) else 0
            for shape in shapes
        )
    )
    if any(m == 1 for m in mismatches):
        return dsl.Invalid("all input arrays must have the same shape")
    return dsl.IntTuple(
        (
            len(shapes)
            if index == norm_axis
            else first[index if index < norm_axis else index - 1]
            for index in range(out_rank)
        )
    )

@type_shape_dsl_function
def vstack_shape(shapes: IntTuples) -> IntTuple:
    if len(shapes) == 0:
        return dsl.Invalid("need at least one array to vstack")
    first = shapes[0]
    if len(first) == 0:
        return dsl.Invalid("zero-dimensional arrays cannot be concatenated")
    zero = 0
    if len(first) == 1:
        return stack_shape(shapes, zero)
    return concatenate_shape(shapes, zero)

@type_shape_dsl_function
def hstack_shape(shapes: IntTuples) -> IntTuple:
    if len(shapes) == 0:
        return dsl.Invalid("need at least one array to hstack")
    first = shapes[0]
    if len(first) == 0:
        return dsl.Invalid("zero-dimensional arrays cannot be concatenated")
    zero = 0
    if len(first) == 1:
        return concatenate_shape(shapes, zero)
    one = 1
    return concatenate_shape(shapes, one)

@type_shape_dsl_function
def broadcast_to_shape(
    shape: IntTuple, to_shape: int | tuple[int, ...] | None
) -> IntTuple:
    if to_shape is None:
        return dsl.Invalid("broadcast_to requires a shape")
    if dsl.is_int_value(to_shape):
        target = (to_shape,)
    else:
        target = to_shape
    if any(dim < 0 for dim in target):
        return dsl.Invalid("all elements of target shape must be non-negative")
    if len(shape) > len(target):
        return dsl.Invalid("incompatible shapes for broadcasting")
    target_tuple = dsl.IntTuple(dim for dim in target)
    operands = dsl.IntTuples((shape, target_tuple))
    spec = "(),()->()"
    return gufunc_broadcast(spec, operands)

@type_shape_dsl_function
def ravel_shape(shape: IntTuple) -> IntTuple:
    return dsl.IntTuple((dsl.prod(shape),))

@type_shape_dsl_function
def swapaxes_shape(shape: IntTuple, axis1: int, axis2: int) -> IntTuple:
    rank = len(shape)
    if rank == 0:
        if (axis1 == 0 or axis1 == -1) and (axis2 == 0 or axis2 == -1):
            return shape
        return dsl.Invalid("axis out of bounds")
    if axis1 < 0 - rank or axis1 >= rank or axis2 < 0 - rank or axis2 >= rank:
        return dsl.Invalid("axis out of bounds")
    if axis1 < 0:
        norm_axis1 = axis1 + rank
    else:
        norm_axis1 = axis1 + 0
    if axis2 < 0:
        norm_axis2 = axis2 + rank
    else:
        norm_axis2 = axis2 + 0
    if norm_axis1 == norm_axis2:
        return shape
    return dsl.IntTuple(
        (
            shape[norm_axis2]
            if index == norm_axis1
            else (shape[norm_axis1] if index == norm_axis2 else shape[index])
            for index in range(rank)
        )
    )

@type_shape_dsl_function
def moveaxis_shape(shape: IntTuple, source: int, destination: int) -> IntTuple:
    rank = len(shape)
    if rank == 0:
        return shape
    if source < 0 - rank or source >= rank:
        return dsl.Invalid("source axis out of bounds")
    if source < 0:
        norm_source = source + rank
    else:
        norm_source = source + 0
    if destination < 0 - rank or destination >= rank:
        return dsl.Invalid("destination axis out of bounds")
    if destination < 0:
        norm_dest = destination + rank
    else:
        norm_dest = destination + 0
    dim = shape[norm_source]
    remaining = dsl.concat(shape[:norm_source], shape[norm_source + 1 :])
    return dsl.concat(
        dsl.concat(remaining[:norm_dest], dsl.IntTuple((dim,))),
        remaining[norm_dest:],
    )

@type_shape_dsl_function
def rollaxis_shape(shape: IntTuple, axis: int, start: int) -> IntTuple:
    rank = len(shape)
    if rank == 0:
        return shape
    if axis < 0 - rank or axis >= rank:
        return dsl.Invalid("axis out of bounds")
    if axis < 0:
        norm_axis = axis + rank
    else:
        norm_axis = axis + 0
    if start < 0 - rank or start > rank:
        return dsl.Invalid("start out of bounds")
    if start < 0:
        norm_start = start + rank
    else:
        norm_start = start + 0
    if norm_axis < norm_start:
        target_pos = norm_start - 1
    else:
        target_pos = norm_start + 0
    dim = shape[norm_axis]
    remaining = dsl.concat(shape[:norm_axis], shape[norm_axis + 1 :])
    return dsl.concat(
        dsl.concat(remaining[:target_pos], dsl.IntTuple((dim,))),
        remaining[target_pos:],
    )

@type_shape_dsl_function
def column_stack_shape(shapes: IntTuples) -> IntTuple:
    if len(shapes) == 0:
        return dsl.Invalid("need at least one array to column_stack")
    first = shapes[0]
    if len(first) == 0:
        return dsl.Invalid("zero-dimensional arrays cannot be concatenated")
    one = 1
    if len(first) == 1:
        return stack_shape(shapes, one)
    return concatenate_shape(shapes, one)

@type_shape_dsl_function
def dstack_shape(shapes: IntTuples) -> IntTuple:
    if len(shapes) == 0:
        return dsl.Invalid("need at least one array to dstack")
    first = shapes[0]
    if len(first) == 0:
        return dsl.Invalid("zero-dimensional arrays cannot be concatenated")
    if len(first) == 1:
        ranks = dsl.IntTuple((0 if len(shape) == 1 else 1 for shape in shapes))
        if any(r == 1 for r in ranks):
            return dsl.Invalid(
                "all input arrays must have the same number of dimensions"
            )
        mismatches = dsl.IntTuple(
            (1 if shape[0] != first[0] else 0 for shape in shapes)
        )
        if any(m == 1 for m in mismatches):
            return dsl.Invalid("all input array dimensions must match exactly")
        return dsl.IntTuple((1, first[0], len(shapes)))
    two = 2
    if len(first) == 2:
        return stack_shape(shapes, two)
    return concatenate_shape(shapes, two)

@type_shape_dsl_function
def flip_shape(shape: IntTuple, axis: int | tuple[int, ...] | None) -> IntTuple:
    if axis is None:
        return shape
    if dsl.is_int_value(axis):
        axes = (axis,)
    else:
        axes = axis
    rank = len(shape)
    if any(item < 0 - rank or item >= rank for item in axes):
        return dsl.Invalid("axis out of bounds")
    return shape

@type_shape_dsl_function
def roll_shape(shape: IntTuple, axis: int | tuple[int, ...] | None) -> IntTuple:
    if axis is None:
        return shape
    if dsl.is_int_value(axis):
        axes = (axis,)
    else:
        axes = axis
    rank = len(shape)
    if any(item < 0 - rank or item >= rank for item in axes):
        return dsl.Invalid("axis out of bounds")
    return shape

@type_shape_dsl_function
def atleast_1d_shape(shape: IntTuple) -> IntTuple:
    if len(shape) == 0:
        return dsl.IntTuple((1,))
    return shape

@type_shape_dsl_function
def atleast_2d_shape(shape: IntTuple) -> IntTuple:
    if len(shape) == 0:
        return dsl.IntTuple((1, 1))
    if len(shape) == 1:
        return dsl.concat(dsl.IntTuple((1,)), shape)
    return shape

@type_shape_dsl_function
def atleast_3d_shape(shape: IntTuple) -> IntTuple:
    if len(shape) == 0:
        return dsl.IntTuple((1, 1, 1))
    if len(shape) == 1:
        return dsl.concat(dsl.IntTuple((1,)), dsl.concat(shape, dsl.IntTuple((1,))))
    if len(shape) == 2:
        return dsl.concat(shape, dsl.IntTuple((1,)))
    return shape

@type_shape_dsl_function
def broadcast_to_rank_shape(shape: IntTuple, rank: int) -> IntTuple:
    if rank < len(shape):
        return dsl.Invalid("rank must be greater than or equal to array rank")
    diff = rank - len(shape)
    ones = dsl.IntTuple(1 for i in range(diff))
    return dsl.concat(ones, shape)

@type_shape_dsl_function
def collapse_to_end_shape(shape: IntTuple, start: int) -> IntTuple:
    rank = len(shape)
    if start < 0:
        norm_start = start + rank
    else:
        norm_start = start + 0
    if norm_start < 0 or norm_start > rank:
        return dsl.Invalid("start_dimension out of bounds")
    collapsed_dim = dsl.prod(shape[norm_start:])
    return dsl.concat(shape[:norm_start], dsl.IntTuple((collapsed_dim,)))

@type_shape_dsl_function
def collapse_shape(shape: IntTuple, start: int, stop: int) -> IntTuple:
    rank = len(shape)
    if start < 0:
        norm_start = start + rank
    else:
        norm_start = start + 0
    if stop < 0:
        norm_stop = stop + rank
    else:
        norm_stop = stop + 0
    if norm_start < 0 or norm_start > rank:
        return dsl.Invalid("start_dimension out of bounds")
    if norm_stop < 0 or norm_stop > rank:
        return dsl.Invalid("stop_dimension out of bounds")
    if norm_stop < norm_start:
        return dsl.Invalid("stop_dimension must be >= start_dimension")
    collapsed_dim = dsl.prod(shape[norm_start:norm_stop])
    return dsl.concat(
        dsl.concat(shape[:norm_start], dsl.IntTuple((collapsed_dim,))),
        shape[norm_stop:],
    )

@type_shape_dsl_function
def lax_squeeze_shape(shape: IntTuple, dimensions: tuple[int, ...]) -> IntTuple:
    rank = len(shape)
    if any(d < 0 or d >= rank for d in dimensions):
        return dsl.Invalid("squeeze axis out of bounds")
    if any(dimensions.count(d) > 1 for d in dimensions):
        return dsl.Invalid("repeated axis in lax.squeeze")
    if any(shape[d] != 1 for d in dimensions):
        return dsl.Invalid(
            "cannot select an axis to squeeze out which has size not equal to one"
        )
    return dsl.IntTuple((shape[i] for i in range(rank) if i not in dimensions))

@type_shape_dsl_function
def sort_shape(shape: IntTuple, axis: int | None) -> IntTuple:
    if axis is None:
        return dsl.IntTuple((dsl.prod(shape),))
    if dsl.is_int_value(axis):
        if len(shape) == 0:
            return dsl.Invalid("axis out of bounds")
        if axis < 0 - len(shape) or axis >= len(shape):
            return dsl.Invalid("axis out of bounds")
        return shape
    return dsl.Invalid("axis must be an integer or None")

@type_shape_dsl_function
def top_k_shape(shape: IntTuple, k: int, axis: int) -> IntTuple:
    rank = len(shape)
    if rank == 0:
        return dsl.Invalid("axis out of bounds")
    if axis < 0:
        norm_axis = axis + rank
    else:
        norm_axis = axis + 0
    if norm_axis < 0 or norm_axis >= rank:
        return dsl.Invalid("axis out of bounds")
    if k < 0:
        return dsl.Invalid("k must be non-negative")
    extent = k + 0
    return dsl.concat(
        dsl.concat(shape[:norm_axis], dsl.IntTuple((extent,))),
        shape[norm_axis + 1 :],
    )
