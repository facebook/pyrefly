# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import shape_extensions.dsl as dsl
from shape_extensions import (
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
    if len(left) == 0:
        return right
    if len(right) == 0:
        return left
    if len(left) != len(right):
        return dsl.Invalid("arrays must have the same number of dimensions")
    if any(
        left[i] != right[i] and left[i] != 1 and right[i] != 1 for i in range(len(left))
    ):
        return dsl.Invalid("incompatible shapes for broadcasting")
    return dsl.IntTuple(
        (right[i] if left[i] == 1 else left[i]) for i in range(len(left))
    )

@type_shape_dsl_function
def symmetric_product_shape(a_shape: IntTuple, c_shape: IntTuple) -> IntTuple:
    if len(a_shape) < 2 or len(c_shape) < 2:
        return dsl.Invalid("symmetric_product requires at least 2-D arrays")
    m_a = a_shape[len(a_shape) - 2]
    m_c1 = c_shape[len(c_shape) - 2]
    m_c2 = c_shape[len(c_shape) - 1]
    if m_c1 != m_c2:
        return dsl.Invalid("c_matrix must be square")
    if m_a != m_c1:
        return dsl.Invalid(
            "leading core dimensions of a_matrix and c_matrix must match"
        )
    batch_a = a_shape[: len(a_shape) - 2]
    batch_c = c_shape[: len(c_shape) - 2]
    if len(batch_a) == 0 and len(batch_c) == 0:
        return dsl.IntTuple((m_a, m_a))
    if len(batch_a) == 0:
        return dsl.concat(batch_c, dsl.IntTuple((m_a, m_a)))
    if len(batch_c) == 0:
        return dsl.concat(batch_a, dsl.IntTuple((m_a, m_a)))
    if len(batch_a) != len(batch_c):
        return dsl.Invalid("arrays must have the same number of batch dimensions")
    if any(
        batch_a[i] != batch_c[i] and batch_a[i] != 1 and batch_c[i] != 1
        for i in range(len(batch_a))
    ):
        return dsl.Invalid("incompatible batch shapes for broadcasting")
    batch = dsl.IntTuple(
        (batch_c[i] if batch_a[i] == 1 else batch_a[i]) for i in range(len(batch_a))
    )
    return dsl.concat(batch, dsl.IntTuple((m_a, m_a)))

@type_shape_dsl_function
def triangular_solve_shape(
    a_shape: IntTuple, b_shape: IntTuple, left_side: bool
) -> IntTuple:
    if len(a_shape) < 2 or len(b_shape) < 2:
        return dsl.Invalid("triangular_solve requires at least 2-D arrays")
    m_a = a_shape[len(a_shape) - 2]
    n_a = a_shape[len(a_shape) - 1]
    if m_a != n_a:
        return dsl.Invalid("a matrix must be square")
    m_b = b_shape[len(b_shape) - 2]
    n_b = b_shape[len(b_shape) - 1]
    if left_side:
        if m_a != m_b:
            return dsl.Invalid("incompatible shapes for triangular_solve")
    else:
        if m_a != n_b:
            return dsl.Invalid("incompatible shapes for triangular_solve")
    batch_a = a_shape[: len(a_shape) - 2]
    batch_b = b_shape[: len(b_shape) - 2]
    if len(batch_a) == 0 and len(batch_b) == 0:
        return dsl.IntTuple((m_b, n_b))
    if len(batch_a) == 0:
        return dsl.concat(batch_b, dsl.IntTuple((m_b, n_b)))
    if len(batch_b) == 0:
        return dsl.concat(batch_a, dsl.IntTuple((m_b, n_b)))
    if len(batch_a) != len(batch_b):
        return dsl.Invalid("arrays must have the same number of batch dimensions")
    if any(
        batch_a[i] != batch_b[i] and batch_a[i] != 1 and batch_b[i] != 1
        for i in range(len(batch_a))
    ):
        return dsl.Invalid("incompatible batch shapes for broadcasting")
    batch = dsl.IntTuple(
        (batch_b[i] if batch_a[i] == 1 else batch_a[i]) for i in range(len(batch_a))
    )
    return dsl.concat(batch, dsl.IntTuple((m_b, n_b)))

@type_shape_dsl_function
def cholesky_update_shape(r_shape: IntTuple, w_shape: IntTuple) -> IntTuple:
    if len(r_shape) < 2 or len(w_shape) < 1:
        return dsl.Invalid(
            "cholesky_update requires at least 2-D matrix and 1-D vector"
        )
    n_r1 = r_shape[len(r_shape) - 2]
    n_r2 = r_shape[len(r_shape) - 1]
    if n_r1 != n_r2:
        return dsl.Invalid("r_matrix must be square")
    n_w = w_shape[len(w_shape) - 1]
    if n_r1 != n_w:
        return dsl.Invalid("r_matrix and w_vector dimensions must match")
    batch_r = r_shape[: len(r_shape) - 2]
    batch_w = w_shape[: len(w_shape) - 1]
    if len(batch_r) == 0 and len(batch_w) == 0:
        return dsl.IntTuple((n_r1, n_r1))
    if len(batch_r) == 0:
        return dsl.concat(batch_w, dsl.IntTuple((n_r1, n_r1)))
    if len(batch_w) == 0:
        return dsl.concat(batch_r, dsl.IntTuple((n_r1, n_r1)))
    if len(batch_r) != len(batch_w):
        return dsl.Invalid("arrays must have the same number of batch dimensions")
    if any(
        batch_r[i] != batch_w[i] and batch_r[i] != 1 and batch_w[i] != 1
        for i in range(len(batch_r))
    ):
        return dsl.Invalid("incompatible batch shapes for broadcasting")
    batch = dsl.IntTuple(
        (batch_w[i] if batch_r[i] == 1 else batch_r[i]) for i in range(len(batch_r))
    )
    return dsl.concat(batch, dsl.IntTuple((n_r1, n_r1)))

@type_shape_dsl_function
def householder_product_shape(a_shape: IntTuple, taus_shape: IntTuple) -> IntTuple:
    if len(a_shape) < 2 or len(taus_shape) < 1:
        return dsl.Invalid(
            "householder_product requires at least 2-D matrix and 1-D taus"
        )
    batch_a = a_shape[: len(a_shape) - 2]
    batch_t = taus_shape[: len(taus_shape) - 1]
    m = a_shape[len(a_shape) - 2]
    n = a_shape[len(a_shape) - 1]
    if len(batch_a) == 0 and len(batch_t) == 0:
        return dsl.IntTuple((m, n))
    if len(batch_a) == 0:
        return dsl.concat(batch_t, dsl.IntTuple((m, n)))
    if len(batch_t) == 0:
        return dsl.concat(batch_a, dsl.IntTuple((m, n)))
    if len(batch_a) != len(batch_t):
        return dsl.Invalid("arrays must have the same number of batch dimensions")
    if any(
        batch_a[i] != batch_t[i] and batch_a[i] != 1 and batch_t[i] != 1
        for i in range(len(batch_a))
    ):
        return dsl.Invalid("incompatible batch shapes for broadcasting")
    batch = dsl.IntTuple(
        (batch_t[i] if batch_a[i] == 1 else batch_a[i]) for i in range(len(batch_a))
    )
    return dsl.concat(batch, dsl.IntTuple((m, n)))

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
    n_dl = dl_shape[len(dl_shape) - 1]
    n_d = d_shape[len(d_shape) - 1]
    n_du = du_shape[len(du_shape) - 1]
    n_b = b_shape[len(b_shape) - 2]
    k_b = b_shape[len(b_shape) - 1]
    if n_dl != n_d or n_du != n_d or n_b != n_d:
        return dsl.Invalid("tridiagonal_solve dimension mismatch")
    batch_dl = dl_shape[: len(dl_shape) - 1]
    batch_b = b_shape[: len(b_shape) - 2]
    if len(batch_dl) == 0 and len(batch_b) == 0:
        return dsl.IntTuple((n_d, k_b))
    if len(batch_dl) == 0:
        return dsl.concat(batch_b, dsl.IntTuple((n_d, k_b)))
    if len(batch_b) == 0:
        return dsl.concat(batch_dl, dsl.IntTuple((n_d, k_b)))
    if len(batch_dl) != len(batch_b):
        return dsl.Invalid("arrays must have the same number of batch dimensions")
    if any(
        batch_dl[i] != batch_b[i] and batch_dl[i] != 1 and batch_b[i] != 1
        for i in range(len(batch_dl))
    ):
        return dsl.Invalid("incompatible batch shapes for broadcasting")
    batch = dsl.IntTuple(
        (batch_b[i] if batch_dl[i] == 1 else batch_dl[i]) for i in range(len(batch_dl))
    )
    return dsl.concat(batch, dsl.IntTuple((n_d, k_b)))

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
    m = left[len(left) - 2]
    k_left = left[len(left) - 1]
    k_right = right[len(right) - 1]
    if (
        dsl.is_concrete_int(k_left)
        and dsl.is_concrete_int(k_right)
        and k_left != k_right
    ):
        return dsl.Invalid("matvec inner dimensions must match")
    b_left_len = len(left) - 2
    b_right_len = len(right) - 1
    if b_left_len >= b_right_len:
        diff = b_left_len - b_right_len
        return dsl.IntTuple(
            (
                (
                    (right[i - diff] if left[i] == 1 else left[i])
                    if i >= diff
                    else left[i]
                )
                if i < b_left_len
                else m
            )
            for i in range(b_left_len + 1)
        )
    diff = b_right_len - b_left_len
    return dsl.IntTuple(
        (
            (
                (right[i] if left[i - diff] == 1 else left[i - diff])
                if i >= diff
                else right[i]
            )
            if i < b_right_len
            else m
        )
        for i in range(b_right_len + 1)
    )

@type_shape_dsl_function
def vecmat_shape(left: IntTuple, right: IntTuple) -> IntTuple:
    if len(left) < 1 or len(right) < 2:
        return dsl.Invalid("vecmat requires at least 1-D vector and 2-D matrix")
    k_left = left[len(left) - 1]
    k_right = right[len(right) - 2]
    m = right[len(right) - 1]
    if (
        dsl.is_concrete_int(k_left)
        and dsl.is_concrete_int(k_right)
        and k_left != k_right
    ):
        return dsl.Invalid("vecmat inner dimensions must match")
    b_left_len = len(left) - 1
    b_right_len = len(right) - 2
    if b_left_len >= b_right_len:
        diff = b_left_len - b_right_len
        return dsl.IntTuple(
            (
                (
                    (right[i - diff] if left[i] == 1 else left[i])
                    if i >= diff
                    else left[i]
                )
                if i < b_left_len
                else m
            )
            for i in range(b_left_len + 1)
        )
    diff = b_right_len - b_left_len
    return dsl.IntTuple(
        (
            (
                (right[i] if left[i - diff] == 1 else left[i - diff])
                if i >= diff
                else right[i]
            )
            if i < b_right_len
            else m
        )
        for i in range(b_right_len + 1)
    )

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
    if len(shape) < 2:
        return dsl.Invalid("diagonal requires at least 2-D array")
    d0 = shape[0]
    d1 = shape[1]
    if d0 == d1:
        return dsl.concat(shape[2:], dsl.IntTuple((d0,)))
    if dsl.is_concrete_int(d0) and dsl.is_concrete_int(d1):
        if d0 < d1:
            return dsl.concat(shape[2:], dsl.IntTuple((d0,)))
        return dsl.concat(shape[2:], dsl.IntTuple((d1,)))
    return dsl.concat(shape[2:], dsl.IntTuple((dsl.Int.gradual(),)))

@type_shape_dsl_function
def trace_shape(shape: IntTuple, offset: int, axis1: int, axis2: int) -> IntTuple:
    if len(shape) < 2:
        return dsl.Invalid("trace requires at least 2-D array")
    if len(shape) == 2:
        return dsl.IntTuple(())
    return shape[2:]
