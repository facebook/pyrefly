# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import jax
import jax.lax as lax
import jax.numpy as jnp
from shape_extensions import assert_shape, IntTuple


def generic_unary_preserves_shape[Shape: IntTuple](
    x: jax.Array[Shape],
) -> jax.Array[Shape]:
    return lax.sin(x)


def test_unary_elementwise() -> None:
    vec = jnp.ones(4)
    mat = jnp.ones((2, 3))
    tensor = jnp.ones((2, 3, 4))

    assert_shape(lax.abs(vec), (4,))
    assert_shape(lax.neg(mat), (2, 3))
    assert_shape(lax.exp(tensor), (2, 3, 4))
    assert_shape(lax.log(mat), (2, 3))
    assert_shape(lax.sqrt(vec), (4,))
    assert_shape(lax.sin(mat), (2, 3))
    assert_shape(lax.cos(tensor), (2, 3, 4))
    assert_shape(lax.tan(vec), (4,))
    assert_shape(lax.tanh(mat), (2, 3))
    assert_shape(lax.ceil(vec), (4,))
    assert_shape(lax.floor(mat), (2, 3))
    assert_shape(lax.round(tensor), (2, 3, 4))
    assert_shape(lax.sign(vec), (4,))
    assert_shape(lax.square(mat), (2, 3))
    assert_shape(lax.rsqrt(tensor), (2, 3, 4))
    assert_shape(lax.is_finite(vec), (4,))
    assert_shape(lax.integer_pow(mat, 3), (2, 3))

    c_vec = vec * 1j
    c_mat = mat * 1j
    c_tensor = tensor * 1j
    assert_shape(lax.real(c_vec), (4,))
    assert_shape(lax.imag(c_mat), (2, 3))
    assert_shape(lax.conj(c_tensor), (2, 3, 4))


def test_binary_broadcasting_with_scalars() -> None:
    scalar_arr = jnp.ones(())
    vec = jnp.ones(4)
    mat = jnp.ones((2, 3))

    # Python scalar with Array
    assert_shape(lax.add(1.0, vec), (4,))
    assert_shape(lax.add(vec, 1.0), (4,))
    assert_shape(lax.sub(2.0, mat), (2, 3))
    assert_shape(lax.mul(mat, 3.0), (2, 3))

    # 0-D Array with N-D Array
    assert_shape(lax.add(scalar_arr, vec), (4,))
    assert_shape(lax.add(vec, scalar_arr), (4,))
    assert_shape(lax.add(scalar_arr, mat), (2, 3))
    assert_shape(lax.add(mat, scalar_arr), (2, 3))
    assert_shape(lax.add(scalar_arr, scalar_arr), ())


def test_binary_broadcasting_same_rank() -> None:
    row = jnp.ones((1, 4))
    col = jnp.ones((3, 1))
    mat = jnp.ones((3, 4))

    assert_shape(lax.add(row, col), (3, 4))
    assert_shape(lax.add(col, row), (3, 4))
    assert_shape(lax.add(row, mat), (3, 4))
    assert_shape(lax.add(mat, col), (3, 4))

    # 3-D same rank broadcasting
    a3d = jnp.ones((2, 1, 4))
    b3d = jnp.ones((1, 3, 4))
    assert_shape(lax.add(a3d, b3d), (2, 3, 4))
    assert_shape(lax.sub(a3d, b3d), (2, 3, 4))
    assert_shape(lax.mul(a3d, b3d), (2, 3, 4))
    assert_shape(lax.div(a3d, b3d), (2, 3, 4))
    assert_shape(lax.max(a3d, b3d), (2, 3, 4))
    assert_shape(lax.min(a3d, b3d), (2, 3, 4))
    assert_shape(lax.atan2(a3d, b3d), (2, 3, 4))
    assert_shape(lax.pow(a3d, b3d), (2, 3, 4))
    assert_shape(lax.rem(a3d, b3d), (2, 3, 4))


def test_binary_bitwise_and_comparison() -> None:
    a = jnp.ones((2, 3), dtype=jnp.int32)
    b = jnp.ones((1, 3), dtype=jnp.int32)

    assert_shape(lax.bitwise_and(a, b), (2, 3))
    assert_shape(lax.bitwise_or(a, b), (2, 3))
    assert_shape(lax.bitwise_xor(a, b), (2, 3))
    assert_shape(lax.shift_left(a, b), (2, 3))
    assert_shape(lax.shift_right_arithmetic(a, b), (2, 3))
    assert_shape(lax.shift_right_logical(a, b), (2, 3))

    assert_shape(lax.eq(a, b), (2, 3))
    assert_shape(lax.ne(a, b), (2, 3))
    assert_shape(lax.lt(a, b), (2, 3))
    assert_shape(lax.le(a, b), (2, 3))
    assert_shape(lax.gt(a, b), (2, 3))
    assert_shape(lax.ge(a, b), (2, 3))


def test_binary_rejects_differing_non_scalar_ranks() -> None:
    vec = jnp.ones(4)
    row = jnp.ones((1, 4))
    mat = jnp.ones((3, 1))

    # Same rank broadcasting works
    assert_shape(lax.add(row, mat), (3, 4))

    try:
        # E: Cannot evaluate type-level shape DSL call: arrays must have the same number of dimensions
        lax.add(vec, mat)
    except TypeError:
        pass
    else:
        raise AssertionError(
            "expected JAX to reject differing-rank non-scalar operands in lax.add"
        )


def test_binary_rejects_incompatible_dimensions() -> None:
    a = jnp.ones((2, 3))
    b = jnp.ones((2, 4))

    assert_shape(lax.add(a, jnp.ones((2, 3))), (2, 3))
    try:
        # E: Cannot evaluate type-level shape DSL call: incompatible shapes for broadcasting
        lax.add(a, b)
    except TypeError:
        pass
    else:
        raise AssertionError(
            "expected JAX to reject incompatible dimensions in lax.add"
        )


def test_lax_linalg() -> None:
    x = jnp.ones((2, 2))
    x_batch = jnp.ones((4, 2, 2))
    x_rec = jnp.ones((2, 3))
    x_rec_batch = jnp.ones((4, 2, 3))

    assert_shape(lax.linalg.cholesky(x), (2, 2))
    assert_shape(lax.linalg.cholesky(x_batch), (4, 2, 2))

    r = jnp.ones((2, 2))
    w = jnp.ones(2)
    assert_shape(lax.linalg.cholesky_update(r, w), (2, 2))

    eig_out = lax.linalg.eig(x)
    assert [e.shape for e in eig_out] == [(2,), (2, 2), (2, 2)]

    eig_out_batch = lax.linalg.eig(x_batch)
    assert [e.shape for e in eig_out_batch] == [(4, 2), (4, 2, 2), (4, 2, 2)]

    v, eig_w = lax.linalg.eigh(x)
    assert_shape(v, (2, 2))
    assert_shape(eig_w, (2,))

    v_batch, eig_w_batch = lax.linalg.eigh(x_batch)
    assert_shape(v_batch, (4, 2, 2))
    assert_shape(eig_w_batch, (4, 2))

    h_mat, taus = lax.linalg.hessenberg(x)
    assert_shape(h_mat, (2, 2))
    assert_shape(taus, (1,))

    a_house = jnp.ones((3, 2))
    taus_house = jnp.ones(2)
    assert_shape(lax.linalg.householder_product(a_house, taus_house), (3, 2))

    lu_out, piv, perm = lax.linalg.lu(x_rec)
    assert_shape(lu_out, (2, 3))
    assert_shape(piv, (2,))
    assert_shape(perm, (2,))

    assert_shape(lax.linalg.lu_pivots_to_permutation(piv, 2), (2,))

    c_mat = jnp.ones((3, 4))
    assert_shape(lax.linalg.ormqr(a_house, taus_house, c_mat, left=True), (3, 4))

    u_qdwh, h_qdwh, iters, conv = lax.linalg.qdwh(x)
    assert_shape(u_qdwh, (2, 2))
    assert_shape(h_qdwh, (2, 2))
    assert_shape(iters, ())
    assert_shape(conv, ())

    q, r_qr = lax.linalg.qr(x_rec, full_matrices=True)
    assert_shape(q, (2, 2))
    assert_shape(r_qr, (2, 3))

    q_red, r_red = lax.linalg.qr(x_rec, full_matrices=False)
    assert_shape(q_red, (2, 2))
    assert_shape(r_red, (2, 3))

    t_schur, z_schur = lax.linalg.schur(x)
    assert_shape(t_schur, (2, 2))
    assert_shape(z_schur, (2, 2))

    u_svd, s_svd, vt_svd = lax.linalg.svd(x_rec, full_matrices=True)
    assert_shape(u_svd, (2, 2))
    assert_shape(s_svd, (2,))
    assert_shape(vt_svd, (3, 3))

    u_svd_r, s_svd_r, vt_svd_r = lax.linalg.svd(x_rec, full_matrices=False)
    assert_shape(u_svd_r, (2, 2))
    assert_shape(s_svd_r, (2,))
    assert_shape(vt_svd_r, (2, 3))

    assert_shape(lax.linalg.svd(x_rec, compute_uv=False), (2,))

    a_sym = jnp.ones((2, 3))
    c_sym = jnp.ones((2, 2))
    assert_shape(lax.linalg.symmetric_product(a_sym, c_sym), (2, 2))
    assert_shape(
        lax.linalg.symmetric_product(jnp.ones((4, 2, 3)), jnp.ones((4, 2, 2))),
        (4, 2, 2),
    )

    b_left = jnp.ones((2, 3))
    assert_shape(lax.linalg.triangular_solve(r, b_left, left_side=True), (2, 3))
    b_right = jnp.ones((3, 2))
    assert_shape(lax.linalg.triangular_solve(r, b_right, left_side=False), (3, 2))

    tri_a, tri_d, tri_e, tri_tau = lax.linalg.tridiagonal(x)
    assert_shape(tri_a, (2, 2))
    assert_shape(tri_d, (2,))
    assert_shape(tri_e, (1,))
    assert_shape(tri_tau, (1,))

    dl = jnp.ones(2)
    d = jnp.ones(2)
    du = jnp.ones(2)
    b_tri = jnp.ones((2, 3))
    assert_shape(lax.linalg.tridiagonal_solve(dl, d, du, b_tri), (2, 3))
    assert_shape(
        lax.linalg.tridiagonal_solve(
            jnp.ones((4, 2)), jnp.ones((4, 2)), jnp.ones((4, 2)), jnp.ones((4, 2, 3))
        ),
        (4, 2, 3),
    )


def test_lax_linalg_shape_errors() -> None:
    assert_shape(
        lax.linalg.symmetric_product(jnp.ones((2, 3)), jnp.ones((2, 2))), (2, 2)
    )

    try:
        # E: Cannot evaluate type-level shape DSL call: leading core dimensions of a_matrix and c_matrix must match
        lax.linalg.symmetric_product(jnp.ones((2, 3)), jnp.ones((3, 3)))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")

    try:
        # E: Cannot evaluate type-level shape DSL call: c_matrix must be square
        lax.linalg.symmetric_product(jnp.ones((2, 3)), jnp.ones((2, 4)))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")

    try:
        # E: Cannot evaluate type-level shape DSL call: incompatible shapes for triangular_solve
        lax.linalg.triangular_solve(jnp.ones((2, 2)), jnp.ones((3, 3)), left_side=True)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")

    try:
        # E: Cannot evaluate type-level shape DSL call: r_matrix must be square
        lax.linalg.cholesky_update(jnp.ones((2, 3)), jnp.ones(2))
    except (TypeError, ValueError):
        pass
    else:
        raise AssertionError("expected error")
