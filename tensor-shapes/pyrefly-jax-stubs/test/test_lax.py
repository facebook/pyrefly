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

    assert_shape(lax.abs(vec).shape, (4,))
    assert_shape(lax.neg(mat).shape, (2, 3))
    assert_shape(lax.exp(tensor).shape, (2, 3, 4))
    assert_shape(lax.log(mat).shape, (2, 3))
    assert_shape(lax.sqrt(vec).shape, (4,))
    assert_shape(lax.sin(mat).shape, (2, 3))
    assert_shape(lax.cos(tensor).shape, (2, 3, 4))
    assert_shape(lax.tan(vec).shape, (4,))
    assert_shape(lax.tanh(mat).shape, (2, 3))
    assert_shape(lax.ceil(vec).shape, (4,))
    assert_shape(lax.floor(mat).shape, (2, 3))
    assert_shape(lax.round(tensor).shape, (2, 3, 4))
    assert_shape(lax.sign(vec).shape, (4,))
    assert_shape(lax.square(mat).shape, (2, 3))
    assert_shape(lax.rsqrt(tensor).shape, (2, 3, 4))
    assert_shape(lax.is_finite(vec).shape, (4,))
    assert_shape(lax.integer_pow(mat, 3).shape, (2, 3))

    c_vec = vec * 1j
    c_mat = mat * 1j
    c_tensor = tensor * 1j
    assert_shape(lax.real(c_vec).shape, (4,))
    assert_shape(lax.imag(c_mat).shape, (2, 3))
    assert_shape(lax.conj(c_tensor).shape, (2, 3, 4))


def test_binary_broadcasting_with_scalars() -> None:
    scalar_arr = jnp.ones(())
    vec = jnp.ones(4)
    mat = jnp.ones((2, 3))

    # Python scalar with Array
    assert_shape(lax.add(1.0, vec).shape, (4,))
    assert_shape(lax.add(vec, 1.0).shape, (4,))
    assert_shape(lax.sub(2.0, mat).shape, (2, 3))
    assert_shape(lax.mul(mat, 3.0).shape, (2, 3))

    # 0-D Array with N-D Array
    assert_shape(lax.add(scalar_arr, vec).shape, (4,))
    assert_shape(lax.add(vec, scalar_arr).shape, (4,))
    assert_shape(lax.add(scalar_arr, mat).shape, (2, 3))
    assert_shape(lax.add(mat, scalar_arr).shape, (2, 3))
    assert_shape(lax.add(scalar_arr, scalar_arr).shape, ())


def test_binary_broadcasting_same_rank() -> None:
    row = jnp.ones((1, 4))
    col = jnp.ones((3, 1))
    mat = jnp.ones((3, 4))

    assert_shape(lax.add(row, col).shape, (3, 4))
    assert_shape(lax.add(col, row).shape, (3, 4))
    assert_shape(lax.add(row, mat).shape, (3, 4))
    assert_shape(lax.add(mat, col).shape, (3, 4))

    # 3-D same rank broadcasting
    a3d = jnp.ones((2, 1, 4))
    b3d = jnp.ones((1, 3, 4))
    assert_shape(lax.add(a3d, b3d).shape, (2, 3, 4))
    assert_shape(lax.sub(a3d, b3d).shape, (2, 3, 4))
    assert_shape(lax.mul(a3d, b3d).shape, (2, 3, 4))
    assert_shape(lax.div(a3d, b3d).shape, (2, 3, 4))
    assert_shape(lax.max(a3d, b3d).shape, (2, 3, 4))
    assert_shape(lax.min(a3d, b3d).shape, (2, 3, 4))
    assert_shape(lax.atan2(a3d, b3d).shape, (2, 3, 4))
    assert_shape(lax.pow(a3d, b3d).shape, (2, 3, 4))
    assert_shape(lax.rem(a3d, b3d).shape, (2, 3, 4))


def test_binary_bitwise_and_comparison() -> None:
    a = jnp.ones((2, 3), dtype=jnp.int32)
    b = jnp.ones((1, 3), dtype=jnp.int32)

    assert_shape(lax.bitwise_and(a, b).shape, (2, 3))
    assert_shape(lax.bitwise_or(a, b).shape, (2, 3))
    assert_shape(lax.bitwise_xor(a, b).shape, (2, 3))
    assert_shape(lax.shift_left(a, b).shape, (2, 3))
    assert_shape(lax.shift_right_arithmetic(a, b).shape, (2, 3))
    assert_shape(lax.shift_right_logical(a, b).shape, (2, 3))

    assert_shape(lax.eq(a, b).shape, (2, 3))
    assert_shape(lax.ne(a, b).shape, (2, 3))
    assert_shape(lax.lt(a, b).shape, (2, 3))
    assert_shape(lax.le(a, b).shape, (2, 3))
    assert_shape(lax.gt(a, b).shape, (2, 3))
    assert_shape(lax.ge(a, b).shape, (2, 3))


def test_binary_rejects_differing_non_scalar_ranks() -> None:
    vec = jnp.ones(4)
    row = jnp.ones((1, 4))
    mat = jnp.ones((3, 1))

    # Same rank broadcasting works
    assert_shape(lax.add(row, mat).shape, (3, 4))

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

    assert_shape(lax.add(a, jnp.ones((2, 3))).shape, (2, 3))
    try:
        # E: Cannot evaluate type-level shape DSL call: Cannot broadcast dimension Int[3] with dimension Int[4] at position 1
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

    assert_shape(lax.linalg.cholesky(x).shape, (2, 2))
    assert_shape(lax.linalg.cholesky(x_batch).shape, (4, 2, 2))

    r = jnp.ones((2, 2))
    w = jnp.ones(2)
    assert_shape(lax.linalg.cholesky_update(r, w).shape, (2, 2))

    eig_out = lax.linalg.eig(x)
    assert [e.shape for e in eig_out] == [(2,), (2, 2), (2, 2)]

    eig_out_batch = lax.linalg.eig(x_batch)
    assert [e.shape for e in eig_out_batch] == [(4, 2), (4, 2, 2), (4, 2, 2)]

    v, eig_w = lax.linalg.eigh(x)
    assert_shape(v.shape, (2, 2))
    assert_shape(eig_w.shape, (2,))

    v_batch, eig_w_batch = lax.linalg.eigh(x_batch)
    assert_shape(v_batch.shape, (4, 2, 2))
    assert_shape(eig_w_batch.shape, (4, 2))

    h_mat, taus = lax.linalg.hessenberg(x)
    assert_shape(h_mat.shape, (2, 2))
    assert_shape(taus.shape, (1,))

    a_house = jnp.ones((3, 2))
    taus_house = jnp.ones(2)
    assert_shape(lax.linalg.householder_product(a_house, taus_house).shape, (3, 2))

    lu_out, piv, perm = lax.linalg.lu(x_rec)
    assert_shape(lu_out.shape, (2, 3))
    assert_shape(piv.shape, (2,))
    assert_shape(perm.shape, (2,))

    assert_shape(lax.linalg.lu_pivots_to_permutation(piv, 2).shape, (2,))

    c_mat = jnp.ones((3, 4))
    assert_shape(lax.linalg.ormqr(a_house, taus_house, c_mat, left=True).shape, (3, 4))

    u_qdwh, h_qdwh, iters, conv = lax.linalg.qdwh(x)
    assert_shape(u_qdwh.shape, (2, 2))
    assert_shape(h_qdwh.shape, (2, 2))
    assert_shape(iters.shape, ())
    assert_shape(conv.shape, ())

    q, r_qr = lax.linalg.qr(x_rec, full_matrices=True)
    assert_shape(q.shape, (2, 2))
    assert_shape(r_qr.shape, (2, 3))

    q_red, r_red = lax.linalg.qr(x_rec, full_matrices=False)
    assert_shape(q_red.shape, (2, 2))
    assert_shape(r_red.shape, (2, 3))

    t_schur, z_schur = lax.linalg.schur(x)
    assert_shape(t_schur.shape, (2, 2))
    assert_shape(z_schur.shape, (2, 2))

    u_svd, s_svd, vt_svd = lax.linalg.svd(x_rec, full_matrices=True)
    assert_shape(u_svd.shape, (2, 2))
    assert_shape(s_svd.shape, (2,))
    assert_shape(vt_svd.shape, (3, 3))

    u_svd_r, s_svd_r, vt_svd_r = lax.linalg.svd(x_rec, full_matrices=False)
    assert_shape(u_svd_r.shape, (2, 2))
    assert_shape(s_svd_r.shape, (2,))
    assert_shape(vt_svd_r.shape, (2, 3))

    assert_shape(lax.linalg.svd(x_rec, compute_uv=False).shape, (2,))

    a_sym = jnp.ones((2, 3))
    c_sym = jnp.ones((2, 2))
    assert_shape(lax.linalg.symmetric_product(a_sym, c_sym).shape, (2, 2))
    assert_shape(
        lax.linalg.symmetric_product(jnp.ones((4, 2, 3)), jnp.ones((4, 2, 2))).shape,
        (4, 2, 2),
    )

    b_left = jnp.ones((2, 3))
    assert_shape(lax.linalg.triangular_solve(r, b_left, left_side=True).shape, (2, 3))
    b_right = jnp.ones((3, 2))
    assert_shape(lax.linalg.triangular_solve(r, b_right, left_side=False).shape, (3, 2))

    tri_a, tri_d, tri_e, tri_tau = lax.linalg.tridiagonal(x)
    assert_shape(tri_a.shape, (2, 2))
    assert_shape(tri_d.shape, (2,))
    assert_shape(tri_e.shape, (1,))
    assert_shape(tri_tau.shape, (1,))

    dl = jnp.ones(2)
    d = jnp.ones(2)
    du = jnp.ones(2)
    b_tri = jnp.ones((2, 3))
    assert_shape(lax.linalg.tridiagonal_solve(dl, d, du, b_tri).shape, (2, 3))
    assert_shape(
        lax.linalg.tridiagonal_solve(
            jnp.ones((4, 2)), jnp.ones((4, 2)), jnp.ones((4, 2)), jnp.ones((4, 2, 3))
        ).shape,
        (4, 2, 3),
    )


def test_lax_linalg_shape_errors() -> None:
    assert_shape(
        lax.linalg.symmetric_product(jnp.ones((2, 3)), jnp.ones((2, 2))).shape, (2, 2)
    )

    try:
        # E: Cannot evaluate type-level shape DSL call: gufunc: core dimension 'm' has conflicting extents 2 and 3
        lax.linalg.symmetric_product(jnp.ones((2, 3)), jnp.ones((3, 3)))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")

    try:
        # E: Cannot evaluate type-level shape DSL call: gufunc: core dimension 'm' has conflicting extents 2 and 4
        lax.linalg.symmetric_product(jnp.ones((2, 3)), jnp.ones((2, 4)))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")

    try:
        # E: Cannot evaluate type-level shape DSL call: gufunc: core dimension 'm' has conflicting extents 2 and 3
        lax.linalg.triangular_solve(jnp.ones((2, 2)), jnp.ones((3, 3)), left_side=True)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")

    try:
        # E: Cannot evaluate type-level shape DSL call: gufunc: core dimension 'n' has conflicting extents 2 and 3
        lax.linalg.cholesky_update(jnp.ones((2, 3)), jnp.ones(2))
    except (TypeError, ValueError):
        pass
    else:
        raise AssertionError("expected error")


def test_lax_creation() -> None:
    # broadcasted_iota
    assert_shape(lax.broadcasted_iota(jnp.int32, (2, 3), 0), (2, 3))
    assert_shape(lax.broadcasted_iota(jnp.int32, (2, 3, 4), 1), (2, 3, 4))

    # empty
    assert_shape(lax.empty((), jnp.float32), ())
    assert_shape(lax.empty(4, jnp.float32), (4,))
    assert_shape(lax.empty((2, 3), jnp.float32), (2, 3))
    assert_shape(lax.empty((2, 3, 4), jnp.float32), (2, 3, 4))

    # full
    assert_shape(lax.full((), 1.0, jnp.float32), ())
    assert_shape(lax.full(4, 1.0, jnp.float32), (4,))
    assert_shape(lax.full((2, 3), 1.0, jnp.float32), (2, 3))
    assert_shape(lax.full((2, 3, 4), 1.0, jnp.float32), (2, 3, 4))

    # full_like
    arr = jnp.ones((2, 3))
    assert_shape(lax.full_like(arr, 5.0), (2, 3))
    assert_shape(lax.full_like(arr, 5.0, shape=(4, 5)), (4, 5))
    assert_shape(lax.full_like(arr, 5.0, shape=4), (4,))

    # iota
    assert_shape(lax.iota(jnp.int32, 5), (5,))
    assert_shape(lax.iota(jnp.float32, 10), (10,))


def test_lax_shape_manipulation() -> None:
    x = jnp.ones((2, 3))
    t = jnp.ones((2, 3, 4, 5))

    # broadcast
    assert_shape(lax.broadcast(x, ()), (2, 3))
    assert_shape(lax.broadcast(x, (4,)), (4, 2, 3))
    assert_shape(lax.broadcast(x, (4, 5)), (4, 5, 2, 3))
    assert_shape(lax.broadcast(x, (4, 5, 6)), (4, 5, 6, 2, 3))

    # broadcast_in_dim
    assert_shape(lax.broadcast_in_dim(x, (2, 4, 3), (0, 2)), (2, 4, 3))
    assert_shape(lax.broadcast_in_dim(x, (5, 2, 3), (1, 2)), (5, 2, 3))

    # broadcast_like
    assert_shape(lax.broadcast_like(x, jnp.ones((4, 2, 3))), (4, 2, 3))

    # broadcast_shapes
    assert lax.broadcast_shapes((2, 1), (3,)) == (2, 3)
    assert lax.broadcast_shapes((1, 4), (3, 1), (3, 4)) == (3, 4)

    # broadcast_to_rank
    assert_shape(lax.broadcast_to_rank(x, 2), (2, 3))
    assert_shape(lax.broadcast_to_rank(x, 4), (1, 1, 2, 3))

    # collapse
    assert_shape(lax.collapse(t, 1, 3), (2, 12, 5))
    assert_shape(lax.collapse(t, 0, 4), (120,))
    assert_shape(lax.collapse(t, 1), (2, 60))

    # concatenate
    a = jnp.ones((2, 3))
    b = jnp.ones((2, 4))
    assert_shape(lax.concatenate([a, b], 1), (2, 7))
    c = jnp.ones((5, 3))
    assert_shape(lax.concatenate([a, c], 0), (7, 3))

    # expand_dims
    res_expand = lax.expand_dims(x, (1, 3))
    assert res_expand.shape == (2, 1, 3, 1)

    # pad
    vec = jnp.ones(4)
    res_pad = lax.pad(vec, 0.0, [(1, 2, 0)])
    assert res_pad.shape == (7,)

    # padtype_to_pads
    pads = lax.padtype_to_pads((10,), (3,), (1,), "SAME")
    assert isinstance(pads, list)

    # reshape
    assert_shape(lax.reshape(x, (6,)), (6,))
    assert_shape(lax.reshape(x, (3, 2), (1, 0)), (3, 2))

    # rev
    assert_shape(lax.rev(x, (0,)), (2, 3))
    assert_shape(lax.rev(x, (0, 1)), (2, 3))

    # slice
    res_slice = lax.slice(x, (0, 1), (2, 3))
    assert res_slice.shape == (2, 2)

    # slice_in_dim
    res_slice_dim = lax.slice_in_dim(x, 0, 1, axis=0)
    assert res_slice_dim.shape == (1, 3)

    # split
    s0, s1 = lax.split(x, (1, 1), axis=0)
    assert s0.shape == (1, 3) and s1.shape == (1, 3)

    # squeeze
    sq = jnp.ones((2, 1, 3, 1))
    assert_shape(lax.squeeze(sq, (1, 3)), (2, 3))
    assert_shape(lax.squeeze(sq, (3,)), (2, 1, 3))

    # stack
    assert_shape(lax.stack([a, a, a], 0), (3, 2, 3))
    assert_shape(lax.stack([a, a], 1), (2, 2, 3))

    # tile
    res_tile = lax.tile(x, (2, 3))
    assert res_tile.shape == (4, 9)

    # transpose
    assert_shape(lax.transpose(x, (1, 0)), (3, 2))
    assert_shape(lax.transpose(t, (0, 2, 1, 3)), (2, 4, 3, 5))

    # unstack
    u0, u1 = lax.unstack(x, axis=0)
    assert u0.shape == (3,) and u1.shape == (3,)


def test_lax_dtype_bitcast():
    x = jnp.ones((2, 3), dtype=jnp.float32)

    # convert_element_type
    assert_shape(lax.convert_element_type(x, jnp.int32), (2, 3))
    assert_shape(lax.convert_element_type(5.0, jnp.int32), ())
    assert_shape(lax.convert_element_type(True, jnp.float32), ())

    # bitcast_convert_type
    res_bitcast = lax.bitcast_convert_type(x, jnp.int32)
    assert res_bitcast.shape == (2, 3)
