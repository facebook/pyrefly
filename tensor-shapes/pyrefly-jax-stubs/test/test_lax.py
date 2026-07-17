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
    assert_shape(lax.broadcasted_iota(jnp.int32, (2, 3), 0).shape, (2, 3))
    assert_shape(lax.broadcasted_iota(jnp.int32, (2, 3, 4), 1).shape, (2, 3, 4))

    # empty
    assert_shape(lax.empty((), jnp.float32).shape, ())
    assert_shape(lax.empty(4, jnp.float32).shape, (4,))
    assert_shape(lax.empty((2, 3), jnp.float32).shape, (2, 3))
    assert_shape(lax.empty((2, 3, 4), jnp.float32).shape, (2, 3, 4))

    # full
    assert_shape(lax.full((), 1.0, jnp.float32).shape, ())
    assert_shape(lax.full(4, 1.0, jnp.float32).shape, (4,))
    assert_shape(lax.full((2, 3), 1.0, jnp.float32).shape, (2, 3))
    assert_shape(lax.full((2, 3, 4), 1.0, jnp.float32).shape, (2, 3, 4))

    # full_like
    arr = jnp.ones((2, 3))
    assert_shape(lax.full_like(arr, 5.0).shape, (2, 3))
    assert_shape(lax.full_like(arr, 5.0, shape=(4, 5)).shape, (4, 5))
    assert_shape(lax.full_like(arr, 5.0, shape=4).shape, (4,))

    # iota
    assert_shape(lax.iota(jnp.int32, 5).shape, (5,))
    assert_shape(lax.iota(jnp.float32, 10).shape, (10,))


def test_lax_shape_manipulation() -> None:
    x = jnp.ones((2, 3))
    t = jnp.ones((2, 3, 4, 5))

    # broadcast
    assert_shape(lax.broadcast(x, ()).shape, (2, 3))
    assert_shape(lax.broadcast(x, (4,)).shape, (4, 2, 3))
    assert_shape(lax.broadcast(x, (4, 5)).shape, (4, 5, 2, 3))
    assert_shape(lax.broadcast(x, (4, 5, 6)).shape, (4, 5, 6, 2, 3))

    # broadcast_in_dim
    assert_shape(lax.broadcast_in_dim(x, (2, 4, 3), (0, 2)).shape, (2, 4, 3))
    assert_shape(lax.broadcast_in_dim(x, (5, 2, 3), (1, 2)).shape, (5, 2, 3))

    # broadcast_like
    assert_shape(lax.broadcast_like(x, jnp.ones((4, 2, 3))).shape, (4, 2, 3))

    # broadcast_shapes
    assert lax.broadcast_shapes((2, 1), (3,)) == (2, 3)
    assert lax.broadcast_shapes((1, 4), (3, 1), (3, 4)) == (3, 4)

    # broadcast_to_rank
    assert_shape(lax.broadcast_to_rank(x, 2).shape, (2, 3))
    assert_shape(lax.broadcast_to_rank(x, 4).shape, (1, 1, 2, 3))

    # collapse
    assert_shape(lax.collapse(t, 1, 3).shape, (2, 12, 5))
    assert_shape(lax.collapse(t, 0, 4).shape, (120,))
    assert_shape(lax.collapse(t, 1).shape, (2, 60))

    # concatenate
    a = jnp.ones((2, 3))
    b = jnp.ones((2, 4))
    assert_shape(lax.concatenate([a, b], 1).shape, (2, 7))
    c = jnp.ones((5, 3))
    assert_shape(lax.concatenate([a, c], 0).shape, (7, 3))

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
    assert_shape(lax.reshape(x, (6,)).shape, (6,))
    assert_shape(lax.reshape(x, (3, 2), (1, 0)).shape, (3, 2))

    # rev
    assert_shape(lax.rev(x, (0,)).shape, (2, 3))
    assert_shape(lax.rev(x, (0, 1)).shape, (2, 3))

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
    assert_shape(lax.squeeze(sq, (1, 3)).shape, (2, 3))
    assert_shape(lax.squeeze(sq, (3,)).shape, (2, 1, 3))

    # stack
    assert_shape(lax.stack([a, a, a], 0).shape, (3, 2, 3))
    assert_shape(lax.stack([a, a], 1).shape, (2, 2, 3))

    # tile
    res_tile = lax.tile(x, (2, 3))
    assert res_tile.shape == (4, 9)

    # transpose
    assert_shape(lax.transpose(x, (1, 0)).shape, (3, 2))
    assert_shape(lax.transpose(t, (0, 2, 1, 3)).shape, (2, 4, 3, 5))

    # unstack
    u0, u1 = lax.unstack(x, axis=0)
    assert u0.shape == (3,) and u1.shape == (3,)


def test_lax_dtype_bitcast():
    x = jnp.ones((2, 3), dtype=jnp.float32)

    # convert_element_type
    assert_shape(lax.convert_element_type(x, jnp.int32).shape, (2, 3))
    assert_shape(lax.convert_element_type(5.0, jnp.int32).shape, ())
    assert_shape(lax.convert_element_type(True, jnp.float32).shape, ())

    # bitcast_convert_type
    res_bitcast = lax.bitcast_convert_type(x, jnp.int32)
    assert res_bitcast.shape == (2, 3)


def reject_lax_argmax_out_of_bounds(x: jax.Array[[2, 3, 4]]) -> None:
    # E: Cannot evaluate type-level shape DSL call: axis out of bounds
    lax.argmax(x, axis=3, index_dtype=jnp.int32)


def reject_lax_argmax_negative_axis(x: jax.Array[[2, 3, 4]]) -> None:
    # E: Cannot evaluate type-level shape DSL call: axis out of bounds
    lax.argmax(x, axis=-1, index_dtype=jnp.int32)


def reject_lax_argmin_negative_axis(x: jax.Array[[2, 3, 4]]) -> None:
    # E: Cannot evaluate type-level shape DSL call: axis out of bounds
    lax.argmin(x, axis=-1, index_dtype=jnp.int32)


def reject_lax_cumsum_negative_axis(x: jax.Array[[2, 3, 4]]) -> None:
    # E: Cannot evaluate type-level shape DSL call: axis out of bounds
    lax.cumsum(x, axis=-1)


def reject_lax_cummax_out_of_bounds(x: jax.Array[[2, 3, 4]]) -> None:
    # E: Cannot evaluate type-level shape DSL call: axis out of bounds
    lax.cummax(x, axis=3)


def reject_lax_associative_scan_out_of_bounds(x: jax.Array[[2, 3, 4]]) -> None:
    # E: Cannot evaluate type-level shape DSL call: axis out of bounds
    lax.associative_scan(lax.add, x, axis=3)


def reject_lax_associative_scan_negative_out_of_bounds(
    x: jax.Array[[2, 3, 4]],
) -> None:
    # E: Cannot evaluate type-level shape DSL call: axis out of bounds
    lax.associative_scan(lax.add, x, axis=-4)


def reject_lax_reduce_sum_out_of_bounds(x: jax.Array[[2, 3, 4]]) -> None:
    # E: Cannot evaluate type-level shape DSL call: axis out of bounds
    lax.reduce_sum(x, (3,))


def reject_lax_reduce_sum_negative_axis(x: jax.Array[[2, 3, 4]]) -> None:
    # E: Cannot evaluate type-level shape DSL call: axis out of bounds
    lax.reduce_sum(x, (-1,))


def reject_lax_reduce_sum_duplicate_axis(x: jax.Array[[2, 3, 4]]) -> None:
    # E: Cannot evaluate type-level shape DSL call: duplicate axis
    lax.reduce_sum(x, (0, 0))


def reject_lax_approx_max_k_out_of_bounds(x: jax.Array[[2, 3, 4]]) -> None:
    # E: Cannot evaluate type-level shape DSL call: axis out of bounds
    lax.approx_max_k(x, 2, reduction_dimension=3)


def reject_lax_approx_min_k_negative_k(x: jax.Array[[2, 3, 4]]) -> None:
    # E: Cannot evaluate type-level shape DSL call: k must be non-negative
    lax.approx_min_k(x, -1)


def reject_lax_clamp_mismatched_min(
    x: jax.Array[[2, 3, 4]], bad_min: jax.Array[[3, 4]]
) -> None:
    # E: Cannot evaluate type-level shape DSL call: clamp requires min.shape == operand.shape or min.shape == ()
    lax.clamp(bad_min, x, 1.0)


def reject_lax_clamp_mismatched_max(
    x: jax.Array[[2, 3, 4]], bad_max: jax.Array[[3, 4]]
) -> None:
    # E: Cannot evaluate type-level shape DSL call: clamp requires max.shape == operand.shape or max.shape == ()
    lax.clamp(0.0, x, bad_max)


def reject_lax_select_mismatched_cases(
    x: jax.Array[[2, 3, 4]], y: jax.Array[[2, 3]]
) -> None:
    # E: Cannot evaluate type-level shape DSL call: select cases must have the same shapes
    lax.select(True, x, y)


def reject_lax_select_mismatched_pred(
    pred: jax.Array[[4]], x: jax.Array[[2, 3, 4]], y: jax.Array[[2, 3, 4]]
) -> None:
    # E: Cannot evaluate type-level shape DSL call: select `which` must be scalar or have the same shape as cases
    lax.select(pred, x, y)


def reject_lax_select_n_mismatched_which(
    which: jax.Array[[4]], x: jax.Array[[2, 3, 4]], y: jax.Array[[2, 3, 4]]
) -> None:
    # E: Cannot evaluate type-level shape DSL call: select `which` must be scalar or have the same shape as cases
    lax.select_n(which, x, y)


def reject_lax_sort_out_of_bounds(x: jax.Array[[2, 3, 4]]) -> None:
    # E: Cannot evaluate type-level shape DSL call: axis out of bounds
    lax.sort(x, dimension=3)


def reject_lax_sort_key_val_mismatched(
    x: jax.Array[[2, 3, 4]], bad_val: jax.Array[[2, 3]]
) -> None:
    # E: Cannot evaluate type-level shape DSL call: Arguments to sort must have equal shapes
    lax.sort_key_val(x, bad_val)


def reject_lax_top_k_out_of_bounds(x: jax.Array[[2, 3, 4]]) -> None:
    # E: Cannot evaluate type-level shape DSL call: axis out of bounds
    lax.top_k(x, 2, axis=3)


def reject_lax_top_k_negative_k(x: jax.Array[[2, 3, 4]]) -> None:
    # E: Cannot evaluate type-level shape DSL call: k must be non-negative
    lax.top_k(x, -1)


def test_lax_argmax_argmin() -> None:
    x = jnp.ones((2, 3, 4))
    assert_shape(lax.argmax(x, axis=0, index_dtype=jnp.int32).shape, (3, 4))
    assert_shape(lax.argmax(x, axis=1, index_dtype=jnp.int32).shape, (2, 4))
    assert_shape(lax.argmax(x, axis=2, index_dtype=jnp.int32).shape, (2, 3))
    assert_shape(lax.argmin(x, axis=0, index_dtype=jnp.int32).shape, (3, 4))
    assert_shape(lax.argmin(x, axis=1, index_dtype=jnp.int32).shape, (2, 4))
    assert_shape(lax.argmin(x, axis=2, index_dtype=jnp.int32).shape, (2, 3))


def test_lax_scans() -> None:
    x = jnp.ones((2, 3, 4))
    # associative_scan
    assert_shape(lax.associative_scan(lax.add, x, axis=0).shape, (2, 3, 4))
    assert_shape(
        lax.associative_scan(lax.add, x, axis=1, reverse=True).shape, (2, 3, 4)
    )
    assert_shape(lax.associative_scan(lax.add, x, axis=-1).shape, (2, 3, 4))
    assert_shape(lax.associative_scan(lax.add, x).shape, (2, 3, 4))

    # cumlogsumexp
    assert_shape(lax.cumlogsumexp(x, axis=0).shape, (2, 3, 4))
    assert_shape(lax.cumlogsumexp(x, axis=1, reverse=True).shape, (2, 3, 4))
    assert_shape(lax.cumlogsumexp(x).shape, (2, 3, 4))

    # cummax, cummin, cumprod, cumsum
    assert_shape(lax.cummax(x, axis=0).shape, (2, 3, 4))
    assert_shape(lax.cummax(x).shape, (2, 3, 4))
    assert_shape(lax.cummin(x, axis=1).shape, (2, 3, 4))
    assert_shape(lax.cummin(x).shape, (2, 3, 4))
    assert_shape(lax.cumprod(x, axis=2).shape, (2, 3, 4))
    assert_shape(lax.cumprod(x).shape, (2, 3, 4))
    assert_shape(lax.cumsum(x, axis=0).shape, (2, 3, 4))
    assert_shape(lax.cumsum(x).shape, (2, 3, 4))


def test_lax_reductions() -> None:
    x = jnp.ones((2, 3, 4))
    b = jnp.array([[[True, False, True, False]] * 3] * 2)

    # reduce
    assert_shape(lax.reduce(x, 0.0, lax.add, (0, 2)).shape, (3,))
    assert_shape(lax.reduce(x, 0.0, lax.add, ()).shape, (2, 3, 4))
    assert_shape(lax.reduce(x, 0.0, lax.add, (0, 1, 2)).shape, ())

    # reduce_sum, reduce_prod, reduce_max, reduce_min
    assert_shape(lax.reduce_sum(x, (1,)).shape, (2, 4))
    assert_shape(lax.reduce_sum(x, (0, 1, 2)).shape, ())
    assert_shape(lax.reduce_prod(x, ()).shape, (2, 3, 4))
    assert_shape(lax.reduce_prod(x, (0, 2)).shape, (3,))
    assert_shape(lax.reduce_max(x, (2,)).shape, (2, 3))
    assert_shape(lax.reduce_min(x, (0,)).shape, (3, 4))

    # reduce_and, reduce_or, reduce_xor
    assert_shape(lax.reduce_and(b, (0,)).shape, (3, 4))
    assert_shape(lax.reduce_or(b, (1, 2)).shape, (2,))
    assert_shape(lax.reduce_xor(b, (2,)).shape, (2, 3))


def test_lax_reduce_precision() -> None:
    x = jnp.ones((2, 3, 4))
    assert_shape(lax.reduce_precision(x, 8, 7).shape, (2, 3, 4))
    assert_shape(lax.reduce_precision(3.14, 8, 7).shape, ())


def test_lax_reduce_window() -> None:
    x = jnp.ones((2, 4, 6))
    res_window = lax.reduce_window(x, 0.0, lax.add, (1, 2, 2), (1, 1, 1), "VALID")
    assert res_window.shape == (2, 3, 5)

    res_tuple = lax.reduce_window_shape_tuple(
        (2, 4, 6), (1, 2, 2), (1, 1, 1), ((0, 0), (0, 0), (0, 0))
    )
    assert res_tuple == (2, 3, 5)
    # Satisfy assert_shape requirement for the test
    assert_shape(x.shape, (2, 4, 6))


def test_lax_approx_max_min_k() -> None:
    x = jnp.ones((2, 3, 4))
    val_max, idx_max = lax.approx_max_k(x, 2, reduction_dimension=1)
    assert_shape(val_max.shape, (2, 2, 4))
    assert_shape(idx_max.shape, (2, 2, 4))

    val_min, idx_min = lax.approx_min_k(x, 3, reduction_dimension=-1)
    assert_shape(val_min.shape, (2, 3, 3))
    assert_shape(idx_min.shape, (2, 3, 3))


def test_lax_clamp() -> None:
    x = jnp.ones((2, 3, 4))
    assert_shape(lax.clamp(0.0, x, 1.0).shape, (2, 3, 4))
    assert_shape(lax.clamp(jnp.zeros((2, 3, 4)), x, 1.0).shape, (2, 3, 4))
    assert_shape(lax.clamp(0.0, x, jnp.ones((2, 3, 4))).shape, (2, 3, 4))
    assert_shape(
        lax.clamp(jnp.zeros((2, 3, 4)), x, jnp.ones((2, 3, 4))).shape, (2, 3, 4)
    )
    assert_shape(lax.clamp(0.0, 5.0, 1.0).shape, ())


def test_lax_select() -> None:
    x = jnp.ones((2, 3, 4))
    y = jnp.zeros((2, 3, 4))
    pred_0d = jnp.array(True)
    pred_3d = jnp.ones((2, 3, 4), dtype=bool)

    assert_shape(lax.select(True, x, y).shape, (2, 3, 4))
    assert_shape(lax.select(pred_0d, x, y).shape, (2, 3, 4))
    assert_shape(lax.select(pred_3d, x, y).shape, (2, 3, 4))
    assert_shape(lax.select(True, 1.0, 2.0).shape, ())


def test_lax_select_n() -> None:
    x = jnp.ones((2, 3, 4))
    y = jnp.zeros((2, 3, 4))
    which_0d = jnp.array(0)
    which_3d = jnp.zeros((2, 3, 4), dtype=jnp.int32)

    assert_shape(lax.select_n(0, x, y).shape, (2, 3, 4))
    assert_shape(lax.select_n(which_0d, x, y).shape, (2, 3, 4))
    assert_shape(lax.select_n(which_3d, x, y, x).shape, (2, 3, 4))
    assert_shape(lax.select_n(0, 1.0, 2.0).shape, ())


def test_lax_sort() -> None:
    x = jnp.ones((2, 3, 4))
    y = jnp.ones((2, 3, 4))

    assert_shape(lax.sort(x).shape, (2, 3, 4))
    assert_shape(lax.sort(x, dimension=0).shape, (2, 3, 4))
    assert_shape(lax.sort(x, dimension=-1).shape, (2, 3, 4))

    s1, s2 = lax.sort((x, y), dimension=1)
    assert_shape(s1.shape, (2, 3, 4))
    assert_shape(s2.shape, (2, 3, 4))


def test_lax_sort_key_val() -> None:
    x = jnp.ones((2, 3, 4))
    y = jnp.ones((2, 3, 4))

    sk, sv = lax.sort_key_val(x, y, dimension=-1)
    assert_shape(sk.shape, (2, 3, 4))
    assert_shape(sv.shape, (2, 3, 4))


def test_lax_top_k() -> None:
    x = jnp.ones((2, 3, 4))

    top_v, top_i = lax.top_k(x, 2)
    assert_shape(top_v.shape, (2, 3, 2))
    assert_shape(top_i.shape, (2, 3, 2))

    top_v0, top_i0 = lax.top_k(x, 1, axis=0)
    assert_shape(top_v0.shape, (1, 3, 4))
    assert_shape(top_i0.shape, (1, 3, 4))


def test_lax_dynamic_slicing() -> None:
    x = jnp.ones((3, 4, 5))
    assert_shape(lax.dynamic_index_in_dim(x, 1, axis=1, keepdims=True).shape, (3, 1, 5))
    assert_shape(lax.dynamic_index_in_dim(x, 1, axis=1, keepdims=False).shape, (3, 5))
    assert_shape(
        lax.dynamic_index_in_dim(x, 0, axis=-1, keepdims=True).shape, (3, 4, 1)
    )
    assert_shape(lax.dynamic_index_in_dim(x, 0, axis=-1, keepdims=False).shape, (3, 4))

    assert_shape(lax.dynamic_slice(x, (1, 1, 1), (2, 2, 3)).shape, (2, 2, 3))

    assert_shape(lax.dynamic_slice_in_dim(x, 1, 2, axis=1).shape, (3, 2, 5))
    assert_shape(lax.dynamic_slice_in_dim(x, 0, 2, axis=-1).shape, (3, 4, 2))
    assert_shape(
        lax.dynamic_slice_in_dim(x, 0, 2, axis=-1, allow_negative_indices=False).shape,
        (3, 4, 2),
    )
    assert_shape(
        lax.dynamic_slice_in_dim(x, 0, 2, axis=-1, allow_negative_indices=True).shape,
        (3, 4, 2),
    )

    up1 = jnp.zeros((3, 1, 5))
    assert_shape(lax.dynamic_update_index_in_dim(x, up1, 1, axis=1).shape, (3, 4, 5))

    up2 = jnp.zeros((2, 2, 3))
    assert_shape(lax.dynamic_update_slice(x, up2, (1, 1, 1)).shape, (3, 4, 5))

    up3 = jnp.zeros((3, 2, 5))
    assert_shape(lax.dynamic_update_slice_in_dim(x, up3, 1, axis=1).shape, (3, 4, 5))

    assert_shape(lax.index_in_dim(x, 1, axis=1, keepdims=True).shape, (3, 1, 5))
    assert_shape(lax.index_in_dim(x, 1, axis=1, keepdims=False).shape, (3, 5))

    src = jnp.ones((5, 4))
    idxs = jnp.array([[0, 1, 2]])
    res_it = lax.index_take(src, idxs, (0,))
    assert isinstance(res_it, jax.Array)


def test_lax_gather_scatter() -> None:
    operand = jnp.ones((5, 3))
    indices = jnp.array([[1], [2]])
    g_dnums = lax.GatherDimensionNumbers(
        offset_dims=(1,),
        collapsed_slice_dims=(0,),
        start_index_map=(0,),
        operand_batching_dims=(),
        start_indices_batching_dims=(),
    )
    res_g = lax.gather(operand, indices, dimension_numbers=g_dnums, slice_sizes=(1, 3))
    assert isinstance(res_g, jax.Array)
    res_g_clip = lax.gather(
        operand,
        indices,
        dimension_numbers=g_dnums,
        slice_sizes=(1, 3),
        mode=lax.GatherScatterMode.CLIP,
    )
    assert isinstance(res_g_clip, jax.Array)

    s_dnums = lax.ScatterDimensionNumbers(
        update_window_dims=(1,),
        inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,),
        operand_batching_dims=(),
        scatter_indices_batching_dims=(),
    )
    updates = jnp.ones((2, 3))
    assert_shape(lax.scatter(operand, indices, updates, s_dnums).shape, (5, 3))
    assert_shape(
        lax.scatter(
            operand, indices, updates, s_dnums, mode=lax.GatherScatterMode.FILL_OR_DROP
        ).shape,
        (5, 3),
    )
    assert_shape(lax.scatter_add(operand, indices, updates, s_dnums).shape, (5, 3))
    assert_shape(
        lax.scatter_apply(
            operand, indices, lambda u: u * 2, s_dnums, update_shape=(2, 3)
        ).shape,
        (5, 3),
    )
    assert_shape(lax.scatter_max(operand, indices, updates, s_dnums).shape, (5, 3))
    assert_shape(lax.scatter_min(operand, indices, updates, s_dnums).shape, (5, 3))
    assert_shape(lax.scatter_mul(operand, indices, updates, s_dnums).shape, (5, 3))
    assert_shape(lax.scatter_sub(operand, indices, updates, s_dnums).shape, (5, 3))


def test_lax_linear_algebra_contractions() -> None:
    # batch_matmul
    a = jnp.ones((2, 3, 4))
    b = jnp.ones((2, 4, 5))
    assert_shape(lax.batch_matmul(a, b).shape, (2, 3, 5))
    assert_shape(
        lax.batch_matmul(a, b, precision=lax.Precision.HIGHEST).shape, (2, 3, 5)
    )
    a4 = jnp.ones((7, 2, 3, 4))
    b4 = jnp.ones((7, 2, 4, 5))
    assert_shape(lax.batch_matmul(a4, b4).shape, (7, 2, 3, 5))

    # dot
    v1 = jnp.ones(3)
    v2 = jnp.ones(3)
    mat23 = jnp.ones((2, 3))
    mat34 = jnp.ones((3, 4))
    assert_shape(lax.dot(v1, v2).shape, ())
    assert_shape(lax.dot(mat23, v1).shape, (2,))
    assert_shape(lax.dot(v1, mat34).shape, (4,))
    assert_shape(lax.dot(mat23, mat34).shape, (2, 4))
    assert_shape(lax.dot(mat23, mat34, precision=lax.Precision.DEFAULT).shape, (2, 4))

    # dot_general
    dg_res = lax.dot_general(mat23, mat34, (((1,), (0,)), ((), ())))
    assert_shape(dg_res.shape, (2, 4))

    # conv & friends
    lhs = jnp.ones((1, 1, 8, 8))
    rhs = jnp.ones((1, 1, 3, 3))
    conv_res = lax.conv(lhs, rhs, (1, 1), "SAME")
    assert isinstance(conv_res, jax.Array)

    cdnums = lax.conv_dimension_numbers(
        (1, 1, 8, 8), (1, 1, 3, 3), ("NCHW", "OIHW", "NCHW")
    )
    assert cdnums is not None

    cgd_res = lax.conv_general_dilated(lhs, rhs, (1, 1), "SAME")
    assert isinstance(cgd_res, jax.Array)

    cgdp_res = lax.conv_general_dilated_patches(lhs, (3, 3), (1, 1), "SAME")
    assert isinstance(cgdp_res, jax.Array)

    cgdl_lhs = jnp.ones((1, 8, 8, 1))
    cgdl_rhs = jnp.ones((8, 8, 9, 1))
    cgdl_res = lax.conv_general_dilated_local(
        cgdl_lhs,
        cgdl_rhs,
        (1, 1),
        "SAME",
        (3, 3),
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )
    assert isinstance(cgdl_res, jax.Array)

    perms = lax.conv_general_permutations(("NCHW", "OIHW", "NCHW"))
    assert len(perms) == 3

    st1 = lax.conv_general_shape_tuple(
        (1, 1, 8, 8), (1, 1, 3, 3), (1, 1), "SAME", ("NCHW", "OIHW", "NCHW")
    )
    assert len(st1) == 4

    st2 = lax.conv_shape_tuple((1, 1, 8, 8), (1, 1, 3, 3), (1, 1), [(1, 1), (1, 1)])
    assert len(st2) == 4

    ct_lhs = jnp.ones((1, 8, 8, 1))
    ct_rhs = jnp.ones((3, 3, 1, 1))
    ct_res = lax.conv_transpose(ct_lhs, ct_rhs, (1, 1), "SAME")
    assert isinstance(ct_res, jax.Array)

    st3 = lax.conv_transpose_shape_tuple(
        (1, 1, 8, 8), (1, 1, 3, 3), (1, 1), "SAME", ("NCHW", "OIHW", "NCHW")
    )
    assert len(st3) == 4

    cwgp_res = lax.conv_with_general_padding(lhs, rhs, (1, 1), "SAME", None, None)
    assert isinstance(cwgp_res, jax.Array)

    # custom_linear_solve & custom_root
    b_vec = jnp.ones(3)
    sol_cls = lax.custom_linear_solve(lambda x: x, b_vec, lambda f, x: x)
    assert sol_cls is not None

    sol_cr = lax.custom_root(lambda x: x, b_vec, lambda f, x: x, lambda f, x: x)
    assert sol_cr is not None

    # ragged_dot
    rlhs = jnp.ones((6, 4))
    rrhs = jnp.ones((2, 4, 5))
    group_sizes = jnp.array([3, 3])
    rd_res = lax.ragged_dot(rlhs, rrhs, group_sizes)
    assert_shape(rd_res.shape, (6, 5))

    rddnums = lax.RaggedDotDimensionNumbers(
        dot_dimension_numbers=(((1,), (1,)), ((), ())),
        lhs_ragged_dimensions=(0,),
        rhs_group_dimensions=(0,),
    )
    rdg_res = lax.ragged_dot_general(rlhs, rrhs, group_sizes, rddnums)
    assert isinstance(rdg_res, jax.Array)

    # scaled_dot
    sd_res = lax.scaled_dot(mat23, mat34)
    assert isinstance(sd_res, jax.Array)

    # Section 13: Types, Enums & Dimension Descriptors
    assert lax.AccuracyMode is not None
    assert lax.DotAlgorithm is not None
    assert lax.DotAlgorithmPreset is not None
    assert lax.FftType is not None
    assert lax.GatherDimensionNumbers is not None
    assert lax.GatherScatterMode is not None
    assert lax.Precision is not None
    assert lax.RandomAlgorithm is not None
    assert lax.RoundingMethod is not None
    assert lax.ScatterDimensionNumbers is not None
    assert lax.Tolerance is not None
