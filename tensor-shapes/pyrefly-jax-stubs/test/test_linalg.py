# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import jax
import jax.numpy as jnp
from shape_extensions import assert_shape, Elements, IntTuple, IntVar


def square_svd_components[N: IntVar](
    x: jax.Array[[N, N]],
) -> tuple[jax.Array[[N, N]], jax.Array[[N]], jax.Array[[N, N]]]:
    return jnp.linalg.svd(x, full_matrices=False)


def test_svd_reduced_wide_matrix() -> None:
    x = jnp.ones((3, 5))

    u, s, vt = jnp.linalg.svd(x, full_matrices=False)

    assert_shape(u, (3, 3))
    assert_shape(s, (3,))
    assert_shape(vt, (3, 5))


def test_svd_reduced_tall_matrix() -> None:
    x = jnp.ones((5, 3))

    u, s, vt = jnp.linalg.svd(x, full_matrices=False)

    assert_shape(u, (5, 3))
    assert_shape(s, (3,))
    assert_shape(vt, (3, 3))


def test_svd_reduced_square_matrix() -> None:
    x = jnp.ones((4, 4))

    u, s, vt = jnp.linalg.svd(x, full_matrices=False)

    assert_shape(u, (4, 4))
    assert_shape(s, (4,))
    assert_shape(vt, (4, 4))

    u_c, s_c, vt_c = square_svd_components(x)
    assert_shape(u_c, (4, 4))
    assert_shape(s_c, (4,))
    assert_shape(vt_c, (4, 4))


def test_svd_compute_uv_false() -> None:
    x = jnp.ones((5, 3))

    s = jnp.linalg.svd(x, compute_uv=False)

    assert_shape(s, (3,))
    assert_shape(jnp.linalg.svdvals(x), (3,))


def test_svd_full_matrices() -> None:
    x = jnp.ones((3, 5))

    u, s, vt = jnp.linalg.svd(x, full_matrices=True)

    assert_shape(u, (3, 3))
    assert_shape(s, (3,))
    assert_shape(vt, (5, 5))


def test_qr_reduced() -> None:
    wide = jnp.ones((3, 5))
    tall = jnp.ones((5, 3))

    q_w, r_w = jnp.linalg.qr(wide, mode="reduced")
    assert_shape(q_w, (3, 3))
    assert_shape(r_w, (3, 5))

    q_t, r_t = jnp.linalg.qr(tall, mode="reduced")
    assert_shape(q_t, (5, 3))
    assert_shape(r_t, (3, 3))


def test_qr_r_mode() -> None:
    tall = jnp.ones((5, 3))

    r = jnp.linalg.qr(tall, mode="r")
    assert_shape(r, (3, 3))


def test_solve_vector_rhs() -> None:
    a = jnp.eye(3)
    b = jnp.ones(3)

    assert_shape(jnp.linalg.solve(a, b), (3,))


def test_solve_matrix_rhs() -> None:
    a = jnp.eye(3)
    b = jnp.ones((3, 2))

    assert_shape(jnp.linalg.solve(a, b), (3, 2))


def test_inv_and_matrix_power() -> None:
    a = jnp.eye(4)

    assert_shape(jnp.linalg.inv(a), (4, 4))
    assert_shape(jnp.linalg.matrix_power(a, 3), (4, 4))
    assert_shape(jnp.linalg.cholesky(a), (4, 4))


def test_eigh_and_eig() -> None:
    a = jnp.eye(5)

    w_h, v_h = jnp.linalg.eigh(a)
    assert_shape(w_h, (5,))
    assert_shape(v_h, (5, 5))
    assert_shape(jnp.linalg.eigvalsh(a), (5,))

    w, v = jnp.linalg.eig(a)
    assert_shape(w, (5,))
    assert_shape(v, (5, 5))
    assert_shape(jnp.linalg.eigvals(a), (5,))


def test_scalar_matrix_properties() -> None:
    a = jnp.eye(3)
    rect = jnp.ones((3, 4))

    assert_shape(jnp.linalg.det(a), ())
    sign, logdet = jnp.linalg.slogdet(a)
    assert_shape(sign, ())
    assert_shape(logdet, ())
    assert_shape(jnp.linalg.matrix_rank(rect), ())
    assert_shape(jnp.linalg.cond(a), ())


def test_matrix_transpose() -> None:
    x = jnp.ones((3, 4))
    batched = jnp.ones((2, 3, 4))

    assert_shape(jnp.linalg.matrix_transpose(x), (4, 3))
    assert_shape(jnp.linalg.matrix_transpose(batched), (2, 4, 3))


def test_matmul() -> None:
    mat23 = jnp.ones((2, 3))
    mat34 = jnp.ones((3, 4))
    vec3 = jnp.ones(3)

    assert_shape(jnp.linalg.matmul(mat23, mat34), (2, 4))
    assert_shape(jnp.linalg.matmul(mat23, vec3), (2,))
    assert_shape(jnp.linalg.matmul(vec3, mat34), (4,))
    assert_shape(jnp.linalg.matmul(vec3, vec3), ())


def test_matmul_rejects_mismatched_inner_dimension() -> None:
    a = jnp.ones((3, 4))

    assert_shape(jnp.linalg.matmul(a, jnp.ones((4, 5))), (3, 5))
    try:
        # E: Cannot evaluate type-level shape DSL call: gufunc: core dimension 'n' has conflicting extents 4 and 7
        jnp.linalg.matmul(a, jnp.ones((7, 5)))
    except TypeError:
        pass
    else:
        raise AssertionError("expected JAX to reject mismatched inner dimensions")


def test_batched_matmul() -> None:
    vec4 = jnp.ones(4)
    mat45 = jnp.ones((4, 5))
    batch_234 = jnp.ones((2, 3, 4))
    batch_245 = jnp.ones((2, 4, 5))

    # (k)(*batch, k, m) -> (*batch, m)
    assert_shape(jnp.linalg.matmul(vec4, batch_245), (2, 5))

    # (*batch, n, k)(k) -> (*batch, n)
    assert_shape(jnp.linalg.matmul(batch_234, vec4), (2, 3))

    # (*batch_left, n, k)(*batch_right, k, m) -> (*broadcast(batch_left, batch_right), n, m)
    assert_shape(jnp.linalg.matmul(batch_234, mat45), (2, 3, 5))
    assert_shape(jnp.linalg.matmul(batch_234, batch_245), (2, 3, 5))


def test_outer_and_cross() -> None:
    v1 = jnp.ones(3)
    v2 = jnp.ones(4)

    assert_shape(jnp.linalg.outer(v1, v2), (3, 4))
    assert_shape(jnp.linalg.cross(v1, v1), (3,))


def test_pinv_and_lstsq() -> None:
    a = jnp.ones((4, 3))
    b = jnp.ones(4)

    assert_shape(jnp.linalg.pinv(a), (3, 4))
    x, residuals, rank, s = jnp.linalg.lstsq(a, b)
    assert_shape(x, (3,))
    assert_shape(rank, ())
    assert_shape(s, (3,))


def test_norm_variations() -> None:
    vec = jnp.ones(5)
    mat = jnp.ones((3, 4))
    cube = jnp.ones((2, 3, 4))

    assert_shape(jnp.linalg.norm(vec), ())
    assert_shape(jnp.linalg.norm(mat, axis=0), (4,))
    assert_shape(jnp.linalg.norm(mat, axis=-1, keepdims=True), (3, 1))
    assert_shape(jnp.linalg.vector_norm(mat, axis=1), (3,))
    assert_shape(jnp.linalg.matrix_norm(mat), ())
    assert_shape(jnp.linalg.matrix_norm(mat, keepdims=True), (1, 1))
    assert_shape(jnp.linalg.matrix_norm(cube), (2,))
    assert_shape(jnp.linalg.matrix_norm(cube, keepdims=True), (2, 1, 1))


def generic_batched_cholesky[Batch: IntTuple, N: IntVar](
    x: jax.Array[[*Elements[Batch], N, N]],
) -> jax.Array[[*Elements[Batch], N, N]]:
    return jnp.linalg.cholesky(x)


def test_batched_linalg_operations() -> None:
    batch_eye = jnp.ones((2, 4, 4))
    batch_mat = jnp.ones((2, 4, 5))
    batch_vec = jnp.ones((2, 4))

    assert_shape(jnp.linalg.cholesky(batch_eye), (2, 4, 4))
    assert_shape(generic_batched_cholesky(batch_eye), (2, 4, 4))
    assert_shape(jnp.linalg.inv(batch_eye), (2, 4, 4))
    assert_shape(jnp.linalg.matrix_power(batch_eye, 2), (2, 4, 4))
    assert_shape(jnp.linalg.det(batch_eye), (2,))
    sign, logdet = jnp.linalg.slogdet(batch_eye)
    assert_shape(sign, (2,))
    assert_shape(logdet, (2,))
    assert_shape(jnp.linalg.matrix_rank(batch_mat), (2,))
    assert_shape(jnp.linalg.cond(batch_eye), (2,))

    w, v = jnp.linalg.eigh(batch_eye)
    assert_shape(w, (2, 4))
    assert_shape(v, (2, 4, 4))
    assert_shape(jnp.linalg.eigvalsh(batch_eye), (2, 4))

    q, r = jnp.linalg.qr(batch_mat, mode="reduced")
    assert_shape(q, (2, 4, 4))
    assert_shape(r, (2, 4, 5))

    u, s, vt = jnp.linalg.svd(batch_mat, full_matrices=False)
    assert_shape(u, (2, 4, 4))
    assert_shape(s, (2, 4))
    assert_shape(vt, (2, 4, 5))
    assert_shape(jnp.linalg.svdvals(batch_mat), (2, 4))

    vec4 = jnp.ones(4)
    assert_shape(jnp.linalg.solve(batch_eye, vec4), (2, 4))
    batch_rhs = jnp.ones((2, 4, 3))
    assert_shape(jnp.linalg.solve(batch_eye, batch_rhs), (2, 4, 3))
    assert_shape(jnp.linalg.pinv(batch_mat), (2, 5, 4))
