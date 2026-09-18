# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from shape_extensions import assert_shape, Elements, IntTuple, IntVar


def square_svd_components[N: IntVar](
    x: jax.Array[[N, N]],
) -> tuple[jax.Array[[N, N]], jax.Array[[N]], jax.Array[[N, N]]]:
    return jnp.linalg.svd(x, full_matrices=False)


def test_svd_reduced_wide_matrix() -> None:
    x = jnp.ones((3, 5))

    u, s, vt = jnp.linalg.svd(x, full_matrices=False)

    assert_shape(u.shape, (3, 3))
    assert_shape(s.shape, (3,))
    assert_shape(vt.shape, (3, 5))


def test_svd_reduced_tall_matrix() -> None:
    x = jnp.ones((5, 3))

    u, s, vt = jnp.linalg.svd(x, full_matrices=False)

    assert_shape(u.shape, (5, 3))
    assert_shape(s.shape, (3,))
    assert_shape(vt.shape, (3, 3))


def test_svd_reduced_square_matrix() -> None:
    x = jnp.ones((4, 4))

    u, s, vt = jnp.linalg.svd(x, full_matrices=False)

    assert_shape(u.shape, (4, 4))
    assert_shape(s.shape, (4,))
    assert_shape(vt.shape, (4, 4))

    u_c, s_c, vt_c = square_svd_components(x)
    assert_shape(u_c.shape, (4, 4))
    assert_shape(s_c.shape, (4,))
    assert_shape(vt_c.shape, (4, 4))


def test_svd_compute_uv_false() -> None:
    x = jnp.ones((5, 3))

    s = jnp.linalg.svd(x, compute_uv=False)

    assert_shape(s.shape, (3,))
    assert_shape(jnp.linalg.svdvals(x).shape, (3,))


def test_svd_full_matrices() -> None:
    x = jnp.ones((3, 5))

    u, s, vt = jnp.linalg.svd(x, full_matrices=True)

    assert_shape(u.shape, (3, 3))
    assert_shape(s.shape, (3,))
    assert_shape(vt.shape, (5, 5))


def test_qr_reduced() -> None:
    wide = jnp.ones((3, 5))
    tall = jnp.ones((5, 3))

    q_w, r_w = jnp.linalg.qr(wide, mode="reduced")
    assert_shape(q_w.shape, (3, 3))
    assert_shape(r_w.shape, (3, 5))

    q_t, r_t = jnp.linalg.qr(tall, mode="reduced")
    assert_shape(q_t.shape, (5, 3))
    assert_shape(r_t.shape, (3, 3))


def test_qr_r_mode() -> None:
    tall = jnp.ones((5, 3))

    r = jnp.linalg.qr(tall, mode="r")
    assert_shape(r.shape, (3, 3))


def test_solve_vector_rhs() -> None:
    a = jnp.eye(3)
    b = jnp.ones(3)

    assert_shape(jnp.linalg.solve(a, b).shape, (3,))


def test_solve_matrix_rhs() -> None:
    a = jnp.eye(3)
    b = jnp.ones((3, 2))

    assert_shape(jnp.linalg.solve(a, b).shape, (3, 2))


def test_inv_and_matrix_power() -> None:
    a = jnp.eye(4)

    assert_shape(jnp.linalg.inv(a).shape, (4, 4))
    assert_shape(jnp.linalg.matrix_power(a, 3).shape, (4, 4))
    assert_shape(jnp.linalg.cholesky(a).shape, (4, 4))


def test_eigh_and_eig() -> None:
    a = jnp.eye(5)

    w_h, v_h = jnp.linalg.eigh(a)
    assert_shape(w_h.shape, (5,))
    assert_shape(v_h.shape, (5, 5))
    assert_shape(jnp.linalg.eigvalsh(a).shape, (5,))

    w, v = jnp.linalg.eig(a)
    assert_shape(w.shape, (5,))
    assert_shape(v.shape, (5, 5))
    assert_shape(jnp.linalg.eigvals(a).shape, (5,))


def test_scalar_matrix_properties() -> None:
    a = jnp.eye(3)
    rect = jnp.ones((3, 4))

    assert_shape(jnp.linalg.det(a).shape, ())
    sign, logdet = jnp.linalg.slogdet(a)
    assert_shape(sign.shape, ())
    assert_shape(logdet.shape, ())
    assert_shape(jnp.linalg.matrix_rank(rect).shape, ())
    assert_shape(jnp.linalg.cond(a).shape, ())


def test_matrix_transpose() -> None:
    x = jnp.ones((3, 4))
    batched = jnp.ones((2, 3, 4))

    assert_shape(jnp.linalg.matrix_transpose(x).shape, (4, 3))
    assert_shape(jnp.linalg.matrix_transpose(batched).shape, (2, 4, 3))


def test_matmul() -> None:
    mat23 = jnp.ones((2, 3))
    mat34 = jnp.ones((3, 4))
    vec3 = jnp.ones(3)

    assert_shape(jnp.linalg.matmul(mat23, mat34).shape, (2, 4))
    assert_shape(jnp.linalg.matmul(mat23, vec3).shape, (2,))
    assert_shape(jnp.linalg.matmul(vec3, mat34).shape, (4,))
    assert_shape(jnp.linalg.matmul(vec3, vec3).shape, ())


def test_matmul_rejects_mismatched_inner_dimension() -> None:
    a = jnp.ones((3, 4))

    assert_shape(jnp.linalg.matmul(a, jnp.ones((4, 5))).shape, (3, 5))
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
    assert_shape(jnp.linalg.matmul(vec4, batch_245).shape, (2, 5))

    # (*batch, n, k)(k) -> (*batch, n)
    assert_shape(jnp.linalg.matmul(batch_234, vec4).shape, (2, 3))

    # (*batch_left, n, k)(*batch_right, k, m) -> (*broadcast(batch_left, batch_right), n, m)
    assert_shape(jnp.linalg.matmul(batch_234, mat45).shape, (2, 3, 5))
    assert_shape(jnp.linalg.matmul(batch_234, batch_245).shape, (2, 3, 5))


def test_outer_and_cross() -> None:
    v1 = jnp.ones(3)
    v2 = jnp.ones(4)

    assert_shape(jnp.linalg.outer(v1, v2).shape, (3, 4))
    assert_shape(jnp.linalg.cross(v1, v1).shape, (3,))


def test_pinv_and_lstsq() -> None:
    a = jnp.ones((4, 3))
    b = jnp.ones(4)

    assert_shape(jnp.linalg.pinv(a).shape, (3, 4))
    x, residuals, rank, s = jnp.linalg.lstsq(a, b)
    assert_shape(x.shape, (3,))
    assert_shape(rank.shape, ())
    assert_shape(s.shape, (3,))


def test_norm_variations() -> None:
    vec = jnp.ones(5)
    mat = jnp.ones((3, 4))
    cube = jnp.ones((2, 3, 4))

    assert_shape(jnp.linalg.norm(vec).shape, ())
    assert_shape(jnp.linalg.norm(mat, axis=0).shape, (4,))
    assert_shape(jnp.linalg.norm(mat, axis=-1, keepdims=True).shape, (3, 1))
    assert_shape(jnp.linalg.vector_norm(mat, axis=1).shape, (3,))
    assert_shape(jnp.linalg.matrix_norm(mat).shape, ())
    assert_shape(jnp.linalg.matrix_norm(mat, keepdims=True).shape, (1, 1))
    assert_shape(jnp.linalg.matrix_norm(cube).shape, (2,))
    assert_shape(jnp.linalg.matrix_norm(cube, keepdims=True).shape, (2, 1, 1))


def generic_batched_cholesky[Batch: IntTuple, N: IntVar](
    x: jax.Array[[*Elements[Batch], N, N]],
) -> jax.Array[[*Elements[Batch], N, N]]:
    return jnp.linalg.cholesky(x)


def test_batched_linalg_operations() -> None:
    batch_eye = jnp.ones((2, 4, 4))
    batch_mat = jnp.ones((2, 4, 5))
    batch_vec = jnp.ones((2, 4))

    assert_shape(jnp.linalg.cholesky(batch_eye).shape, (2, 4, 4))
    assert_shape(generic_batched_cholesky(batch_eye).shape, (2, 4, 4))
    assert_shape(jnp.linalg.inv(batch_eye).shape, (2, 4, 4))
    assert_shape(jnp.linalg.matrix_power(batch_eye, 2).shape, (2, 4, 4))
    assert_shape(jnp.linalg.det(batch_eye).shape, (2,))
    sign, logdet = jnp.linalg.slogdet(batch_eye)
    assert_shape(sign.shape, (2,))
    assert_shape(logdet.shape, (2,))
    assert_shape(jnp.linalg.matrix_rank(batch_mat).shape, (2,))
    assert_shape(jnp.linalg.cond(batch_eye).shape, (2,))

    w, v = jnp.linalg.eigh(batch_eye)
    assert_shape(w.shape, (2, 4))
    assert_shape(v.shape, (2, 4, 4))
    assert_shape(jnp.linalg.eigvalsh(batch_eye).shape, (2, 4))

    q, r = jnp.linalg.qr(batch_mat, mode="reduced")
    assert_shape(q.shape, (2, 4, 4))
    assert_shape(r.shape, (2, 4, 5))

    u, s, vt = jnp.linalg.svd(batch_mat, full_matrices=False)
    assert_shape(u.shape, (2, 4, 4))
    assert_shape(s.shape, (2, 4))
    assert_shape(vt.shape, (2, 4, 5))
    assert_shape(jnp.linalg.svdvals(batch_mat).shape, (2, 4))

    vec4 = jnp.ones(4)
    assert_shape(jnp.linalg.solve(batch_eye, vec4).shape, (2, 4))
    batch_rhs = jnp.ones((2, 4, 3))
    assert_shape(jnp.linalg.solve(batch_eye, batch_rhs).shape, (2, 4, 3))
    assert_shape(jnp.linalg.pinv(batch_mat).shape, (2, 5, 4))


def test_linalg_arraylike() -> None:
    np_eye = np.ones((4, 4)) + np.eye(4)
    np_vec = np.ones((4,))
    np_mat = np.ones((3, 4))

    assert_shape(jnp.linalg.inv(np_eye).shape, (4, 4))
    assert_shape(jnp.linalg.solve(np_eye, np_vec).shape, (4,))
    assert_shape(jnp.linalg.matmul(np_mat, np_vec).shape, (3,))
    assert_shape(jnp.linalg.norm(1.0).shape, ())
    assert_shape(jnp.linalg.norm(np_vec).shape, ())
    assert_shape(jnp.linalg.det(np_eye).shape, ())
    u, s, vt = jnp.linalg.svd(np_mat, full_matrices=False)
    assert_shape(u.shape, (3, 3))
    assert_shape(s.shape, (3,))
    assert_shape(vt.shape, (3, 4))


def generic_tensorinv[N: IntVar](
    x: jax.Array[[N, N]],
) -> jax.Array[[N, N]]:
    return jnp.linalg.tensorinv(x, ind=1)


def test_tensorinv() -> None:
    # 4D tensor with ind=2: (4, 6) and (8, 3) both have prod=24
    a4 = jnp.ones((4, 6, 8, 3))
    assert_shape(jnp.linalg.tensorinv(a4, ind=2).shape, (8, 3, 4, 6))

    # 3D tensor with ind=1: (24,) and (8, 3) both have prod=24
    a3 = jnp.ones((24, 8, 3))
    assert_shape(jnp.linalg.tensorinv(a3, ind=1).shape, (8, 3, 24))

    # Default ind=2: (2, 2) and (4,) both have prod=4
    a_def = jnp.ones((2, 2, 4))
    assert_shape(jnp.linalg.tensorinv(a_def).shape, (4, 2, 2))
    assert_shape(jnp.linalg.tensorinv(a_def, 2).shape, (4, 2, 2))

    # Generic / symbolic tensor
    eye4 = jnp.eye(4)
    assert_shape(generic_tensorinv(eye4).shape, (4, 4))

    # Negative cases
    a_nonsquare = jnp.ones((4, 6))
    try:
        # E: Cannot evaluate type-level shape DSL call: tensorinv requires prod(a.shape[:ind]) == prod(a.shape[ind:])
        jnp.linalg.tensorinv(a_nonsquare, ind=2)
    except ValueError:
        pass
    else:
        raise AssertionError("expected JAX to reject non-square tensorinv")

    a_square = jnp.ones((4, 4))
    try:
        # E: Cannot evaluate type-level shape DSL call: ind must be positive
        jnp.linalg.tensorinv(a_square, ind=0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected JAX to reject ind=0 in tensorinv")

    try:
        # E: Cannot evaluate type-level shape DSL call: ind must be positive
        jnp.linalg.tensorinv(a_square, ind=-1)
    except ValueError:
        pass
    else:
        raise AssertionError("expected JAX to reject negative ind in tensorinv")


def generic_tensorsolve[N: IntVar](
    a: jax.Array[[N, N]],
    b: jax.Array[[N]],
) -> jax.Array[[N]]:
    return jnp.linalg.tensorsolve(a, b)


def test_tensorsolve() -> None:
    # 5D a and 2D b (from NumPy documentation):
    # prod(a[:2]) == prod((6, 4)) == 24, prod(a[2:]) == prod((2, 3, 4)) == 24
    a5 = jnp.ones((6, 4, 2, 3, 4))
    b2 = jnp.ones((6, 4))
    assert_shape(jnp.linalg.tensorsolve(a5, b2).shape, (2, 3, 4))

    # 3D a and 2D b (from JAX documentation):
    a3 = jnp.ones((2, 2, 4))
    b_22 = jnp.ones((2, 2))
    assert_shape(jnp.linalg.tensorsolve(a3, b_22).shape, (4,))
    assert_shape(jnp.linalg.tensorsolve(a3, b_22, axes=None).shape, (4,))

    # With axes reordering: moving axis 1 (size 4) to end results in (2, 2, 4)
    a_moved = jnp.ones((2, 4, 2))
    assert_shape(jnp.linalg.tensorsolve(a_moved, b_22, axes=(1,)).shape, (4,))

    # Generic / symbolic tensorsolve
    eye4 = jnp.eye(4)
    vec4 = jnp.ones(4)
    assert_shape(generic_tensorsolve(eye4, vec4).shape, (4,))

    # Negative cases
    try:
        # E: Cannot evaluate type-level shape DSL call: leading shape of a must match shape of b
        jnp.linalg.tensorsolve(jnp.ones((2, 3, 6)), jnp.ones((2, 2)))
    except ValueError:
        pass
    else:
        raise AssertionError(
            "expected JAX to reject leading shape mismatch in tensorsolve"
        )

    try:
        # E: Cannot evaluate type-level shape DSL call: tensorsolve requires prod(a.shape[:b.ndim]) == prod(a.shape[b.ndim:])
        jnp.linalg.tensorsolve(jnp.ones((2, 2, 5)), jnp.ones((2, 2)))
    except ValueError:
        pass
    else:
        raise AssertionError("expected JAX to reject non-square tensorsolve")

    try:
        # E: Cannot evaluate type-level shape DSL call: axis out of bounds
        jnp.linalg.tensorsolve(jnp.ones((2, 2, 4)), jnp.ones((2, 2)), axes=(5,))
    except ValueError:
        pass
    else:
        raise AssertionError("expected JAX to reject axis out of bounds in tensorsolve")

    try:
        # E: Cannot evaluate type-level shape DSL call: duplicate axis
        jnp.linalg.tensorsolve(jnp.ones((2, 2, 4)), jnp.ones((2, 2)), axes=(1, 1))
    except TypeError:
        pass
    else:
        raise AssertionError("expected JAX to reject duplicate axis in tensorsolve")


def test_tensordot() -> None:
    assert_shape(jnp.linalg.tensordot(jnp.ones((2, 3)), jnp.ones((2, 3))).shape, ())
    assert_shape(
        jnp.linalg.tensordot(jnp.ones((2, 3)), jnp.ones((2, 3)), axes=2).shape, ()
    )
    assert_shape(
        jnp.linalg.tensordot(jnp.ones((2, 3)), jnp.ones((3, 4)), axes=1).shape,
        (2, 4),
    )
    assert_shape(
        jnp.linalg.tensordot(jnp.ones((2, 3)), jnp.ones((4, 5)), axes=0).shape,
        (2, 3, 4, 5),
    )
    assert_shape(
        jnp.linalg.tensordot(jnp.ones((2, 3, 4)), jnp.ones((3, 4, 5)), axes=2).shape,
        (2, 5),
    )
    assert_shape(
        jnp.linalg.tensordot(jnp.ones((2, 3)), jnp.ones((3, 4)), axes=(1, 0)).shape,
        (2, 4),
    )
    assert_shape(
        jnp.linalg.tensordot(
            jnp.ones((2, 3, 4)), jnp.ones((4, 3, 5)), axes=(2, 0)
        ).shape,
        (2, 3, 3, 5),
    )
    assert_shape(
        jnp.linalg.tensordot(
            jnp.ones((2, 3, 4)), jnp.ones((4, 3, 5)), axes=(-1, 0)
        ).shape,
        (2, 3, 3, 5),
    )
    assert_shape(
        jnp.linalg.tensordot(
            jnp.ones((2, 3, 4)), jnp.ones((5, 4, 3)), axes=(2, -2)
        ).shape,
        (2, 3, 5, 3),
    )

    try:
        # E: Cannot evaluate type-level shape DSL call: tensordot dims exceeds input rank
        jnp.linalg.tensordot(jnp.ones((2, 3)), jnp.ones((3, 4)), axes=3)
    except TypeError:
        pass
    else:
        raise AssertionError("expected JAX to reject tensordot dims exceeds rank")

    try:
        # E: Cannot evaluate type-level shape DSL call: axis out of bounds
        jnp.linalg.tensordot(jnp.ones((2, 3)), jnp.ones((3, 4)), axes=(5, 0))
    except (TypeError, ValueError):
        pass
    else:
        raise AssertionError("expected JAX to reject axis out of bounds in tensordot")

    try:
        # E: Cannot evaluate type-level shape DSL call: tensordot contracted dimensions must match
        jnp.linalg.tensordot(jnp.ones((2, 3)), jnp.ones((4, 5)), axes=(1, 0))
    except TypeError:
        pass
    else:
        raise AssertionError(
            "expected JAX to reject contracted dimension mismatch in tensordot"
        )
