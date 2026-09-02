# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import jax.numpy as jnp
from shape_extensions import assert_shape


def test_einsum() -> None:
    v1 = jnp.ones(3)
    v2 = jnp.ones(3)
    a = jnp.ones((2, 3))
    b = jnp.ones((3, 4))
    batch_a = jnp.ones((5, 2, 3))
    batch_b = jnp.ones((5, 3, 4))

    # Dot product
    assert_shape(jnp.einsum("i,i->", v1, v2), ())

    # Outer product
    assert_shape(jnp.einsum("i,j->ij", v1, v2), (3, 3))

    # Matrix multiplication
    assert_shape(jnp.einsum("ij,jk->ik", a, b), (2, 4))

    # Batch matrix multiplication
    assert_shape(jnp.einsum("bij,bjk->bik", batch_a, batch_b), (5, 2, 4))

    # Transpose
    assert_shape(jnp.einsum("ij->ji", a), (3, 2))

    # Diagonal
    assert_shape(jnp.einsum("ii->i", jnp.ones((3, 3))), (3,))

    # Trace
    assert_shape(jnp.einsum("ii->", jnp.ones((3, 3))), ())


def test_einsum_path() -> None:
    a = jnp.ones((2, 3))
    b = jnp.ones((3, 4))
    path, _ = jnp.einsum_path("ij,jk->ik", a, b)
    assert len(path) > 0
    assert_shape(jnp.einsum("ij,jk->ik", a, b), (2, 4))


def test_dot() -> None:
    v1 = jnp.ones(3)
    v2 = jnp.ones(3)
    mat23 = jnp.ones((2, 3))
    mat34 = jnp.ones((3, 4))
    tensor234 = jnp.ones((2, 3, 4))
    tensor45 = jnp.ones((4, 5))

    # 1D dot 1D -> scalar ()
    assert_shape(jnp.dot(v1, v2), ())
    assert_shape(v1.dot(v2), ())

    # 2D dot 2D -> 2D
    assert_shape(jnp.dot(mat23, mat34), (2, 4))
    assert_shape(mat23.dot(mat34), (2, 4))

    # 2D dot 1D -> 1D
    assert_shape(jnp.dot(mat23, v1), (2,))
    assert_shape(mat23.dot(v1), (2,))

    # 1D dot 2D -> 1D
    assert_shape(jnp.dot(jnp.ones(2), mat23), (3,))
    assert_shape(jnp.ones(2).dot(mat23), (3,))

    # 3D dot 2D -> 3D
    assert_shape(jnp.dot(tensor234, tensor45), (2, 3, 5))
    assert_shape(tensor234.dot(tensor45), (2, 3, 5))


def test_vdot() -> None:
    assert_shape(jnp.vdot(jnp.ones(3), jnp.ones(3)), ())
    assert_shape(jnp.vdot(jnp.ones((2, 3)), jnp.ones((2, 3))), ())


def test_inner() -> None:
    assert_shape(jnp.inner(jnp.ones(3), jnp.ones(3)), ())
    assert_shape(jnp.inner(jnp.ones((2, 3)), jnp.ones((4, 3))), (2, 4))
    assert_shape(jnp.inner(jnp.ones((2, 3, 4)), jnp.ones((5, 4))), (2, 3, 5))
    assert_shape(jnp.inner(jnp.ones((2, 3, 4)), jnp.ones((5, 6, 4))), (2, 3, 5, 6))


def test_outer() -> None:
    assert_shape(jnp.outer(jnp.ones(3), jnp.ones(4)), (3, 4))
    assert jnp.outer(jnp.ones((2, 3)), jnp.ones((4, 5))).shape == (6, 20)


def test_kron() -> None:
    assert_shape(jnp.kron(jnp.ones(2), jnp.ones(3)), (6,))
    assert_shape(jnp.kron(jnp.ones((2, 3)), jnp.ones((4, 5))), (8, 15))
    assert_shape(jnp.kron(jnp.ones(2), jnp.ones((4, 5))), (4, 10))


def test_matvec_and_vecmat() -> None:
    mat = jnp.ones((2, 3))
    vec = jnp.ones(3)
    batch_mat = jnp.ones((4, 2, 3))
    batch_vec = jnp.ones((4, 3))

    assert_shape(jnp.matvec(mat, vec), (2,))
    assert_shape(jnp.matvec(batch_mat, vec), (4, 2))
    assert_shape(jnp.matvec(batch_mat, batch_vec), (4, 2))

    assert_shape(jnp.vecmat(jnp.ones(2), mat), (3,))
    assert_shape(jnp.vecmat(jnp.ones(2), batch_mat), (4, 3))
    assert_shape(jnp.vecmat(jnp.ones((4, 2)), batch_mat), (4, 3))


def test_vecdot() -> None:
    assert_shape(jnp.vecdot(jnp.ones(3), jnp.ones(3)), ())
    assert_shape(jnp.vecdot(jnp.ones((2, 3)), jnp.ones((2, 3))), (2,))
    assert_shape(jnp.vecdot(jnp.ones((2, 3)), jnp.ones((2, 3)), axis=0), (3,))


def test_cross() -> None:
    # 3D vectors
    assert_shape(jnp.cross(jnp.ones(3), jnp.ones(3)), (3,))
    assert_shape(jnp.cross(jnp.ones((2, 3)), jnp.ones((2, 3))), (2, 3))

    # 2D vectors
    assert_shape(jnp.cross(jnp.ones(2), jnp.ones(2)), ())
    assert_shape(jnp.cross(jnp.ones((4, 2)), jnp.ones((4, 2))), (4,))

    # Mixed 2D and 3D
    assert_shape(jnp.cross(jnp.ones((4, 2)), jnp.ones((4, 3))), (4, 3))
    assert_shape(jnp.cross(jnp.ones((4, 3)), jnp.ones((4, 2))), (4, 3))

    # axis parameter
    assert_shape(jnp.cross(jnp.ones((3, 4)), jnp.ones((3, 4)), axis=0), (3, 4))
    assert_shape(jnp.cross(jnp.ones((2, 4)), jnp.ones((2, 4)), axis=0), (4,))

    # axisa, axisb, axisc
    assert_shape(jnp.cross(jnp.ones((4, 3)), jnp.ones((4, 3)), axisc=0), (3, 4))
    assert_shape(
        jnp.cross(jnp.ones((3, 4)), jnp.ones((4, 3)), axisa=0, axisb=1, axisc=0),
        (3, 4),
    )


def test_tensordot() -> None:
    assert_shape(jnp.tensordot(jnp.ones((2, 3)), jnp.ones((2, 3)), axes=2), ())
    assert_shape(jnp.tensordot(jnp.ones((2, 3)), jnp.ones((3, 4)), axes=1), (2, 4))
    assert_shape(
        jnp.tensordot(jnp.ones((2, 3, 4)), jnp.ones((3, 4, 5)), axes=2),
        (2, 5),
    )


def test_diagonal_and_trace() -> None:
    mat = jnp.ones((3, 3))
    rect = jnp.ones((4, 5))
    batch_mat = jnp.ones((2, 3, 3))

    assert_shape(jnp.diagonal(mat), (3,))
    assert_shape(jnp.diagonal(rect), (4,))
    assert_shape(mat.diagonal(), (3,))

    assert_shape(jnp.trace(mat), ())
    assert_shape(mat.trace(), ())
    assert_shape(jnp.trace(batch_mat), (3,))
