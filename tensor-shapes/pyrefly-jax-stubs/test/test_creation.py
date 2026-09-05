# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import jax.numpy as jnp
from shape_extensions import assert_shape

# A multi-argument `arange` has a length the DSL cannot compute, and a shape
# outside the exact ranks is gradual, so `assert_shape` cannot be used for
# either. See `arange` and the constructors in `jax/numpy/__init__.pyi`.
GRADUAL_SHAPE_RUNTIME_TESTS = {
    "test_multi_argument_arange_length_is_gradual",
    "test_shapes_outside_the_exact_ranks_are_gradual",
}


def test_zeros_ones_and_empty() -> None:
    assert_shape(jnp.zeros(4).shape, (4,))
    assert_shape(jnp.zeros((3, 4)).shape, (3, 4))
    assert_shape(jnp.zeros((2, 3, 4)).shape, (2, 3, 4))
    assert_shape(jnp.ones(4).shape, (4,))
    assert_shape(jnp.ones((3, 4)).shape, (3, 4))
    assert_shape(jnp.ones((2, 3, 4)).shape, (2, 3, 4))
    assert_shape(jnp.empty(4).shape, (4,))
    assert_shape(jnp.empty((3, 4)).shape, (3, 4))
    assert_shape(jnp.empty((2, 3, 4)).shape, (2, 3, 4))


def test_like_constructors() -> None:
    x23 = jnp.ones((2, 3))
    x234 = jnp.ones((2, 3, 4))

    assert_shape(jnp.empty_like(x23).shape, (2, 3))
    assert_shape(jnp.empty_like(x234).shape, (2, 3, 4))
    assert_shape(jnp.empty_like(x23, shape=()).shape, ())
    assert_shape(jnp.empty_like(x23, shape=4).shape, (4,))
    assert_shape(jnp.empty_like(x23, shape=(4, 5)).shape, (4, 5))
    assert_shape(jnp.empty_like(x23, shape=(2, 3, 4, 5)).shape, (2, 3, 4, 5))

    assert_shape(jnp.zeros_like(x23).shape, (2, 3))
    assert_shape(jnp.zeros_like(x234).shape, (2, 3, 4))
    assert_shape(jnp.zeros_like(x23, shape=()).shape, ())
    assert_shape(jnp.zeros_like(x23, shape=4).shape, (4,))
    assert_shape(jnp.zeros_like(x23, shape=(4, 5)).shape, (4, 5))
    assert_shape(jnp.zeros_like(x23, shape=(2, 3, 4, 5)).shape, (2, 3, 4, 5))

    assert_shape(jnp.ones_like(x23).shape, (2, 3))
    assert_shape(jnp.ones_like(x234).shape, (2, 3, 4))
    assert_shape(jnp.ones_like(x23, shape=()).shape, ())
    assert_shape(jnp.ones_like(x23, shape=4).shape, (4,))
    assert_shape(jnp.ones_like(x23, shape=(4, 5)).shape, (4, 5))
    assert_shape(jnp.ones_like(x23, shape=(2, 3, 4, 5)).shape, (2, 3, 4, 5))

    assert_shape(jnp.full_like(x23, 7.0).shape, (2, 3))
    assert_shape(jnp.full_like(x234, 7.0).shape, (2, 3, 4))
    assert_shape(jnp.full_like(x23, 7.0, shape=()).shape, ())
    assert_shape(jnp.full_like(x23, 7.0, shape=4).shape, (4,))
    assert_shape(jnp.full_like(x23, 7.0, shape=(4, 5)).shape, (4, 5))
    assert_shape(jnp.full_like(x23, 7.0, shape=(2, 3, 4, 5)).shape, (2, 3, 4, 5))


def test_shapes_outside_the_exact_ranks_are_gradual() -> None:
    # Ranks 1 through 3 given as a tuple are exact; anything else -- a longer
    # tuple, or any non-tuple sequence -- is accepted but gradual.
    assert jnp.zeros((2, 3, 4, 5)).shape == (2, 3, 4, 5)
    assert jnp.zeros([2, 3]).shape == (2, 3)
    assert jnp.ones([2, 3]).shape == (2, 3)
    assert jnp.empty([2, 3]).shape == (2, 3)
    assert jnp.full([2, 3], 1.0).shape == (2, 3)


def test_full() -> None:
    assert_shape(jnp.full(4, 2.0).shape, (4,))
    assert_shape(jnp.full((3, 4), 2.0).shape, (3, 4))
    assert_shape(jnp.full((2, 3, 4), 2.0).shape, (2, 3, 4))


def test_arange_and_eye() -> None:
    assert_shape(jnp.arange(5).shape, (5,))
    # JAX names the sole argument `start`, so the keyword form is valid.
    assert_shape(jnp.arange(start=5).shape, (5,))
    assert_shape(jnp.eye(3).shape, (3, 3))
    assert_shape(jnp.eye(2, 3).shape, (2, 3))
    assert_shape(jnp.identity(4).shape, (4, 4))


def test_linspace_logspace_geomspace() -> None:
    assert_shape(jnp.linspace(0.0, 1.0, 10).shape, (10,))
    assert_shape(jnp.logspace(0.0, 2.0, 20).shape, (20,))
    assert_shape(jnp.geomspace(1.0, 100.0, 15).shape, (15,))


def test_diag_and_triangular() -> None:
    v4 = jnp.ones(4)
    m34 = jnp.ones((3, 4))

    # diag
    assert_shape(jnp.diag(v4).shape, (4, 4))
    assert_shape(jnp.diag(m34).shape, (3,))
    assert_shape(jnp.diagflat(v4).shape, (4, 4))

    # tri, tril, triu
    assert_shape(jnp.tri(4).shape, (4, 4))
    assert_shape(jnp.tri(3, 5).shape, (3, 5))
    assert_shape(jnp.tril(m34).shape, (3, 4))
    assert_shape(jnp.triu(m34).shape, (3, 4))


def test_vander_indices_meshgrid() -> None:
    v4 = jnp.ones(4)
    assert_shape(jnp.vander(v4).shape, (4, 4))
    assert_shape(jnp.vander(v4, 6).shape, (4, 6))

    # indices
    assert_shape(jnp.indices((3, 5)).shape, (2, 3, 5))
    assert_shape(jnp.indices((2, 3, 4)).shape, (3, 2, 3, 4))

    # meshgrid
    x = jnp.ones(3)
    y = jnp.ones(5)
    gx, gy = jnp.meshgrid(x, y)
    assert_shape(gx.shape, (5, 3))
    assert_shape(gy.shape, (5, 3))

    gx_ij, gy_ij = jnp.meshgrid(x, y, indexing="ij")
    assert_shape(gx_ij.shape, (3, 5))
    assert_shape(gy_ij.shape, (3, 5))


def test_fromfunction() -> None:
    assert_shape(jnp.fromfunction(lambda i: i, (4,)).shape, (4,))
    assert_shape(jnp.fromfunction(lambda i, j: i + j, (2, 3)).shape, (2, 3))


def test_window_functions() -> None:
    assert_shape(jnp.bartlett(10).shape, (10,))
    assert_shape(jnp.blackman(12).shape, (12,))
    assert_shape(jnp.hamming(14).shape, (14,))
    assert_shape(jnp.hanning(16).shape, (16,))
    assert_shape(jnp.kaiser(18, 5.0).shape, (18,))


def test_multi_argument_arange_length_is_gradual() -> None:
    # Statically rank-1 with an unknown length, so assert the runtime shape only.
    # The empty cases are why the length is not computed: the DSL cannot clamp a
    # negative span to zero, and claiming a negative dimension would be worse.
    assert jnp.arange(2, 7).shape == (5,)
    assert jnp.arange(5.0).shape == (5,)
    assert jnp.arange(0.0, 1.0, 0.2).shape == (5,)
    assert jnp.arange(0, 10, 2).shape == (5,)
    assert jnp.arange(10, 0, -2).shape == (5,)
    assert jnp.arange(7, 2).shape == (0,)


def test_dtype_argument_preserves_shape() -> None:
    assert_shape(jnp.zeros(4, jnp.int32).shape, (4,))
    assert_shape(jnp.ones((3, 4), jnp.float32).shape, (3, 4))


def test_array_and_asarray() -> None:
    # Python scalars
    assert_shape(jnp.array(5), ())
    assert_shape(jnp.asarray(5), ())
    assert_shape(jnp.array(2.5), ())
    assert_shape(jnp.asarray(2.5), ())
    assert_shape(jnp.array(True), ())
    assert_shape(jnp.asarray(True), ())
    assert_shape(jnp.array(1 + 2j), ())
    assert_shape(jnp.asarray(1 + 2j), ())
    assert_shape(jnp.array(5, dtype=jnp.float32), ())
    assert_shape(jnp.asarray(5, dtype=jnp.float32), ())

    # JAX Array inputs
    x0 = jnp.ones(())
    x1 = jnp.ones(4)
    x2 = jnp.ones((2, 3))
    x3 = jnp.ones((2, 3, 4))
    assert_shape(jnp.array(x0), ())
    assert_shape(jnp.asarray(x0), ())
    assert_shape(jnp.array(x1), (4,))
    assert_shape(jnp.asarray(x1), (4,))
    assert_shape(jnp.array(x2), (2, 3))
    assert_shape(jnp.asarray(x2), (2, 3))
    assert_shape(jnp.array(x3), (2, 3, 4))
    assert_shape(jnp.asarray(x3), (2, 3, 4))

    # Generic inputs
    assert jnp.array([1, 2, 3]).shape == (3,)
    assert jnp.asarray([1, 2, 3]).shape == (3,)
    assert jnp.array([[1, 2], [3, 4]]).shape == (2, 2)
    assert jnp.asarray([[1, 2], [3, 4]]).shape == (2, 2)
