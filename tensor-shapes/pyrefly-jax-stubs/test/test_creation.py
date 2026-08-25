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


def test_zeros_and_ones() -> None:
    assert_shape(jnp.zeros(4), (4,))
    assert_shape(jnp.zeros((3, 4)), (3, 4))
    assert_shape(jnp.zeros((2, 3, 4)), (2, 3, 4))
    assert_shape(jnp.ones(4), (4,))
    assert_shape(jnp.ones((3, 4)), (3, 4))
    assert_shape(jnp.ones((2, 3, 4)), (2, 3, 4))


def test_shapes_outside_the_exact_ranks_are_gradual() -> None:
    # Ranks 1 through 3 given as a tuple are exact; anything else -- a longer
    # tuple, or any non-tuple sequence -- is accepted but gradual.
    assert jnp.zeros((2, 3, 4, 5)).shape == (2, 3, 4, 5)
    assert jnp.zeros([2, 3]).shape == (2, 3)
    assert jnp.ones([2, 3]).shape == (2, 3)
    assert jnp.full([2, 3], 1.0).shape == (2, 3)


def test_full() -> None:
    assert_shape(jnp.full(4, 2.0), (4,))
    assert_shape(jnp.full((3, 4), 2.0), (3, 4))
    assert_shape(jnp.full((2, 3, 4), 2.0), (2, 3, 4))


def test_arange_and_eye() -> None:
    assert_shape(jnp.arange(5), (5,))
    # JAX names the sole argument `start`, so the keyword form is valid.
    assert_shape(jnp.arange(start=5), (5,))
    assert_shape(jnp.eye(3), (3, 3))
    assert_shape(jnp.eye(2, 3), (2, 3))
    assert_shape(jnp.identity(4), (4, 4))


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
    assert_shape(jnp.zeros(4, jnp.int32), (4,))
    assert_shape(jnp.ones((3, 4), jnp.float32), (3, 4))
