# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import jax
import jax.numpy as jnp
from shape_extensions import assert_shape, IntVar


N = IntVar("N")
M = IntVar("M")


# Only a tuple is a Flag domain, so any other sequence axis is gradual.
GRADUAL_SHAPE_RUNTIME_TESTS = {"test_non_tuple_sequence_axis_is_accepted"}


def reject_out_of_bounds_axis(x: jax.Array[[N, M]]) -> None:
    # E: Cannot evaluate type-level shape DSL call: axis out of bounds
    jnp.sum(x, axis=2)


def reject_duplicate_axis(x: jax.Array[[N, M]]) -> None:
    # E: Cannot evaluate type-level shape DSL call: duplicate axis
    jnp.sum(x, axis=(0, 0))


def test_reductions_accept_their_other_keywords() -> None:
    a = jnp.ones((3, 4))

    assert_shape(jnp.sum(a, axis=0, dtype=jnp.float32), (4,))
    assert_shape(jnp.sum(a, axis=0, where=None), (4,))
    assert_shape(a.sum(axis=0, dtype=jnp.float32), (4,))


def test_reduce_all_axes() -> None:
    a = jnp.ones((3, 4))

    assert_shape(jnp.sum(a), ())
    assert_shape(jnp.mean(a), ())
    assert_shape(jnp.max(a), ())
    assert_shape(jnp.min(a), ())
    assert_shape(jnp.prod(a), ())


def test_reduce_single_axis() -> None:
    a = jnp.ones((3, 4))

    assert_shape(jnp.sum(a, axis=0), (4,))
    assert_shape(jnp.sum(a, axis=1), (3,))
    assert_shape(jnp.mean(a, axis=0), (4,))
    assert_shape(jnp.max(a, axis=1), (3,))


def test_reduce_negative_axis() -> None:
    a = jnp.ones((3, 4))

    assert_shape(jnp.sum(a, axis=-1), (3,))
    assert_shape(jnp.sum(a, axis=-2), (4,))


def test_reduce_multiple_axes() -> None:
    a = jnp.ones((2, 3, 4))

    assert_shape(jnp.sum(a, axis=(0, 2)), (3,))
    assert_shape(jnp.mean(a, axis=(1, 2)), (2,))


def test_reduce_keepdims() -> None:
    a = jnp.ones((3, 4))

    assert_shape(jnp.sum(a, axis=1, keepdims=True), (3, 1))
    assert_shape(jnp.sum(a, axis=0, keepdims=True), (1, 4))
    assert_shape(jnp.mean(a, keepdims=True), (1, 1))


def test_reduce_methods() -> None:
    a = jnp.ones((3, 4))

    assert_shape(a.sum(), ())
    assert_shape(a.sum(axis=0), (4,))
    assert_shape(a.prod(), ())
    assert_shape(a.prod(axis=0), (4,))
    assert_shape(a.mean(axis=1), (3,))
    assert_shape(a.max(axis=1, keepdims=True), (3, 1))
    assert_shape(a.min(axis=0), (4,))


def test_non_tuple_sequence_axis_is_accepted() -> None:
    c = jnp.ones((2, 3, 4))

    assert jnp.sum(c, axis=[0, 2]).shape == (3,)
    assert jnp.sum(c, axis=range(2)).shape == (4,)
    assert c.mean(axis=[0, 2]).shape == (3,)
    assert c.mean(axis=range(2)).shape == (4,)


def test_reduce_rejects_out_of_bounds_axis() -> None:
    a = jnp.ones((3, 4))

    assert_shape(jnp.sum(a, axis=1), (3,))
    try:
        # E: Cannot evaluate type-level shape DSL call: axis out of bounds
        jnp.sum(a, axis=2)
    except ValueError:
        pass
    else:
        raise AssertionError("expected JAX to reject an out-of-bounds axis")
