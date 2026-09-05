# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type, reveal_type, TYPE_CHECKING

import jax
import jax.numpy as jnp
from shape_extensions import assert_shape, IntVar

N = IntVar("N")
M = IntVar("M")

# The variadic method spelling produces a gradual static shape because its
# argument list cannot be captured as a `Flag`.
GRADUAL_SHAPE_RUNTIME_TESTS = {
    "test_reshape_accepts_the_variadic_method_spelling",
    "test_reshape_accepts_a_sequence_shape",
    # `assert_shape` cannot express a zero dimension, so this test pins its
    # precise static result with `reveal_type` instead.
    "test_reshape_zero_size_placeholder",
}


def reject_negative_size(x: jax.Array[[N, M]]) -> None:
    # E: Cannot evaluate type-level shape DSL call: reshape sizes must be -1 or non-negative
    jnp.reshape(x, (2, -5))


def test_reshape_to_explicit_shape() -> None:
    a = jnp.ones((3, 4))

    assert_shape(jnp.reshape(a, (2, 6)).shape, (2, 6))
    assert_shape(jnp.reshape(a, (12,)).shape, (12,))
    assert_shape(jnp.reshape(a, 12).shape, (12,))
    assert_shape(jnp.reshape(a, (2, 3, 2)).shape, (2, 3, 2))


def test_reshape_method() -> None:
    a = jnp.ones((2, 3, 4))

    assert_shape(a.reshape((6, 4)).shape, (6, 4))
    assert_shape(a.reshape(24).shape, (24,))


def test_reshape_infers_placeholder_dimension() -> None:
    a = jnp.ones((3, 4))

    assert_shape(jnp.reshape(a, (2, -1)), (2, 6))
    assert_shape(jnp.reshape(a, -1), (12,))
    assert_shape(a.reshape((3, -1)), (3, 4))
    assert_shape(a.reshape(-1), (12,))


def test_reshape_accepts_the_variadic_method_spelling() -> None:
    a = jnp.ones((3, 4))

    # Only the method is variadic; `jnp.reshape(a, 2, 6)` is an error in JAX
    # itself. The static shape is gradual, so assert the runtime shape only.
    assert a.reshape(2, 6).shape == (2, 6)
    assert a.reshape(2, 2, 3).shape == (2, 2, 3)


def test_reshape_accepts_keywords_where_jax_does() -> None:
    a = jnp.ones((3, 4))

    # The free function names both parameters and takes `order`; the method
    # takes only `order`, since its shape arguments are variadic.
    assert_shape(jnp.reshape(a=a, shape=(6, 2)).shape, (6, 2))
    assert_shape(jnp.reshape(a, (6, 2), order="F").shape, (6, 2))
    assert_shape(a.reshape((6, 2), order="C").shape, (6, 2))
    assert_shape(jnp.reshape(a, (6, 2), copy=True).shape, (6, 2))
    assert_shape(a.reshape((6, 2), out_sharding=None).shape, (6, 2))


def test_reshape_accepts_a_sequence_shape() -> None:
    a = jnp.ones((3, 4))

    # A list is not a Flag domain, so these are gradual like the other
    # sequence-valued parameters in the package.
    assert jnp.reshape(a, [2, 6]).shape == (2, 6)
    assert a.reshape([2, 6]).shape == (2, 6)


def test_reshape_method_is_positional_only() -> None:
    a = jnp.ones((3, 4))

    assert_shape(a.reshape((2, 6)).shape, (2, 6))
    try:
        a.reshape(shape=(2, 6))  # E: Unexpected keyword argument `shape`
    except TypeError:
        pass
    else:
        raise AssertionError("expected JAX to reject a `shape` keyword")


def test_reshape_rejects_multiple_placeholders() -> None:
    a = jnp.ones((3, 4))

    assert_shape(jnp.reshape(a, (2, 6)).shape, (2, 6))
    try:
        # E: Cannot evaluate type-level shape DSL call: reshape accepts at most one -1
        jnp.reshape(a, (2, -1, -1))
    except TypeError:
        pass
    else:
        raise AssertionError("expected JAX to reject more than one -1")


def test_reshape_rejects_incompatible_element_count() -> None:
    a = jnp.ones((3, 4))

    assert_shape(jnp.reshape(a, (4, 3)).shape, (4, 3))
    try:
        # E: reshape target element count does not match the input
        jnp.reshape(a, (5, 5))
    except TypeError:
        pass
    else:
        raise AssertionError("expected JAX to reject a changed element count")

    try:
        # E: reshape target element count does not match the input
        a.reshape((5, 5))
    except TypeError:
        pass
    else:
        raise AssertionError("expected JAX method to reject a changed element count")


def test_reshape_rejects_non_integral_placeholder() -> None:
    a = jnp.ones((3, 4))

    assert_shape(jnp.reshape(a, (2, -1)), (2, 6))
    try:
        # E: could not infer size for dimension -1
        jnp.reshape(a, (5, -1))
    except TypeError:
        pass
    else:
        raise AssertionError("expected JAX to reject a non-integral inferred size")

    try:
        # E: could not infer size for dimension -1
        a.reshape((5, -1))
    except TypeError:
        pass
    else:
        raise AssertionError(
            "expected JAX method to reject a non-integral inferred size"
        )


def test_reshape_zero_size_placeholder() -> None:
    empty = jnp.ones((0,))
    ambiguous = jnp.ones((0, 2))

    assert jnp.reshape(empty, (-1,)).shape == (0,)
    assert empty.reshape((-1,)).shape == (0,)
    if TYPE_CHECKING:
        reveal_type(jnp.reshape(empty, (-1,)))  # E: revealed type: Array[[0]]
        reveal_type(empty.reshape((-1,)))  # E: revealed type: Array[[0]]

    try:
        # E: could not infer size for dimension -1
        jnp.reshape(ambiguous, (0, -1))
    except ZeroDivisionError:
        pass
    else:
        raise AssertionError("expected JAX to reject ambiguous zero-size inference")

    try:
        # E: could not infer size for dimension -1
        ambiguous.reshape((0, -1))
    except ZeroDivisionError:
        pass
    else:
        raise AssertionError(
            "expected JAX method to reject ambiguous zero-size inference"
        )


def test_atleast_1d() -> None:
    # Python scalars
    assert_shape(jnp.atleast_1d(5), (1,))
    assert_shape(jnp.atleast_1d(2.5), (1,))
    assert_shape(jnp.atleast_1d(True), (1,))
    assert_shape(jnp.atleast_1d(1 + 2j), (1,))

    # Arrays
    assert_shape(jnp.atleast_1d(jnp.ones(())), (1,))
    assert_shape(jnp.atleast_1d(jnp.ones(4)), (4,))
    assert_shape(jnp.atleast_1d(jnp.ones((2, 3))), (2, 3))
    assert_shape(jnp.atleast_1d(jnp.ones((2, 3, 4))), (2, 3, 4))

    # Zero arguments: returns list of arrays
    res0 = jnp.atleast_1d()
    assert_type(res0, list[jax.Array])
    assert res0 == []

    # Two arguments: returns list of arrays
    res2 = jnp.atleast_1d(1, jnp.ones((2, 3)))
    assert_type(res2, list[jax.Array])
    assert isinstance(res2, list) and len(res2) == 2
    assert all(isinstance(x, jax.Array) for x in res2)
    assert res2[0].shape == (1,)
    assert res2[1].shape == (2, 3)


def test_atleast_2d() -> None:
    # Python scalars
    assert_shape(jnp.atleast_2d(5), (1, 1))
    assert_shape(jnp.atleast_2d(2.5), (1, 1))
    assert_shape(jnp.atleast_2d(True), (1, 1))
    assert_shape(jnp.atleast_2d(1 + 2j), (1, 1))

    # Arrays
    assert_shape(jnp.atleast_2d(jnp.ones(())), (1, 1))
    assert_shape(jnp.atleast_2d(jnp.ones(4)), (1, 4))
    assert_shape(jnp.atleast_2d(jnp.ones((2, 3))), (2, 3))
    assert_shape(jnp.atleast_2d(jnp.ones((2, 3, 4))), (2, 3, 4))

    # Zero arguments: returns list of arrays
    res0 = jnp.atleast_2d()
    assert_type(res0, list[jax.Array])
    assert res0 == []

    # Two arguments: returns list of arrays
    res2 = jnp.atleast_2d(jnp.ones(3), jnp.ones((2, 3)))
    assert_type(res2, list[jax.Array])
    assert isinstance(res2, list) and len(res2) == 2
    assert all(isinstance(x, jax.Array) for x in res2)
    assert res2[0].shape == (1, 3)
    assert res2[1].shape == (2, 3)


def test_atleast_3d() -> None:
    # Python scalars
    assert_shape(jnp.atleast_3d(5), (1, 1, 1))
    assert_shape(jnp.atleast_3d(2.5), (1, 1, 1))
    assert_shape(jnp.atleast_3d(True), (1, 1, 1))
    assert_shape(jnp.atleast_3d(1 + 2j), (1, 1, 1))

    # Arrays
    assert_shape(jnp.atleast_3d(jnp.ones(())), (1, 1, 1))
    assert_shape(jnp.atleast_3d(jnp.ones(4)), (1, 4, 1))
    assert_shape(jnp.atleast_3d(jnp.ones((2, 3))), (2, 3, 1))
    assert_shape(jnp.atleast_3d(jnp.ones((2, 3, 4))), (2, 3, 4))
    assert_shape(jnp.atleast_3d(jnp.ones((2, 3, 4, 5))), (2, 3, 4, 5))

    # Zero arguments: returns list of arrays
    res0 = jnp.atleast_3d()
    assert_type(res0, list[jax.Array])
    assert res0 == []

    # Two arguments: returns list of arrays
    res2 = jnp.atleast_3d(jnp.ones(3), jnp.ones((2, 3)))
    assert_type(res2, list[jax.Array])
    assert isinstance(res2, list) and len(res2) == 2
    assert all(isinstance(x, jax.Array) for x in res2)
    assert res2[0].shape == (1, 3, 1)
    assert res2[1].shape == (2, 3, 1)
