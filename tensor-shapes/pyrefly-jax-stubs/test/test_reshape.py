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

# A `-1` placeholder and the variadic method spelling both produce a gradual
# static shape, so `assert_shape` cannot be used for them. See `reshape_shape`
# in `jax/_shapes.pyi` and `Array.reshape` in `jax/_array.pyi`.
GRADUAL_SHAPE_RUNTIME_TESTS = {
    "test_reshape_infers_placeholder_dimension",
    "test_reshape_accepts_the_variadic_method_spelling",
    "test_reshape_accepts_a_sequence_shape",
}


def reject_negative_size(x: jax.Array[[N, M]]) -> None:
    # E: Cannot evaluate type-level shape DSL call: reshape sizes must be -1 or non-negative
    jnp.reshape(x, (2, -5))


def test_reshape_to_explicit_shape() -> None:
    a = jnp.ones((3, 4))

    assert_shape(jnp.reshape(a, (2, 6)), (2, 6))
    assert_shape(jnp.reshape(a, (12,)), (12,))
    assert_shape(jnp.reshape(a, 12), (12,))
    assert_shape(jnp.reshape(a, (2, 3, 2)), (2, 3, 2))


def test_reshape_method() -> None:
    a = jnp.ones((2, 3, 4))

    assert_shape(a.reshape((6, 4)), (6, 4))
    assert_shape(a.reshape(24), (24,))


def test_reshape_infers_placeholder_dimension() -> None:
    a = jnp.ones((3, 4))

    # Statically gradual, so only the runtime shape is asserted here.
    assert jnp.reshape(a, (2, -1)).shape == (2, 6)
    assert a.reshape(-1).shape == (12,)


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
    assert_shape(jnp.reshape(a=a, shape=(6, 2)), (6, 2))
    assert_shape(jnp.reshape(a, (6, 2), order="F"), (6, 2))
    assert_shape(a.reshape((6, 2), order="C"), (6, 2))
    assert_shape(jnp.reshape(a, (6, 2), copy=True), (6, 2))
    assert_shape(a.reshape((6, 2), out_sharding=None), (6, 2))


def test_reshape_accepts_a_sequence_shape() -> None:
    a = jnp.ones((3, 4))

    # A list is not a Flag domain, so these are gradual like the other
    # sequence-valued parameters in the package.
    assert jnp.reshape(a, [2, 6]).shape == (2, 6)
    assert a.reshape([2, 6]).shape == (2, 6)


def test_reshape_method_is_positional_only() -> None:
    a = jnp.ones((3, 4))

    assert_shape(a.reshape((2, 6)), (2, 6))
    try:
        a.reshape(shape=(2, 6))  # E: Unexpected keyword argument `shape`
    except TypeError:
        pass
    else:
        raise AssertionError("expected JAX to reject a `shape` keyword")


def test_reshape_rejects_multiple_placeholders() -> None:
    a = jnp.ones((3, 4))

    assert_shape(jnp.reshape(a, (2, 6)), (2, 6))
    try:
        # E: Cannot evaluate type-level shape DSL call: reshape accepts at most one -1
        jnp.reshape(a, (2, -1, -1))
    except TypeError:
        pass
    else:
        raise AssertionError("expected JAX to reject more than one -1")


def test_reshape_rejects_incompatible_element_count() -> None:
    a = jnp.ones((3, 4))

    assert_shape(jnp.reshape(a, (4, 3)), (4, 3))
    # TODO(stroxler): Reject this statically as well. `reshape_shape` cannot
    # compare element counts until the type-level DSL exposes a product
    # intrinsic, so for now only the runtime half rejects it.
    try:
        jnp.reshape(a, (5, 5))
    except TypeError:
        pass
    else:
        raise AssertionError("expected JAX to reject a changed element count")
