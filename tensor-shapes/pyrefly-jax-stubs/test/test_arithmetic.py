# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import jax.numpy as jnp
from shape_extensions import assert_shape


def test_elementwise_operators_preserve_shape() -> None:
    a = jnp.ones((3, 4))
    b = jnp.full((3, 4), 2.0)

    assert_shape(a + b, (3, 4))
    assert_shape(b - a, (3, 4))
    assert_shape(a * b, (3, 4))
    assert_shape(b / a, (3, 4))
    assert_shape(b**2, (3, 4))


def test_scalar_operands_preserve_shape() -> None:
    a = jnp.ones((3, 4))
    v = jnp.full(5, 2.0)

    assert_shape(a + 1.0, (3, 4))
    assert_shape(1.0 + a, (3, 4))
    assert_shape(a - 1.0, (3, 4))
    assert_shape(1.0 - a, (3, 4))
    assert_shape(a * 2.0, (3, 4))
    assert_shape(2.0 * a, (3, 4))
    assert_shape(a / 2.0, (3, 4))
    assert_shape(v**2, (5,))


def test_complex_scalars_preserve_shape() -> None:
    a = jnp.ones((3, 4))

    assert_shape(a + 1j, (3, 4))
    assert_shape(1j + a, (3, 4))
    assert_shape(a * 2j, (3, 4))
    assert_shape(jnp.add(a, 1j), (3, 4))
    assert_shape(jnp.multiply(2j, a), (3, 4))


def test_comparisons_produce_boolean_arrays() -> None:
    a = jnp.ones((3, 4))

    # These are elementwise: the result is an array, not a `bool`.
    assert_shape(a == a, (3, 4))
    assert_shape(a != jnp.ones((1, 4)), (3, 4))
    assert_shape(a > 0, (3, 4))
    assert_shape(a <= 1.0, (3, 4))
    assert_shape(a >= jnp.ones((1, 4)), (3, 4))
    assert_shape(a < jnp.ones((3, 1)), (3, 4))


def test_unary_operators_preserve_shape() -> None:
    a = jnp.full((3, 4), -1.0)

    assert_shape(-a, (3, 4))
    assert_shape(+a, (3, 4))
    assert_shape(abs(a), (3, 4))
    assert_shape(jnp.abs(a), (3, 4))
    assert_shape(jnp.negative(a), (3, 4))


def test_broadcasting_expands_singleton_dimensions() -> None:
    column = jnp.ones((3, 1))
    row = jnp.ones((1, 4))

    assert_shape(column + row, (3, 4))
    assert_shape(row * column, (3, 4))
    assert_shape(jnp.add(column, row), (3, 4))
    assert_shape(jnp.multiply(row, column), (3, 4))
    assert_shape(jnp.maximum(column, row), (3, 4))
    assert_shape(jnp.minimum(column, row), (3, 4))


def test_binary_functions_accept_scalars() -> None:
    a = jnp.ones((3, 4))

    assert_shape(jnp.add(a, 1), (3, 4))
    assert_shape(jnp.add(1, a), (3, 4))
    assert_shape(jnp.subtract(a, 1.0), (3, 4))
    assert_shape(jnp.multiply(2.0, a), (3, 4))
    assert_shape(jnp.divide(a, 2.0), (3, 4))
    assert_shape(jnp.maximum(a, 0), (3, 4))
    assert_shape(jnp.minimum(0, a), (3, 4))


def test_elementwise_unary_functions_preserve_shape() -> None:
    a = jnp.full((2, 3), 0.5)

    assert_shape(jnp.exp(a), (2, 3))
    assert_shape(jnp.log(a), (2, 3))
    assert_shape(jnp.sqrt(a), (2, 3))
    assert_shape(jnp.square(a), (2, 3))
    assert_shape(jnp.sin(a), (2, 3))
    assert_shape(jnp.cos(a), (2, 3))
    assert_shape(jnp.tanh(a), (2, 3))


def test_subtraction_rejects_incompatible_broadcast() -> None:
    a = jnp.ones((3, 4))
    b = jnp.ones(5)

    assert_shape(a - jnp.ones((3, 4)), (3, 4))
    try:
        # E: Cannot evaluate type-level shape DSL call
        a - b
    except (TypeError, ValueError):
        pass
    else:
        raise AssertionError("expected JAX to reject incompatible shapes")


def test_binary_functions_reject_incompatible_broadcast() -> None:
    a = jnp.ones((3, 4))
    b = jnp.ones(5)

    assert_shape(jnp.maximum(a, jnp.ones((3, 4))), (3, 4))
    try:
        jnp.maximum(  # E: Cannot evaluate type-level shape DSL call
            a, b
        )
    except (TypeError, ValueError):
        pass
    else:
        raise AssertionError("expected JAX to reject incompatible shapes")
