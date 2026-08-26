# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import jax
import jax.nn as jnn
import jax.numpy as jnp
from shape_extensions import assert_shape, IntVar


N = IntVar("N")
M = IntVar("M")


def generic_activation[N: IntVar, M: IntVar](x: jax.Array[[N, M]]) -> jax.Array[[N, M]]:
    """Activations are elementwise, so they preserve a symbolic shape."""

    return jnn.relu(x)


def test_elementwise_activations_preserve_shape() -> None:
    x = jnp.full((3, 4), 0.5)

    assert_shape(jnn.relu(x), (3, 4))
    assert_shape(jnn.relu(x=x), (3, 4))
    assert_shape(jnn.relu6(x), (3, 4))
    assert_shape(jnn.sigmoid(x), (3, 4))
    assert_shape(jnn.softplus(x), (3, 4))
    assert_shape(jnn.soft_sign(x), (3, 4))
    assert_shape(jnn.silu(x), (3, 4))
    assert_shape(jnn.swish(x), (3, 4))
    assert_shape(jnn.hard_tanh(x), (3, 4))


def test_parameterized_activations_preserve_shape() -> None:
    x = jnp.full((2, 5), -0.5)

    assert_shape(jnn.elu(x), (2, 5))
    assert_shape(jnn.elu(x, 0.5), (2, 5))
    assert_shape(jnn.leaky_relu(x), (2, 5))
    assert_shape(jnn.leaky_relu(x, 0.2), (2, 5))
    assert_shape(jnn.gelu(x), (2, 5))
    assert_shape(jnn.gelu(x, False), (2, 5))


def test_parameterized_activations_broadcast_array_parameters() -> None:
    x = jnp.full((3, 4), -0.5)
    parameter = jnp.full((5, 1, 4), 0.2)

    assert_shape(jnn.leaky_relu(x, parameter), (5, 3, 4))
    assert_shape(jnn.elu(x, parameter), (5, 3, 4))


def test_softmax_normalizes_without_reducing() -> None:
    x = jnp.full((3, 4), 0.25)

    assert_shape(jnn.softmax(x), (3, 4))
    assert_shape(jnn.softmax(x, 0), (3, 4))
    assert_shape(jnn.log_softmax(x), (3, 4))
    assert_shape(jnn.log_softmax(x, -1), (3, 4))
    assert_shape(jnn.softmax(x, [0]), (3, 4))


def test_activations_compose_with_matmul() -> None:
    inputs = jnp.ones((8, 16))
    weights = jnp.full((16, 4), 0.1)

    assert_shape(jnn.relu(inputs @ weights), (8, 4))
    assert_shape(jnn.softmax(jnn.relu(inputs @ weights), -1), (8, 4))
