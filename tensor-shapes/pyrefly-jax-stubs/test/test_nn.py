# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import jax
import jax.nn as jnn
import jax.numpy as jnp
from shape_extensions import assert_shape, Int, IntVar

N = IntVar("N")
M = IntVar("M")


def generic_activation[N: IntVar, M: IntVar](x: jax.Array[[N, M]]) -> jax.Array[[N, M]]:
    """Activations are elementwise, so they preserve a symbolic shape."""

    return jnn.relu(x)


def test_elementwise_activations_preserve_shape() -> None:
    x = jnp.full((3, 4), 0.5)

    assert_shape(jnn.relu(x).shape, (3, 4))
    assert_shape(jnn.relu(x=x).shape, (3, 4))
    assert_shape(jnn.relu6(x).shape, (3, 4))
    assert_shape(jnn.sigmoid(x).shape, (3, 4))
    assert_shape(jnn.softplus(x).shape, (3, 4))
    assert_shape(jnn.soft_sign(x).shape, (3, 4))
    assert_shape(jnn.silu(x).shape, (3, 4))
    assert_shape(jnn.swish(x).shape, (3, 4))
    assert_shape(jnn.hard_tanh(x).shape, (3, 4))


def test_parameterized_activations_preserve_shape() -> None:
    x = jnp.full((2, 5), -0.5)

    assert_shape(jnn.elu(x).shape, (2, 5))
    assert_shape(jnn.elu(x, 0.5).shape, (2, 5))
    assert_shape(jnn.leaky_relu(x).shape, (2, 5))
    assert_shape(jnn.leaky_relu(x, 0.2).shape, (2, 5))
    assert_shape(jnn.gelu(x).shape, (2, 5))
    assert_shape(jnn.gelu(x, False).shape, (2, 5))


def test_parameterized_activations_broadcast_array_parameters() -> None:
    x = jnp.full((3, 4), -0.5)
    parameter = jnp.full((5, 1, 4), 0.2)

    assert_shape(jnn.leaky_relu(x, parameter).shape, (5, 3, 4))
    assert_shape(jnn.elu(x, parameter).shape, (5, 3, 4))


def test_softmax_normalizes_without_reducing() -> None:
    x = jnp.full((3, 4), 0.25)

    assert_shape(jnn.softmax(x).shape, (3, 4))
    assert_shape(jnn.softmax(x, 0).shape, (3, 4))
    assert_shape(jnn.log_softmax(x).shape, (3, 4))
    assert_shape(jnn.log_softmax(x, -1).shape, (3, 4))
    assert_shape(jnn.softmax(x, [0]).shape, (3, 4))


def test_activations_compose_with_matmul() -> None:
    inputs = jnp.ones((8, 16))
    weights = jnp.full((16, 4), 0.1)

    assert_shape(jnn.relu(inputs @ weights).shape, (8, 4))
    assert_shape(jnn.softmax(jnn.relu(inputs @ weights), -1).shape, (8, 4))


def test_activations_accept_scalars() -> None:
    assert_shape(jnn.relu(0.5).shape, ())
    assert_shape(jnn.gelu(1.0).shape, ())
    assert_shape(jnn.elu(-1.0, 0.5).shape, ())
    assert_shape(jnn.leaky_relu(-1.0, 0.2).shape, ())


def generic_one_hot[N: IntVar, K: IntVar](
    x: jax.Array[[N]], k: Int[K]
) -> jax.Array[[N, K]]:
    return jnn.one_hot(x, k)


def test_one_hot() -> None:
    x_1d = jnp.full((3,), 1)
    assert_shape(jnn.one_hot(x_1d, 5).shape, (3, 5))
    assert_shape(jnn.one_hot(x_1d, 5, axis=0).shape, (5, 3))
    assert_shape(jnn.one_hot(x_1d, 5, axis=-1).shape, (3, 5))

    x_2d = jnp.full((3, 4), 2)
    assert_shape(jnn.one_hot(x_2d, 10).shape, (3, 4, 10))
    assert_shape(jnn.one_hot(x_2d, 10, axis=0).shape, (10, 3, 4))
    assert_shape(jnn.one_hot(x_2d, 10, axis=1).shape, (3, 10, 4))
    assert_shape(jnn.one_hot(x_2d, 10, axis=-1).shape, (3, 4, 10))

    assert_shape(jnn.one_hot(1, 5).shape, (5,))
    assert_shape(jnn.one_hot(1, 5, axis=0).shape, (5,))
    assert_shape(jnn.one_hot(1, 5, axis=-1).shape, (5,))


def test_more_elementwise_activations_preserve_shape() -> None:
    x = jnp.full((3, 4), 0.5)

    assert_shape(jnn.identity(x).shape, (3, 4))
    assert_shape(jnn.sparse_plus(x).shape, (3, 4))
    assert_shape(jnn.sparse_sigmoid(x).shape, (3, 4))
    assert_shape(jnn.log_sigmoid(x).shape, (3, 4))
    assert_shape(jnn.hard_sigmoid(x).shape, (3, 4))
    assert_shape(jnn.hard_silu(x).shape, (3, 4))
    assert_shape(jnn.hard_swish(x).shape, (3, 4))
    assert_shape(jnn.selu(x).shape, (3, 4))
    assert_shape(jnn.mish(x).shape, (3, 4))
    assert_shape(jnn.tanh(x).shape, (3, 4))
    assert_shape(jnn.log1mexp(x).shape, (3, 4))

    assert_shape(jnn.identity(0.5).shape, ())
    assert_shape(jnn.tanh(0.5).shape, ())
    assert_shape(jnn.selu(0.5).shape, ())
    assert_shape(jnn.mish(0.5).shape, ())


def test_more_parameterized_activations_preserve_shape() -> None:
    x = jnp.full((2, 5), -0.5)

    assert_shape(jnn.celu(x).shape, (2, 5))
    assert_shape(jnn.celu(x, 0.5).shape, (2, 5))
    assert_shape(jnn.squareplus(x).shape, (2, 5))
    assert_shape(jnn.squareplus(x, 2.0).shape, (2, 5))


def test_more_parameterized_activations_broadcast_parameters() -> None:
    x = jnp.full((3, 4), -0.5)
    parameter = jnp.full((5, 1, 4), 0.2)

    assert_shape(jnn.celu(x, parameter).shape, (5, 3, 4))
    assert_shape(jnn.squareplus(x, parameter).shape, (5, 3, 4))


def test_glu_splits_axis() -> None:
    x = jnp.ones((4, 6))

    assert_shape(jnn.glu(x).shape, (4, 3))
    assert_shape(jnn.glu(x, -1).shape, (4, 3))
    assert_shape(jnn.glu(x, 0).shape, (2, 6))
    assert_shape(jnn.glu(x, axis=1).shape, (4, 3))


def test_standardize_normalizes_without_reducing() -> None:
    x = jnp.full((3, 4), 0.25)

    assert_shape(jnn.standardize(x).shape, (3, 4))
    assert_shape(jnn.standardize(x, 0).shape, (3, 4))
    assert_shape(jnn.standardize(x, -1).shape, (3, 4))
    assert_shape(jnn.standardize(x, [0]).shape, (3, 4))


def test_logsumexp_and_logmeanexp() -> None:
    x = jnp.full((3, 4), 0.5)

    assert_shape(jnn.logsumexp(x).shape, ())
    assert_shape(jnn.logsumexp(x, 0).shape, (4,))
    assert_shape(jnn.logsumexp(x, -1).shape, (3,))
    assert_shape(jnn.logsumexp(x, 0, keepdims=True).shape, (1, 4))

    res, sign = jnn.logsumexp(x, 0, return_sign=True)
    assert_shape(res.shape, (4,))
    assert_shape(sign.shape, (4,))

    assert_shape(jnn.logmeanexp(x).shape, ())
    assert_shape(jnn.logmeanexp(x, 0).shape, (4,))
    assert_shape(jnn.logmeanexp(x, -1).shape, (3,))
    assert_shape(jnn.logmeanexp(x, 0, keepdims=True).shape, (1, 4))


def test_dot_product_attention() -> None:
    q = jnp.ones((2, 8, 4, 16))
    k = jnp.ones((2, 8, 4, 16))
    v = jnp.ones((2, 8, 4, 16))

    assert_shape(jnn.dot_product_attention(q, k, v).shape, (2, 8, 4, 16))
    out, res = jnn.dot_product_attention(q, k, v, return_residual=True)
    assert_shape(out.shape, (2, 8, 4, 16))


def test_scaled_matmul() -> None:
    lhs = jnp.ones((2, 4, 8))
    rhs = jnp.ones((2, 6, 8))
    lhs_s = jnp.ones((2, 4, 1))
    rhs_s = jnp.ones((2, 6, 1))

    assert_shape(jnn.scaled_matmul(lhs, rhs, lhs_s, rhs_s).shape, (2, 4, 6))


def test_initializers() -> None:
    key = jax.random.key(0)

    assert_shape(jnn.initializers.zeros(key, (3, 4)).shape, (3, 4))
    assert_shape(jnn.initializers.ones(key, (2, 5)).shape, (2, 5))
    init = jnn.initializers.normal()
    assert_shape(init(key, (4, 4), jnp.float32).shape, (4, 4))
