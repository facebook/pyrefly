# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import jax.numpy as jnp
from jax import random
from shape_extensions import assert_shape, IntTuple


def test_key_operations() -> None:
    key = random.key(0)
    assert_shape(key.shape, ())
    assert_shape(random.fold_in(key, 1).shape, ())
    assert_shape(random.split(key).shape, (2,))
    assert_shape(random.split(key, 3).shape, (3,))
    assert_shape(random.split(key, (2, 3)).shape, (2, 3))

    legacy_key = random.PRNGKey(0)
    assert_shape(legacy_key.shape, IntTuple, runtime=(2,))

    try:
        # E: Argument `Array[[2]]` is not assignable to parameter `seed`
        random.key(jnp.ones((2,), dtype=jnp.int32))
    except TypeError:
        pass
    else:
        raise AssertionError("expected JAX to reject a non-scalar seed")


def test_explicit_shape_samplers() -> None:
    key = random.key(0)
    assert_shape(random.bits(key).shape, ())
    assert_shape(random.bits(key, (2, 3)).shape, (2, 3))
    assert_shape(random.uniform(key).shape, ())
    assert_shape(random.uniform(key, (2, 3)).shape, (2, 3))
    assert_shape(random.randint(key, (2, 3), 0, 10).shape, (2, 3))
    assert_shape(random.normal(key).shape, ())
    assert_shape(random.normal(key, (2, 3)).shape, (2, 3))

    shape = [2, 3]
    assert_shape(random.normal(key, shape).shape, IntTuple, runtime=(2, 3))


def test_basic_continuous_distributions() -> None:
    key = random.key(0)
    assert_shape(random.cauchy(key).shape, ())
    assert_shape(random.cauchy(key, (2, 3)).shape, (2, 3))
    assert_shape(random.exponential(key, (2, 3)).shape, (2, 3))
    assert_shape(random.laplace(key, (2, 3)).shape, (2, 3))
    assert_shape(random.logistic(key, (2, 3)).shape, (2, 3))


def test_additional_explicit_shape_distributions() -> None:
    key = random.key(0)
    assert_shape(random.gumbel(key).shape, ())
    assert_shape(random.gumbel(key, (2, 3), mode="high").shape, (2, 3))
    assert_shape(random.maxwell(key, (2, 3)).shape, (2, 3))
    assert_shape(random.rademacher(key, (2, 3)).shape, (2, 3))


def test_single_parameter_distributions() -> None:
    key = random.key(0)
    parameter = jnp.full((2, 3), 0.5)
    assert_shape(random.bernoulli(key, parameter).shape, (2, 3))
    assert_shape(random.bernoulli(key, parameter, (4, 2, 3)).shape, (4, 2, 3))
    assert_shape(random.gamma(key, parameter).shape, (2, 3))
    assert_shape(random.gamma(key, parameter, (4, 2, 3)).shape, (4, 2, 3))
    assert_shape(random.poisson(key, parameter).shape, (2, 3))
    assert_shape(random.poisson(key, parameter, (4, 2, 3)).shape, (4, 2, 3))

    try:
        # E: Cannot broadcast dimension
        random.bernoulli(key, parameter, (4,))
    except ValueError:
        pass
    else:
        raise AssertionError("expected JAX to reject an incompatible requested shape")


def test_additional_single_parameter_distributions() -> None:
    key = random.key(0)
    parameter = jnp.ones((2, 3))
    assert_shape(random.chisquare(key, parameter).shape, (2, 3))
    assert_shape(random.chisquare(key, 1.0, (4, 2, 3)).shape, (4, 2, 3))
    assert_shape(random.loggamma(key, parameter).shape, (2, 3))
    assert_shape(random.loggamma(key, 1.0, (4, 2, 3)).shape, (4, 2, 3))
    assert_shape(random.pareto(key, parameter).shape, (2, 3))
    assert_shape(random.pareto(key, 1.0, (4, 2, 3)).shape, (4, 2, 3))
    assert_shape(random.t(key, parameter).shape, (2, 3))
    assert_shape(random.t(key, 1.0, (4, 2, 3)).shape, (4, 2, 3))


def test_two_parameter_distributions() -> None:
    key = random.key(0)
    left = jnp.ones((2, 1))
    right = jnp.ones((1, 3))
    assert_shape(random.beta(key, left, right).shape, (2, 3))
    assert_shape(random.beta(key, left, right, (4, 2, 3)).shape, (4, 2, 3))
    assert_shape(random.f(key, left, right).shape, (2, 3))
    assert_shape(random.f(key, 1.0, 2.0, (4, 2, 3)).shape, (4, 2, 3))
    assert_shape(random.binomial(key, left, right).shape, (2, 3))
    assert_shape(random.binomial(key, left, right, (4, 2, 3)).shape, (4, 2, 3))


def test_permutation() -> None:
    key = random.key(0)
    assert_shape(random.permutation(key, 5).shape, (5,))
    values = jnp.ones((2, 3, 4))
    assert_shape(random.permutation(key, values).shape, (2, 3, 4))
    assert_shape(random.permutation(key, values, axis=1).shape, (2, 3, 4))


def test_categorical() -> None:
    key = random.key(0)
    logits = jnp.ones((2, 3, 4))
    assert_shape(random.categorical(key, logits).shape, (2, 3))
    assert_shape(random.categorical(key, logits, axis=1).shape, (2, 4))
    assert_shape(random.categorical(key, logits, shape=(5, 2, 3)).shape, (5, 2, 3))


def test_choice() -> None:
    key = random.key(0)
    assert_shape(random.choice(key, 5).shape, ())
    assert_shape(random.choice(key, 5, (2, 3)).shape, (2, 3))

    values = jnp.ones((2, 3, 4))
    assert_shape(random.choice(key, values).shape, (3, 4))
    assert_shape(random.choice(key, values, (5,), axis=1).shape, (2, 5, 4))

    try:
        # E: Cannot evaluate type-level shape DSL call: axis out of bounds
        random.choice(key, values, axis=3)
    except ValueError:
        pass
    else:
        raise AssertionError("expected JAX to reject an out-of-bounds axis")
