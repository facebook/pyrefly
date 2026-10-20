# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, assert_type, TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax import Array
from jax.stages import Compiled, Lowered, Traced
from shape_extensions import assert_raises, assert_shape, IntTuple, IntVar


def test_jit_direct_call() -> None:
    def add(x: Array[[2, 3]], y: Array[[2, 3]]) -> Array[[2, 3]]:
        return x + y

    jitted_add = jax.jit(add)
    x = jnp.ones((2, 3))
    y = jnp.ones((2, 3))

    res = jitted_add(x, y)
    assert_type(res, Array[[2, 3]])
    assert_shape(res.shape, (2, 3))

    # Test lower and compile
    lowered = jitted_add.lower(x, y)
    assert_type(lowered, Lowered)
    if not TYPE_CHECKING:
        assert isinstance(lowered.as_text(), str)

    compiled = lowered.compile()
    assert_type(compiled, Compiled)
    compiled_res = compiled(x, y)
    assert_type(compiled_res, Any)
    assert_shape(compiled_res.shape, IntTuple, runtime=(2, 3))

    # Test trace
    traced = jitted_add.trace(x, y)
    assert_type(traced, Traced)
    traced_lowered = traced.lower()
    assert_type(traced_lowered, Lowered)


def test_jit_decorator_without_args() -> None:
    @jax.jit
    def multiply(x: Array[[4, 5]], c: float) -> Array[[4, 5]]:
        return x * c

    x = jnp.ones((4, 5))
    res = multiply(x, 2.0)
    assert_type(res, Array[[4, 5]])
    assert_shape(res.shape, (4, 5))

    lowered = multiply.lower(x, 2.0)
    assert_type(lowered, Lowered)

    traced = multiply.trace(x, 2.0)
    assert_type(traced, Traced)


def test_jit_decorator_factory() -> None:
    @jax.jit(static_argnums=1)
    def scale(x: Array[[3, 4]], factor: int) -> Array[[3, 4]]:
        return x * factor

    x = jnp.ones((3, 4))
    res = scale(x, 2)
    assert_type(res, Array[[3, 4]])
    assert_shape(res.shape, (3, 4))

    lowered = scale.lower(x, 2)
    assert_type(lowered, Lowered)

    traced = scale.trace(x, 2)
    assert_type(traced, Traced)


def test_jit_decorator_factory_with_static_argnames() -> None:
    @jax.jit(static_argnames="mode")
    def process(x: Array[[2, 2]], *, mode: str = "fast") -> Array[[2, 2]]:
        return x

    x = jnp.ones((2, 2))
    res = process(x, mode="slow")
    assert_type(res, Array[[2, 2]])
    assert_shape(res.shape, (2, 2))

    lowered = process.lower(x, mode="slow")
    assert_type(lowered, Lowered)

    traced = process.trace(x, mode="slow")
    assert_type(traced, Traced)


def test_jit_call_with_options() -> None:
    def fn(x: Array[[5]], n: int) -> Array[[5]]:
        return x + n

    jitted_fn = jax.jit(fn, static_argnums=1)
    x = jnp.ones((5,))
    res = jitted_fn(x, 3)
    assert_type(res, Array[[5]])
    assert_shape(res.shape, (5,))

    lowered = jitted_fn.lower(x, 3)
    assert_type(lowered, Lowered)


def test_jit_separate_factory_application() -> None:
    factory = jax.jit(static_argnums=(1,))

    def subtract(x: Array[[2, 3]], offset: int) -> Array[[2, 3]]:
        return x - offset

    jitted_sub = factory(subtract)
    x = jnp.ones((2, 3))
    res = jitted_sub(x, 1)
    assert_type(res, Array[[2, 3]])
    assert_shape(res.shape, (2, 3))

    lowered = jitted_sub.lower(x, 1)
    assert_type(lowered, Lowered)


def check_jit_symbolic[N: IntVar, M: IntVar](x: Array[[N, M]]) -> None:
    @jax.jit
    def identity(a: Array[[N, M]]) -> Array[[N, M]]:
        return a

    res = identity(x)
    assert_type(res, Array[[N, M]])

    lowered = identity.lower(x)
    assert_type(lowered, Lowered)


def test_jit_negative_cases() -> None:
    def add(x: Array[[2, 3]], y: Array[[2, 3]]) -> Array[[2, 3]]:
        return x + y

    jitted_add = jax.jit(add)
    x = jnp.ones((2, 3))

    with assert_raises(TypeError):
        jitted_add(x)  # E: Missing argument `y`

    with assert_raises(TypeError):
        jitted_add.lower(x)  # E: Missing argument `y`

    with assert_raises(TypeError):
        jitted_add.trace(x)  # E: Missing argument `y`

    with assert_raises(TypeError):
        jitted_add.lower(x, "bad")  # E: not assignable to parameter `y`

    with assert_raises(TypeError):
        jitted_add.trace(x, "bad")  # E: not assignable to parameter `y`
