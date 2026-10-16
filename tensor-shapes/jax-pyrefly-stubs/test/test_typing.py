# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
from typing import Any, assert_type, TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike, DTypeLike
from shape_extensions import assert_raises, assert_shape, IntTuple


def asarray(x: ArrayLike) -> jax.Array:
    return jnp.asarray(x)


def rejects_sequence(x: list[int]) -> None:
    asarray(x)  # E: not assignable to parameter `x`


def accepts_numpy_bool(x: np.bool_) -> None:
    asarray(x)


def accepts_dtype(dtype: DTypeLike) -> None:
    pass


def test_array_like() -> None:
    assert_shape(asarray(jnp.ones((2, 3))).shape, IntTuple, runtime=(2, 3))
    assert_shape(asarray(np.ones((4,))).shape, IntTuple, runtime=(4,))
    assert_shape(asarray(1).shape, IntTuple, runtime=())
    assert_shape(asarray(np.int32(1)).shape, IntTuple, runtime=())


def test_dtype_like() -> None:
    accepts_dtype("float32")
    accepts_dtype(jnp.float32)
    accepts_dtype(np.dtype("float32"))
    assert_shape(jnp.ones((2,), dtype=jnp.float32).shape, (2,))


def test_scalar_conversions() -> None:
    scalar = jnp.array(3.0)
    assert_shape(scalar.shape, ())
    assert_type(float(scalar), float)
    assert_type(int(scalar), int)
    assert_type(complex(scalar), complex)
    assert_type(bool(scalar), bool)
    assert float(scalar) == 3.0
    assert int(scalar) == 3
    assert complex(scalar) == 3.0 + 0j
    assert bool(scalar) is True
    assert math.ceil(scalar) == 3
    assert math.floor(scalar) == 3

    int_scalar = jnp.array(3)
    assert_shape(int_scalar.shape, ())
    assert_type(int_scalar.__index__(), int)
    assert int_scalar.__index__() == 3


def test_array_attributes_and_methods() -> None:
    x = jnp.ones((2, 3))
    assert_shape(x.shape, (2, 3))
    _ = x.aval
    _ = x.weak_type
    _ = x.traceback
    _ = x.format
    _ = x.devices()
    _ = x.platform()
    _ = x.is_deleted()
    _ = x.is_ready()
    _ = x.on_device_size_in_bytes()
    _ = x.unsafe_buffer_pointer()
    _ = x.tobytes()
    _ = x.tolist()
    assert_shape(x.block_until_ready().shape, (2, 3))
    assert_shape(x.clone().shape, (2, 3))
    vec = jnp.array([1.0, 2.0, 3.0])
    assert (1.0 in vec) is True
    assert (99.0 in vec) is False

    q, r = divmod(x, 2.0)
    assert_shape(q.shape, (2, 3))
    assert_shape(r.shape, (2, 3))

    rq, rr = divmod(2.0, x)
    assert_shape(rq.shape, (2, 3))
    assert_shape(rr.shape, (2, 3))

    m = jnp.ones((3, 4))
    prod = x @ m
    assert_shape(prod.shape, (2, 4))
    rprod = m.T @ x.T
    assert_shape(rprod.shape, (4, 2))


def test_invalid_operations_remain_errors() -> None:
    values = jnp.ones((2, 3))
    assert_shape(values.shape, (2, 3))
    with assert_raises(TypeError):
        values.__float__()  # E: not assignable

    with assert_raises(TypeError):
        values.__int__()  # E: not assignable

    if TYPE_CHECKING:

        def check_unknown_shape(array: jax.Array) -> None:
            float(array)
            int(array)
            complex(array)
