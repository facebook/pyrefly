# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import jax.numpy as jnp
from shape_extensions import assert_shape


def test_basic_indexing() -> None:
    x = jnp.ones((2, 3, 4))

    assert_shape(x[0], (3, 4))
    assert_shape(x[:, 1:], (2, 2, 4))
    assert_shape(x[..., 0], (2, 3))
    assert_shape(x[None, ...], (1, 2, 3, 4))


def test_integer_tuple_indexing() -> None:
    x = jnp.ones((2, 4, 5))

    assert_shape(x[:, (0, 2, 3), :], (2, 3, 5))


def test_indexing_scalar_is_rejected() -> None:
    x = jnp.ones(())
    assert_shape(x, ())
    try:
        x[0]  # E: Cannot index scalar tensor (rank 0)
    except IndexError:
        pass
    else:
        raise AssertionError("expected JAX to reject indexing a scalar")
