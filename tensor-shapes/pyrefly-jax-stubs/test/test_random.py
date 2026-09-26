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
