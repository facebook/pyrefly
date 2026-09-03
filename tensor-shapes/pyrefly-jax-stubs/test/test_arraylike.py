# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
from shape_extensions import assert_shape, IntTuple

type JaxOrNumpyArray[Shape: IntTuple] = jax.Array[Shape] | np.ndarray[Shape, Any]


def preserve_shape[Shape: IntTuple](
    value: JaxOrNumpyArray[Shape],
) -> jax.Array[Shape]:
    return cast(Any, value)


def preserve_shapes[LeftShape: IntTuple, RightShape: IntTuple](
    left: JaxOrNumpyArray[LeftShape],
    right: JaxOrNumpyArray[RightShape],
) -> tuple[jax.Array[LeftShape], jax.Array[RightShape]]:
    return cast(Any, (left, right))


def test_union_alias_infers_shape_from_jax_array() -> None:
    assert_shape(preserve_shape(jnp.ones((2, 3))), (2, 3))


def test_union_alias_infers_shape_from_numpy_array() -> None:
    assert_shape(preserve_shape(np.ones((4, 5))), (4, 5))


def test_union_alias_infers_independent_shapes_for_mixed_inputs() -> None:
    left, right = preserve_shapes(jnp.ones((3, 1)), np.ones((1, 4)))

    assert_shape(left, (3, 1))
    assert_shape(right, (1, 4))
