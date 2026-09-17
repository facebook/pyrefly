# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import jax.numpy as jnp
import numpy as np
from einops import einsum, rearrange, reduce, repeat
from shape_extensions import assert_shape, IntTuple
from torch import ones


def test_pattern_operations() -> None:
    image = ones((2, 3, 5, 7))
    assert_shape(rearrange(image, "b c h w -> b h w c").shape, (2, 5, 7, 3))
    assert_shape(reduce(np.ones((2, 3)), "row col -> row", "mean").shape, (2,))
    assert_shape(
        repeat(jnp.ones((2, 3)), "row col -> row col 2").shape,
        (2, 3, 2),
    )
    assert_shape(
        repeat(ones((2, 3)), "b c -> b c copies", copies=4).shape,
        IntTuple,
        runtime=(2, 3, 4),
    )


def test_einsum() -> None:
    left = ones((2, 3, 5))
    right = ones((2, 5, 7))
    assert_shape(
        einsum(
            left,
            right,
            "batch row inner, batch inner col -> batch row col",
        ).shape,
        (2, 3, 7),
    )
    assert_shape(
        einsum(np.ones((3, 3)), "row row ->").shape,
        (),
    )
    assert_shape(
        einsum(
            jnp.ones((3,)),
            jnp.ones((5,)),
            "row, col -> row col",
        ).shape,
        (3, 5),
    )
    assert_shape(
        einsum(
            ones((2,)),
            ones((2,)),
            ones((2,)),
            ones((2,)),
            ones((2,)),
            "i, i, i, i, i -> i",
        ).shape,
        IntTuple,
        runtime=(2,),
    )
