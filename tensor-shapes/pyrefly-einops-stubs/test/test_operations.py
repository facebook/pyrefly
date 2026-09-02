# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import assert_type, TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
from einops import (
    __version__,
    asnumpy,
    EinopsError,
    einsum,
    pack,
    parse_shape,
    rearrange,
    reduce,
    repeat,
    unpack,
)
from shape_extensions import assert_shape, IntTuple
from torch import ones, Tensor


def test_pattern_operations() -> None:
    image = ones((2, 3, 5, 7))
    rearranged = rearrange(image, "b c h w -> b h w c")
    assert_type(rearranged, Tensor[[2, 5, 7, 3]])
    assert_shape(rearranged.shape, (2, 5, 7, 3))
    reduced = reduce(ones((2, 3)), "row col -> row", "mean")
    assert_type(reduced, Tensor[[2]])
    assert_shape(reduced.shape, (2,))
    repeated = repeat(ones((2, 3)), "row col -> row col 2")
    assert_type(repeated, Tensor[[2, 3, 2]])
    assert_shape(
        repeated.shape,
        (2, 3, 2),
    )
    assert_shape(
        repeat(ones((2, 3)), "b c -> b c copies", copies=4).shape,
        IntTuple,
        runtime=(2, 3, 4),
    )
    if TYPE_CHECKING:
        rearrange(  # E: named axes must appear on both sides of the pattern
            image, "b c h w -> b c h missing"
        )
    else:
        try:
            rearrange(image, "b c h w -> b c h missing")
        except EinopsError:
            pass
        else:
            raise AssertionError("invalid rearrange pattern should raise EinopsError")


def test_einsum() -> None:
    left = ones((2, 3, 5))
    right = ones((2, 5, 7))
    product = einsum(
        left,
        right,
        "batch row inner, batch inner col -> batch row col",
    )
    assert_type(product, Tensor[[2, 3, 7]])
    assert_shape(product.shape, (2, 3, 7))
    assert_type(einsum(ones((3, 3)), "row row ->"), Tensor[[]])
    assert_type(einsum(ones((3,)), ones((5,)), "row, col -> row col"), Tensor[[3, 5]])
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


def test_other_backends_at_runtime() -> None:
    if not TYPE_CHECKING:
        assert_shape(reduce(np.ones((2, 3)), "row col -> row", "mean").shape, (2,))
        assert_shape(
            repeat(jnp.ones((2, 3)), "row col -> row col 2").shape,
            (2, 3, 2),
        )
        assert_shape(einsum(np.ones((3, 3)), "row row ->").shape, ())
        assert_shape(
            einsum(
                jnp.ones((3,)),
                jnp.ones((5,)),
                "row, col -> row col",
            ).shape,
            (3, 5),
        )


def test_auxiliary_api() -> None:
    assert_type(__version__, str)
    tensor = ones((2, 3, 5))
    assert parse_shape(tensor, "batch channel width") == {
        "batch": 2,
        "channel": 3,
        "width": 5,
    }
    assert_type(parse_shape(tensor, "batch channel width"), dict[str, int])
    assert_shape(asnumpy(tensor).shape, (2, 3, 5))

    packed, packed_shapes = pack([ones((2, 3)), ones((2, 5))], "batch *")
    assert_type(packed, Tensor[IntTuple])
    assert_shape(packed.shape, IntTuple, runtime=(2, 8))
    first, second = unpack(packed, packed_shapes, "batch *")
    assert_type(first, Tensor[IntTuple])
    assert_type(second, Tensor[IntTuple])
    assert_shape(first.shape, IntTuple, runtime=(2, 3))
    assert_shape(second.shape, IntTuple, runtime=(2, 5))
    assert issubclass(EinopsError, RuntimeError)
