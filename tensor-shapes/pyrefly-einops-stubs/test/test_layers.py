# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import assert_type, NotRequired, TYPE_CHECKING, TypedDict

from einops.layers.torch import EinMix, Rearrange, Reduce
from shape_extensions import assert_shape, IntTuple
from torch import ones, Tensor


def test_rearrange_layer() -> None:
    patch_embedding = Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=2, p2=5)
    first = patch_embedding(ones((2, 3, 8, 10)))
    assert_type(first, Tensor[[2, 8, 30]])
    assert_shape(first.shape, (2, 8, 30))
    assert_type(patch_embedding.forward(ones((4, 3, 12, 15))), Tensor[[4, 18, 30]])


def test_reduce_layer() -> None:
    pool_heads = Reduce(pattern="b (h d) n -> b h", reduction="sum", h=3)
    output = pool_heads(ones((2, 12, 10)))
    assert_type(output, Tensor[[2, 3]])
    assert_shape(output.shape, (2, 3))
    assert_type(pool_heads.forward(ones((4, 15, 7))), Tensor[[4, 3]])


def test_dynamic_axes_layer() -> None:
    axes_lengths: dict[str, int] = {"p1": 2, "p2": 5}
    patch_embedding = Rearrange(
        "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", **axes_lengths
    )
    output = patch_embedding(ones((2, 3, 8, 10)))
    assert_type(output, Tensor[[2, int, int]])
    assert_shape(output.shape, (2, int, int), runtime=(2, 8, 30))


def test_einmix_fallback() -> None:
    projection = EinMix("b c -> b d", "c d", c=3, d=4)
    output = projection(ones((2, 3)))
    assert_type(output, Tensor[IntTuple])
    assert_shape(output.shape, IntTuple, runtime=(2, 4))


if TYPE_CHECKING:

    class OptionalPatchAxes(TypedDict, closed=True):
        p1: NotRequired[int]
        p2: NotRequired[int]

    def check_optional_axes(axes: OptionalPatchAxes) -> None:
        layer = Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", **axes)
        assert_type(layer(ones((2, 3, 8, 10))), Tensor[[2, int, int]])
