# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test nn.functional operations with generic TypeVarTuple signatures."""

from typing import assert_type, cast

from shape_extensions import IntTuple, IntVar
from torch import Tensor
from torch.nn import functional as F


def test_cosine_similarity_reduction() -> None:
    x = cast(Tensor[[2, 3]], ...)
    rank_three = cast(Tensor[[2, 3, 4]], ...)
    y = cast(Tensor[[1, 3]], ...)
    z = cast(Tensor[[2, 1]], ...)
    scalar = cast(Tensor[[]], ...)

    assert_type(F.cosine_similarity(x, x), Tensor[[2]])
    assert_type(F.cosine_similarity(y, z, dim=0), Tensor[[3]])
    assert_type(F.cosine_similarity(y, z, dim=1), Tensor[[2]])
    assert_type(F.cosine_similarity(rank_three, rank_three, dim=-2), Tensor[[2, 4]])
    assert_type(F.cosine_similarity(x, x, dim=-1), Tensor[[2]])
    assert_type(F.cosine_similarity(scalar, scalar, dim=0), Tensor[[]])
    assert_type(F.cosine_similarity(scalar, scalar, dim=-1), Tensor[[]])


def check_cosine_similarity_symbolic[B: IntVar, M: IntVar, N: IntVar](
    x: Tensor[[B, 1, N]], y: Tensor[[1, M, N]]
) -> None:
    assert_type(F.cosine_similarity(x, y, dim=-1), Tensor[[B, M]])
    assert_type(F.cosine_similarity(x, y, dim=1), Tensor[[B, N]])


def check_cosine_similarity_gradual(
    x: Tensor[[int, int, int]],
    y: Tensor[[1, int, 1]],
    bare: Tensor,
    open_rank: Tensor[IntTuple],
    dim: int,
) -> None:
    assert_type(F.cosine_similarity(x, y, dim=-1), Tensor[[int, int]])
    assert_type(F.cosine_similarity(x, y, dim=dim), Tensor[IntTuple])
    assert_type(F.cosine_similarity(bare, x, dim=-1), Tensor[IntTuple])
    assert_type(F.cosine_similarity(open_rank, x, dim=0), Tensor[IntTuple])


def test_adaptive_pool_shapes() -> None:
    one_d = cast(Tensor[[2, 3, 12]], ...)
    two_d = cast(Tensor[[2, 3, 12, 15]], ...)
    three_d = cast(Tensor[[2, 3, 8, 12, 15]], ...)

    assert_type(F.adaptive_avg_pool1d(one_d, 4), Tensor[[2, 3, 4]])
    assert_type(F.adaptive_avg_pool1d(input=one_d, output_size=4), Tensor[[2, 3, 4]])
    assert_type(F.adaptive_avg_pool2d(two_d, (4, 5)), Tensor[[2, 3, 4, 5]])
    assert_type(F.adaptive_avg_pool3d(three_d, 4), Tensor[[2, 3, 4, 4, 4]])

    assert_type(F.adaptive_max_pool1d(one_d, (5,)), Tensor[[2, 3, 5]])
    assert_type(
        F.adaptive_max_pool2d(two_d, (4, 5), return_indices=False),
        Tensor[[2, 3, 4, 5]],
    )
    assert_type(
        F.adaptive_max_pool3d(three_d, (4, 5, 6), return_indices=True),
        tuple[Tensor, Tensor],
    )


def test_adaptive_pool_unbatched_shapes() -> None:
    two_d = cast(Tensor[[3, 12, 15]], ...)
    three_d = cast(Tensor[[3, 8, 12, 15]], ...)

    assert_type(F.adaptive_avg_pool2d(two_d, 4), Tensor[[3, 4, 4]])
    assert_type(F.adaptive_max_pool3d(three_d, (4, 5, 6)), Tensor[[3, 4, 5, 6]])


def check_adaptive_pool_fallbacks(
    exact: Tensor[[2, 3, 12, 15]],
    open_rank: Tensor[IntTuple],
    scalar: int,
    pair: tuple[int, int],
    return_indices: bool,
) -> None:
    assert_type(F.adaptive_avg_pool2d(exact, scalar), Tensor[[2, 3, int, int]])
    assert_type(F.adaptive_avg_pool2d(exact, pair), Tensor[[2, 3, int, int]])
    assert_type(F.adaptive_avg_pool2d(open_rank, (4, 5)), Tensor[IntTuple])
    assert_type(F.adaptive_avg_pool2d(exact, (None, 5)), Tensor)
    assert_type(
        F.adaptive_max_pool2d(exact, (4, None), return_indices=True),
        tuple[Tensor, Tensor],
    )
    three_d = cast(Tensor[[2, 3, 8, 12, 15]], ...)
    assert_type(F.adaptive_avg_pool3d(three_d, (None, 5, None)), Tensor)
    assert_type(F.adaptive_max_pool3d(three_d, (4, None, 6)), Tensor)
    assert_type(
        F.adaptive_max_pool2d(exact, (4, 5), return_indices=return_indices),
        Tensor | tuple[Tensor, Tensor],
    )
