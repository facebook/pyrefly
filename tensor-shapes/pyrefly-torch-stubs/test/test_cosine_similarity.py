# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type, TYPE_CHECKING

import torch
import torch.nn.functional as F
from shape_extensions import assert_raises, assert_shape, IntTuple, IntVar
from torch import Tensor


def test_cosine_similarity_shapes() -> None:
    matrix = torch.randn((2, 3))
    assert_shape(F.cosine_similarity(matrix, matrix).shape, (2,))
    assert_shape(F.cosine_similarity(matrix, matrix, dim=-1).shape, (2,))

    row = torch.randn((1, 3))
    column = torch.randn((2, 1))
    assert_shape(F.cosine_similarity(row, column, dim=0).shape, (3,))
    assert_shape(F.cosine_similarity(row, column, dim=1).shape, (2,))

    rank_three = torch.randn((2, 3, 4))
    assert_shape(
        F.cosine_similarity(rank_three, rank_three, dim=-2).shape,
        (2, 4),
    )

    scalar = torch.randn(())
    assert_shape(F.cosine_similarity(scalar, scalar, dim=0).shape, ())
    assert_shape(F.cosine_similarity(scalar, scalar, dim=-1).shape, ())


def test_cosine_similarity_rejects_invalid_shapes() -> None:
    left = torch.randn((2, 3))
    assert_shape(left.shape, (2, 3))

    with assert_raises(RuntimeError):
        # E: Cannot broadcast dimension
        F.cosine_similarity(left, torch.randn((4, 5)))
    with assert_raises(IndexError):
        # E: dimension out of range
        F.cosine_similarity(left, left, dim=2)
    with assert_raises(IndexError):
        F.cosine_similarity(left, left, dim=-3)  # E: dimension out of range

    scalar = torch.randn(())
    with assert_raises(IndexError):
        F.cosine_similarity(scalar, scalar, dim=1)  # E: dimension out of range


if TYPE_CHECKING:

    def check_symbolic_cosine_similarity[B: IntVar, M: IntVar, N: IntVar](
        left: Tensor[[B, 1, N]], right: Tensor[[1, M, N]]
    ) -> None:
        assert_type(F.cosine_similarity(left, right, dim=-1), Tensor[[B, M]])
        assert_type(F.cosine_similarity(left, right, dim=1), Tensor[[B, N]])

    def check_gradual_cosine_similarity(
        left: Tensor[[int, int, int]],
        right: Tensor[[1, int, 1]],
        bare: Tensor,
        open_rank: Tensor[IntTuple],
        dim: int,
    ) -> None:
        assert_type(F.cosine_similarity(left, right, dim=-1), Tensor[[int, int]])
        assert_type(F.cosine_similarity(left, right, dim=dim), Tensor[IntTuple])
        assert_type(F.cosine_similarity(bare, left, dim=-1), Tensor[IntTuple])
        assert_type(F.cosine_similarity(open_rank, left, dim=0), Tensor[IntTuple])
