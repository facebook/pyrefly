# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type, TYPE_CHECKING

import torch
from shape_extensions import assert_raises, assert_shape, IntTuple, IntVar
from torch import Tensor


def test_matmul_rank_cases() -> None:
    vector = torch.ones(4)
    matrix = torch.ones((3, 4))
    right = torch.ones((4, 5))

    assert_shape(torch.matmul(vector, vector).shape, ())
    assert_shape(torch.matmul(vector, right).shape, (5,))
    assert_shape(torch.matmul(matrix, vector).shape, (3,))
    assert_shape(torch.matmul(matrix, right).shape, (3, 5))
    assert_shape(torch.matmul(torch.ones((2, 3, 4)), right).shape, (2, 3, 5))
    assert_shape(torch.matmul(torch.ones((2, 0)), torch.ones((0, 3))).shape, (2, 3))


def test_matmul_broadcasts_batch_dimensions() -> None:
    vector = torch.ones(4)
    matrix = torch.ones((3, 4))
    batched = torch.ones((2, 4, 5))

    assert_shape(torch.matmul(matrix, batched).shape, (2, 3, 5))
    assert_shape(torch.matmul(vector, batched).shape, (2, 5))
    assert_shape(torch.matmul(batched, torch.ones((5, 4))).shape, (2, 4, 4))
    assert_shape(torch.matmul(batched, torch.ones(5)).shape, (2, 4))
    assert_shape(
        torch.matmul(torch.ones((7, 1, 3, 4)), torch.ones((5, 4, 6))).shape,
        (7, 5, 3, 6),
    )


def test_matmul_entry_points() -> None:
    left = torch.ones((3, 4))
    right = torch.ones((4, 5))

    assert_shape(torch.matmul(left, right).shape, (3, 5))
    assert_shape(left.matmul(right).shape, (3, 5))
    assert_shape((left @ right).shape, (3, 5))


def test_matmul_rejects_invalid_shapes() -> None:
    left = torch.ones((2, 3))
    assert_shape((left @ torch.ones((3, 5))).shape, (2, 5))

    with assert_raises(RuntimeError):
        # E: core dimension 'n' has conflicting extents 3 and 4
        left @ torch.ones((4, 5))

    with assert_raises(RuntimeError):
        # E: Cannot evaluate type-level shape DSL call: gufunc: core dimension 'n' has conflicting extents 3 and 4
        left.matmul(torch.ones((4, 5)))

    with assert_raises(RuntimeError):
        # E: Cannot evaluate type-level shape DSL call: Cannot broadcast dimension Int[2] with dimension Int[5] at position 0
        torch.matmul(torch.ones((2, 3, 4)), torch.ones((5, 4, 6)))

    with assert_raises(RuntimeError):
        # E: Cannot evaluate type-level shape DSL call: matmul expects at least 1-D tensors
        torch.matmul(torch.ones(()), torch.ones(2))


if TYPE_CHECKING:

    def check_symbolic_matmul[N: IntVar, M: IntVar, K: IntVar, B: IntVar](
        matrix: Tensor[[N, M]],
        right: Tensor[[M, K]],
        batched: Tensor[[B, N, M]],
        batched_right: Tensor[[B, M, K]],
    ) -> None:
        assert_type(torch.matmul(matrix, right), Tensor[[N, K]])
        assert_type(matrix.matmul(right), Tensor[[N, K]])
        assert_type(matrix @ right, Tensor[[N, K]])
        assert_type(torch.matmul(batched, right), Tensor[[B, N, K]])
        assert_type(torch.matmul(batched, batched_right), Tensor[[B, N, K]])

    def check_gradual_matmul(
        left: Tensor, right: Tensor, shaped: Tensor[[2, 3]]
    ) -> None:
        assert_type(torch.matmul(left, right), Tensor[IntTuple])
        assert_type(left.matmul(right), Tensor[IntTuple])
        assert_type(torch.matmul(shaped, right), Tensor[IntTuple])
        assert_type(torch.matmul(left, shaped), Tensor[IntTuple])
