# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type, TYPE_CHECKING

import torch
from shape_extensions import (
    assert_raises,
    assert_shape,
    Elements,
    Int,
    IntTuple,
    IntVar,
)
from torch import Tensor


def test_topk_shapes() -> None:
    values, indices = torch.topk(torch.ones(10), k=3)
    assert_shape(values.shape, (3,))
    assert_shape(indices.shape, (3,))

    values, indices = torch.ones((4, 5)).topk(k=2, dim=1)
    assert_shape(values.shape, (4, 2))
    assert_shape(indices.shape, (4, 2))

    values, indices = torch.topk(torch.ones((3, 6)), k=4, dim=-1)
    assert_shape(values.shape, (3, 4))
    assert_shape(indices.shape, (3, 4))

    values, indices = torch.ones((2, 3, 4)).topk(0, dim=1)
    assert_shape(values.shape, (2, 0, 4))
    assert_shape(indices.shape, (2, 0, 4))

    scalar_values, scalar_indices = torch.tensor(1).topk(1)
    assert_shape(scalar_values.shape, ())
    assert_shape(scalar_indices.shape, ())


def test_topk_rejects_invalid_dimensions() -> None:
    matrix = torch.ones((2, 3))
    values, _ = matrix.topk(1)
    assert_shape(values.shape, (2, 1))

    with assert_raises(IndexError):
        torch.topk(matrix, 1, dim=2)  # E: dimension out of range

    with assert_raises(IndexError):
        matrix.topk(1, dim=-3)  # E: dimension out of range


def test_topk_rejects_invalid_k() -> None:
    vector = torch.ones(3)
    values, _ = vector.topk(3)
    assert_shape(values.shape, (3,))

    with assert_raises(RuntimeError):
        torch.topk(vector, -1)  # E: topk k must be non-negative

    with assert_raises(RuntimeError):
        vector.topk(4)  # E: topk k exceeds dimension size

    scalar = torch.tensor(1)
    with assert_raises(RuntimeError):
        scalar.topk(2)  # E: topk k exceeds dimension size


if TYPE_CHECKING:

    def check_symbolic[N: IntVar, M: IntVar](x: Tensor[[N, M]]) -> None:
        result = torch.topk(x, k=3, dim=1)
        assert_type(result, torch.return_types.topk[[N, 3]])
        assert_type(result.values, Tensor[[N, 3]])
        assert_type(result.indices, Tensor[[N, 3]])

    def check_symbolic_suffix[Shape: IntTuple, K: IntVar](
        x: Tensor[[*Elements[Shape], 3]], k: Int[K]
    ) -> None:
        assert_type(
            x.topk(k),
            torch.return_types.topk[[*Elements[Shape], K]],
        )
        assert_type(
            torch.topk(x, k),
            torch.return_types.topk[[*Elements[Shape], K]],
        )

    def check_gradual_boundaries(
        x: Tensor[[2, 3, 4]], bare: Tensor, dim: int, k: int
    ) -> None:
        assert_type(x.topk(k, dim=1), torch.return_types.topk[[2, int, 4]])
        assert_type(torch.topk(x, 2, dim=dim), torch.return_types.topk[IntTuple])
        assert_type(bare.topk(2), torch.return_types.topk[IntTuple])

    def check_invalid_k_type[T, S: str](
        x: Tensor[[4, 32]], unconstrained: T, string: S
    ) -> None:
        # E: `T` is not assignable to upper bound `Int[int]` of type variable `K`
        torch.topk(x, unconstrained)
        # E: `S` is not assignable to upper bound `Int[int]` of type variable `K`
        torch.topk(x, string)
