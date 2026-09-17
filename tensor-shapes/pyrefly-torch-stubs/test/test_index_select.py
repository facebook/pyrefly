# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type, TYPE_CHECKING

import torch
from shape_extensions import assert_raises, assert_shape, Elements, IntTuple, IntVar
from torch import Tensor


def test_index_select_shapes() -> None:
    tensor = torch.arange(24).reshape(2, 3, 4)
    indices = torch.tensor([0, 2])
    assert_shape(torch.index_select(tensor, 1, indices).shape, (2, 2, 4))
    assert_shape(tensor.index_select(-1, indices).shape, (2, 3, 2))

    scalar_index = torch.tensor(1)
    assert_shape(torch.index_select(tensor, 1, scalar_index).shape, (2, 1, 4))

    empty_index = torch.zeros(0, dtype=torch.int64)
    assert_shape(tensor.index_select(0, empty_index).shape, (0, 3, 4))


def test_index_select_rejects_invalid_dimensions() -> None:
    matrix = torch.ones((2, 3))
    indices = torch.tensor([0])
    assert_shape(matrix.index_select(0, indices).shape, (1, 3))

    with assert_raises(IndexError):
        torch.index_select(matrix, 2, indices)  # E: index_select dimension out of range

    scalar = torch.tensor(1)
    scalar_result = scalar.index_select(-1, indices)
    assert_shape(scalar_result.shape, ())

    with assert_raises(IndexError):
        scalar.index_select(1, indices)  # E: index_select dimension out of range


def test_index_select_validates_scalar_index_count() -> None:
    scalar = torch.tensor(1)
    scalar_index = torch.tensor(0)
    assert_shape(torch.index_select(scalar, 0, scalar_index).shape, ())

    empty_index = torch.zeros(0, dtype=torch.int64)
    with assert_raises(RuntimeError):
        # E: index_select scalar index must have one element
        scalar.index_select(0, empty_index)

    repeated_index = torch.tensor([0, 0])
    with assert_raises(RuntimeError):
        # E: index_select scalar index must have one element
        torch.index_select(scalar, -1, repeated_index)


def test_index_select_rejects_matrix_indices() -> None:
    matrix = torch.ones((2, 3))
    vector_index = torch.tensor([0])
    assert_shape(matrix.index_select(0, vector_index).shape, (1, 3))

    matrix_index = torch.tensor([[0, 1]])
    with assert_raises(IndexError):
        # E: index_select index must be 0D or 1D
        torch.index_select(matrix, 0, matrix_index)


if TYPE_CHECKING:

    def check_symbolic[N: IntVar, K: IntVar](
        x: Tensor[[N, 10]], indices: Tensor[[K]]
    ) -> None:
        assert_type(torch.index_select(x, dim=1, index=indices), Tensor[[N, K]])

    def check_symbolic_suffix[Shape: IntTuple, K: IntVar](
        x: Tensor[[*Elements[Shape], 3]], indices: Tensor[[K]]
    ) -> None:
        assert_type(x.index_select(-1, indices), Tensor[[*Elements[Shape], K]])
        assert_type(torch.index_select(x, -1, indices), Tensor[[*Elements[Shape], K]])

    def check_gradual_boundaries(
        x: Tensor[[2, 3]], dim: int, indices: Tensor[[4]], bare: Tensor
    ) -> None:
        assert_type(x.index_select(dim, indices), Tensor[IntTuple])
        assert_type(torch.index_select(bare, 0, indices), Tensor[IntTuple])
