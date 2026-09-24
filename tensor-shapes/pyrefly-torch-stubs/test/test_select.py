# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type, TYPE_CHECKING

import torch
from shape_extensions import assert_raises, assert_shape, Elements, IntTuple, IntVar
from torch import Tensor


def test_select_shapes() -> None:
    tensor = torch.ones((2, 3, 4))
    assert_shape(torch.select(tensor, dim=1, index=0).shape, (2, 4))
    assert_shape(tensor.select(dim=-1, index=2).shape, (2, 3))
    assert_shape(torch.select(tensor, dim=0, index=-1).shape, (3, 4))


def test_select_rejects_invalid_dimensions() -> None:
    matrix = torch.ones((2, 3))
    assert_shape(matrix.select(0, 0).shape, (3,))

    with assert_raises(IndexError):
        matrix.select(-3, 0)  # E: select dimension out of range

    scalar = torch.tensor(1)
    with assert_raises(IndexError):
        scalar.select(-1, 0)  # E: select dimension out of range


def test_select_rejects_invalid_indices() -> None:
    vector = torch.ones(3)
    assert_shape(vector.select(0, 2).shape, ())

    with assert_raises(IndexError):
        vector.select(0, 3)  # E: select index out of range

    with assert_raises(IndexError):
        torch.select(vector, 0, -4)  # E: select index out of range

    with assert_raises(IndexError):
        torch.select(vector, -1, 3)  # E: select index out of range

    empty = torch.empty(0)
    with assert_raises(IndexError):
        empty.select(0, 0)  # E: select index out of range


if TYPE_CHECKING:

    def check_symbolic[N: IntVar, M: IntVar](x: Tensor[[N, M, 3]]) -> None:
        assert_type(x.select(1, 0), Tensor[[N, 3]])
        assert_type(torch.select(x, -1, 0), Tensor[[N, M]])

    def check_symbolic_suffix[Shape: IntTuple](
        x: Tensor[[*Elements[Shape], 3]],
    ) -> None:
        assert_type(x.select(-1, 0), Tensor[Shape])
        assert_type(torch.select(x, -1, 0), Tensor[Shape])

    def check_gradual_boundaries(
        x: Tensor[[2, 3]], dim: int, index: int, bare: Tensor
    ) -> None:
        assert_type(torch.select(x, dim, 0), Tensor[IntTuple])
        assert_type(x.select(0, index), Tensor[[3]])
        assert_type(bare.select(0, 0), Tensor[IntTuple])
