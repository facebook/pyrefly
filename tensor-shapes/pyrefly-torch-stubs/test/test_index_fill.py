# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type, TYPE_CHECKING

import torch
from shape_extensions import assert_raises, assert_shape, IntTuple, IntVar
from torch import Tensor


def test_index_fill_shapes() -> None:
    tensor = torch.zeros((3, 4))
    indices = torch.tensor([0, 2])
    assert_shape(torch.index_fill(tensor, 0, indices, 1.0).shape, (3, 4))
    assert_shape(tensor.index_fill(-1, indices, 1.0).shape, (3, 4))
    assert_shape(tensor.index_fill_(0, torch.tensor(1), 1.0).shape, (3, 4))

    scalar = torch.tensor(0.0)
    assert_shape(scalar.index_fill(0, torch.tensor([0, 0]), 1.0).shape, ())


def test_index_fill_rejects_invalid_dimensions_and_indices() -> None:
    tensor = torch.zeros((3, 4))
    assert_shape(tensor.index_fill(0, torch.tensor([0]), 1.0).shape, (3, 4))

    with assert_raises(IndexError):
        tensor.index_fill(2, torch.tensor([0]), 1.0)  # E: dimension out of range

    with assert_raises(RuntimeError):
        # E: index_fill index must be a scalar or vector
        torch.index_fill(tensor, 0, torch.tensor([[0, 1]]), 1.0)


if TYPE_CHECKING:

    def check_symbolic[N: IntVar, M: IntVar](
        tensor: Tensor[[N, M]], indices: Tensor[[N]]
    ) -> None:
        assert_type(torch.index_fill(tensor, 0, indices, 1.0), Tensor[[N, M]])
        assert_type(tensor.index_fill(0, indices, 1.0), Tensor[[N, M]])

    def check_gradual_index(tensor: Tensor[[2, 3]], indices: Tensor[IntTuple]) -> None:
        assert_type(tensor.index_fill(0, indices, 1.0), Tensor[[2, 3]])
