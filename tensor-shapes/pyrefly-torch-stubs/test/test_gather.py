# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type, TYPE_CHECKING

import torch
from shape_extensions import assert_raises, assert_shape, IntTuple, IntVar
from torch import Tensor


def test_gather_shapes() -> None:
    tensor = torch.arange(12).reshape(3, 4)
    column_indices = torch.tensor([[0, 1], [2, 3], [1, 0]])
    assert_shape(torch.gather(tensor, 1, column_indices).shape, (3, 2))
    assert_shape(tensor.gather(-1, column_indices).shape, (3, 2))

    row_indices = torch.tensor([[0, 1, 2, 0], [2, 0, 1, 2]])
    assert_shape(tensor.gather(0, row_indices).shape, (2, 4))

    scalar = torch.tensor(1.0)
    assert_shape(scalar.gather(0, torch.tensor(0)).shape, ())


def test_gather_rejects_invalid_dimensions_and_shapes() -> None:
    tensor = torch.zeros((3, 4))
    indices = torch.zeros((3, 2), dtype=torch.int64)
    assert_shape(tensor.gather(1, indices).shape, (3, 2))

    with assert_raises(IndexError):
        tensor.gather(2, indices)  # E: gather dimension out of range

    with assert_raises(RuntimeError):
        # E: gather index rank must match input rank
        torch.gather(tensor, 1, torch.zeros(2, dtype=torch.int64))

    with assert_raises(RuntimeError):
        # E: gather index shape exceeds input shape
        tensor.gather(1, torch.zeros((4, 2), dtype=torch.int64))


if TYPE_CHECKING:

    def check_symbolic[N: IntVar, M: IntVar, K: IntVar](
        tensor: Tensor[[N, M]], indices: Tensor[[N, K]]
    ) -> None:
        assert_type(torch.gather(tensor, 1, indices), Tensor[[N, K]])
        assert_type(tensor.gather(1, indices), Tensor[[N, K]])

    def check_gradual_input(tensor: Tensor[IntTuple], indices: Tensor[[2, 3]]) -> None:
        assert_type(tensor.gather(0, indices), Tensor[[2, 3]])
