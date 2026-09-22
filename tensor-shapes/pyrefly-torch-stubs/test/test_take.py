# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type, TYPE_CHECKING

import torch
from shape_extensions import assert_raises, assert_shape, IntTuple, IntVar
from torch import Tensor


def test_take_shapes() -> None:
    tensor = torch.arange(12).reshape(3, 4)
    assert_shape(torch.take(tensor, torch.tensor([0, 5, 11])).shape, (3,))
    assert_shape(tensor.take(torch.tensor([[0, 1], [10, 11]])).shape, (2, 2))
    assert_shape(torch.tensor(1.0).take(torch.tensor([0, 0])).shape, (2,))


def test_take_empty_indices_and_inputs() -> None:
    tensor = torch.zeros((2, 3))
    assert_shape(tensor.take(torch.tensor([], dtype=torch.int64)).shape, (0,))

    with assert_raises(IndexError):
        # E: take cannot select from an empty input
        torch.take(torch.zeros((0, 3)), torch.tensor([0]))


if TYPE_CHECKING:

    def check_symbolic[N: IntVar, M: IntVar, IndexShape: IntTuple](
        tensor: Tensor[[N, M]], indices: Tensor[IndexShape]
    ) -> None:
        assert_type(torch.take(tensor, indices), Tensor[IndexShape])
        assert_type(tensor.take(indices), Tensor[IndexShape])

    def check_gradual_input(tensor: Tensor[IntTuple], indices: Tensor[[2, 3]]) -> None:
        assert_type(tensor.take(indices), Tensor[[2, 3]])
