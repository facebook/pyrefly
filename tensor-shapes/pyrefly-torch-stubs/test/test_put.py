# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type, TYPE_CHECKING

import torch
from shape_extensions import assert_raises, assert_shape, IntTuple, IntVar
from torch import Tensor


def test_put_shapes() -> None:
    tensor = torch.zeros((2, 3))
    indices = torch.tensor([[0, 1], [4, 5]])
    source = torch.ones(4)
    assert_shape(torch.put(tensor, indices, source).shape, (2, 3))
    assert_shape(tensor.put(indices, source).shape, (2, 3))
    assert_shape(tensor.put_(indices, source, accumulate=True).shape, (2, 3))

    scalar = torch.tensor(0.0)
    assert_shape(scalar.put(torch.tensor([0]), torch.tensor([1.0])).shape, ())


def test_put_rejects_invalid_shapes() -> None:
    tensor = torch.zeros((2, 3))
    indices = torch.tensor([[0, 1], [4, 5]])
    assert_shape(tensor.put(indices, torch.ones(4)).shape, (2, 3))

    with assert_raises(IndexError):
        # E: source must have the same number of elements
        tensor.put(indices, torch.ones(3))

    with assert_raises(IndexError):
        # E: cannot index an empty input
        torch.put(torch.zeros((0, 3)), torch.tensor([0]), torch.ones(1))


if TYPE_CHECKING:

    def check_symbolic[N: IntVar, M: IntVar, K: IntVar](
        tensor: Tensor[[N, M]], indices: Tensor[[K]], source: Tensor[[K]]
    ) -> None:
        assert_type(torch.put(tensor, indices, source), Tensor[[N, M]])
        assert_type(tensor.put(indices, source), Tensor[[N, M]])

    def check_gradual_inputs(
        tensor: Tensor[[2, 3]], indices: Tensor[IntTuple], source: Tensor[IntTuple]
    ) -> None:
        assert_type(tensor.put(indices, source), Tensor[[2, 3]])
