# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type, TYPE_CHECKING

import torch
from shape_extensions import assert_raises, assert_shape, IntTuple, IntVar
from torch import Tensor


def test_index_copy_shapes() -> None:
    tensor = torch.zeros((3, 4))
    indices = torch.tensor([0, 2])
    source = torch.ones((2, 4))
    assert_shape(torch.index_copy(tensor, 0, indices, source).shape, (3, 4))
    assert_shape(tensor.index_copy(0, indices, source).shape, (3, 4))
    assert_shape(tensor.index_copy_(0, indices, source).shape, (3, 4))

    column_indices = torch.tensor([0, 2])
    columns = torch.ones((3, 2))
    assert_shape(tensor.index_copy(-1, column_indices, columns).shape, (3, 4))

    scalar = torch.tensor(0.0)
    assert_shape(scalar.index_copy(0, torch.tensor([0]), torch.tensor(1.0)).shape, ())


def test_index_copy_rejects_invalid_dimensions_and_indices() -> None:
    tensor = torch.zeros((3, 4))
    source = torch.ones((1, 4))
    assert_shape(tensor.index_copy(0, torch.tensor([0]), source).shape, (3, 4))

    # TODO: BUG: Reject an out-of-range dimension statically.
    with assert_raises(IndexError):
        tensor.index_copy(2, torch.tensor([0]), torch.ones((3, 1)))

    # TODO: BUG: Reject indices with rank greater than one statically.
    with assert_raises(IndexError):
        tensor.index_copy(0, torch.tensor([[0, 1]]), torch.ones((2, 4)))


def test_index_copy_rejects_invalid_source_shapes() -> None:
    tensor = torch.zeros((3, 4))
    indices = torch.tensor([0, 1])
    assert_shape(tensor.index_copy(0, indices, torch.ones((2, 4))).shape, (3, 4))

    # TODO: BUG: The source rank must match the input rank.
    with assert_raises(IndexError):
        tensor.index_copy(0, indices, torch.ones((2, 4, 1)))

    # TODO: BUG: Source dimensions outside the selected axis must match the input.
    with assert_raises(RuntimeError):
        tensor.index_copy(0, indices, torch.ones((2, 5)))

    # TODO: BUG: The source extent along the selected axis must match the index length.
    with assert_raises(IndexError):
        torch.index_copy(tensor, 0, indices, torch.ones((1, 4)))


if TYPE_CHECKING:

    def check_symbolic[N: IntVar, M: IntVar, K: IntVar](
        tensor: Tensor[[N, M]], indices: Tensor[[K]], source: Tensor[[N, K]]
    ) -> None:
        assert_type(torch.index_copy(tensor, 1, indices, source), Tensor[[N, M]])
        assert_type(tensor.index_copy(1, indices, source), Tensor[[N, M]])

    def check_gradual_inputs(
        tensor: Tensor[[2, 3]], indices: Tensor[IntTuple], source: Tensor[IntTuple]
    ) -> None:
        assert_type(tensor.index_copy(0, indices, source), Tensor[[2, 3]])
