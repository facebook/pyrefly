# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type, TYPE_CHECKING

import torch
from shape_extensions import assert_raises, assert_shape, IntTuple, IntVar
from torch import Tensor


def test_index_add_shapes() -> None:
    tensor = torch.zeros((3, 4))
    indices = torch.tensor([0, 2])
    source = torch.ones((2, 4))
    assert_shape(torch.index_add(tensor, 0, indices, source).shape, (3, 4))
    assert_shape(tensor.index_add(0, indices, source).shape, (3, 4))
    assert_shape(tensor.index_add_(0, indices, source).shape, (3, 4))
    assert_shape(tensor.index_add(0, torch.tensor(1), torch.ones((1, 4))).shape, (3, 4))

    column_indices = torch.tensor([0, 2])
    columns = torch.ones((3, 2))
    assert_shape(tensor.index_add(-1, column_indices, columns).shape, (3, 4))

    scalar = torch.tensor(0.0)
    assert_shape(scalar.index_add(0, torch.tensor([0]), torch.tensor(1.0)).shape, ())
    assert_shape(scalar.index_add(0, torch.tensor(0), torch.tensor(1.0)).shape, ())


def test_index_add_rejects_invalid_dimensions_and_indices() -> None:
    tensor = torch.zeros((3, 4))
    source = torch.ones((1, 4))
    assert_shape(tensor.index_add(0, torch.tensor([0]), source).shape, (3, 4))

    with assert_raises(IndexError):
        # E: dimension out of range
        tensor.index_add(2, torch.tensor([0]), torch.ones((3, 1)))

    with assert_raises(IndexError):
        # E: index must be 0D or 1D
        tensor.index_add(0, torch.tensor([[0, 1]]), torch.ones((2, 4)))

    with assert_raises(IndexError):
        # E: scalar index must have one element
        torch.tensor(0.0).index_add(0, torch.tensor([0, 0]), torch.tensor(1.0))


def test_index_add_rejects_invalid_source_shapes() -> None:
    tensor = torch.zeros((3, 4))
    indices = torch.tensor([0, 1])
    assert_shape(tensor.index_add(0, indices, torch.ones((2, 4))).shape, (3, 4))

    with assert_raises(RuntimeError):
        # E: source rank must match input rank
        tensor.index_add(0, indices, torch.ones((2, 4, 1)))

    # `assert_raises` checks these exact runtime behaviors; Torch does not
    # promote a scalar source or receiver for `index_add`.
    with assert_raises(RuntimeError):
        # E: source rank must match input rank
        tensor.index_add(0, torch.tensor([0]), torch.tensor(1.0))

    with assert_raises(RuntimeError):
        # E: source rank must match input rank
        torch.tensor(0.0).index_add(0, torch.tensor([0]), torch.tensor([1.0]))

    with assert_raises(RuntimeError):
        # E: source shape is incompatible with input
        tensor.index_add(0, indices, torch.ones((2, 5)))

    with assert_raises(RuntimeError):
        # E: source shape is incompatible with input
        torch.index_add(tensor, 0, indices, torch.ones((1, 4)))


if TYPE_CHECKING:

    def check_symbolic[N: IntVar, M: IntVar, K: IntVar](
        tensor: Tensor[[N, M]], indices: Tensor[[K]], source: Tensor[[N, K]]
    ) -> None:
        assert_type(torch.index_add(tensor, 1, indices, source), Tensor[[N, M]])
        assert_type(tensor.index_add(1, indices, source), Tensor[[N, M]])

    def check_gradual_inputs(
        tensor: Tensor[[2, 3]], indices: Tensor[IntTuple], source: Tensor[IntTuple]
    ) -> None:
        assert_type(tensor.index_add(0, indices, source), Tensor[[2, 3]])

    def check_gradual_index_rejects_known_source_rank(
        tensor: Tensor[[2, 3]], indices: Tensor[IntTuple], source: Tensor[[1, 2, 3]]
    ) -> None:
        tensor.index_add(0, indices, source)  # E: source rank must match input rank
