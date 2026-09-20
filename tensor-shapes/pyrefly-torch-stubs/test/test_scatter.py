# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type, TYPE_CHECKING

import torch
from shape_extensions import assert_raises, assert_shape, IntTuple, IntVar
from torch import Tensor


def test_scatter_shapes() -> None:
    tensor = torch.zeros((3, 4))
    column_indices = torch.tensor([[0, 1], [2, 3], [1, 0]])
    columns = torch.ones((3, 2))
    assert_shape(torch.scatter(tensor, 1, column_indices, columns).shape, (3, 4))
    assert_shape(tensor.scatter(-1, column_indices, columns).shape, (3, 4))

    row_indices = torch.tensor([[0, 1, 2, 0], [2, 0, 1, 2]])
    rows = torch.ones((2, 4))
    assert_shape(tensor.scatter(0, row_indices, rows).shape, (3, 4))


def test_scatter_rejects_invalid_dimensions_and_shapes() -> None:
    tensor = torch.zeros((3, 4))
    indices = torch.zeros((3, 2), dtype=torch.int64)
    source = torch.ones((3, 2))
    assert_shape(tensor.scatter(1, indices, source).shape, (3, 4))

    # TODO: BUG: Reject an out-of-range dimension statically.
    with assert_raises(IndexError):
        tensor.scatter(2, indices, source)

    # TODO: BUG: The index rank must match the input rank.
    with assert_raises(RuntimeError):
        torch.scatter(tensor, 1, torch.zeros(2, dtype=torch.int64), source)

    # TODO: BUG: The source rank must match the index rank.
    with assert_raises(IndexError):
        tensor.scatter(1, indices, torch.ones(6))

    # TODO: BUG: Index dimensions outside the scatter axis cannot exceed the input.
    with assert_raises(RuntimeError):
        tensor.scatter(1, torch.zeros((4, 2), dtype=torch.int64), torch.ones((4, 2)))

    # TODO: BUG: The index cannot exceed the source along any axis.
    with assert_raises(RuntimeError):
        tensor.scatter(1, torch.zeros((3, 3), dtype=torch.int64), torch.ones((3, 2)))


if TYPE_CHECKING:

    def check_symbolic[N: IntVar, M: IntVar, K: IntVar](
        tensor: Tensor[[N, M]], indices: Tensor[[N, K]], source: Tensor[[N, K]]
    ) -> None:
        assert_type(torch.scatter(tensor, 1, indices, source), Tensor[[N, M]])
        assert_type(tensor.scatter(1, indices, source), Tensor[[N, M]])

    def check_gradual_inputs(
        tensor: Tensor[[2, 3]], indices: Tensor[IntTuple], source: Tensor[IntTuple]
    ) -> None:
        assert_type(tensor.scatter(0, indices, source), Tensor[[2, 3]])
