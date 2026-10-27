# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type, TYPE_CHECKING

import torch
from shape_extensions import assert_raises, assert_shape, IntVar
from torch import Tensor


def test_index_put_shapes() -> None:
    tensor = torch.zeros((3, 4))
    rows = torch.tensor([0, 2])
    columns = torch.tensor([1, 3])
    values = torch.ones(2)
    assert_shape(torch.index_put(tensor, (rows, columns), values).shape, (3, 4))
    assert_shape(tensor.index_put((rows, columns), values).shape, (3, 4))
    assert_shape(tensor.index_put_((rows, columns), values).shape, (3, 4))

    broadcast_rows = torch.tensor([[0], [2]])
    broadcast_columns = torch.tensor([[0, 1, 3]])
    assert_shape(
        tensor.index_put((broadcast_rows, broadcast_columns), torch.ones((2, 3))).shape,
        (3, 4),
    )


def test_index_put_rejects_invalid_shapes() -> None:
    tensor = torch.zeros((3, 4))
    assert_shape(
        tensor.index_put((torch.tensor([0]),), torch.ones((1, 4))).shape, (3, 4)
    )

    # TODO: BUG: Reject more index tensors than input dimensions statically.
    with assert_raises(IndexError):
        tensor.index_put(
            (torch.tensor([0]), torch.tensor([0]), torch.tensor([0])),
            torch.ones(1),
        )

    # TODO: BUG: Index tensor shapes must broadcast together.
    with assert_raises(IndexError):
        torch.index_put(
            tensor,
            (torch.tensor([0, 1]), torch.tensor([0, 1, 2])),
            torch.ones(3),
        )

    # TODO: BUG: Values must broadcast to the indexed result shape.
    with assert_raises(RuntimeError):
        tensor.index_put((torch.tensor([0, 1]), torch.tensor([0, 1])), torch.ones(3))


if TYPE_CHECKING:

    def check_symbolic[N: IntVar, M: IntVar](
        tensor: Tensor[[N, M]],
        indices: tuple[Tensor[[2]], Tensor[[2]]],
        values: Tensor[[2]],
    ) -> None:
        assert_type(torch.index_put(tensor, indices, values), Tensor[[N, M]])
        assert_type(tensor.index_put(indices, values), Tensor[[N, M]])
