# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type, TYPE_CHECKING

import torch
from shape_extensions import assert_raises, assert_shape, IntTuple, IntVar
from torch import Tensor


def test_take_along_dim_shapes() -> None:
    tensor = torch.arange(24).reshape(2, 3, 4)
    indices = torch.tensor([[[0, 1], [2, 3], [1, 0]], [[3, 2], [1, 0], [0, 2]]])
    assert_shape(torch.take_along_dim(tensor, indices, dim=2).shape, (2, 3, 2))
    assert_shape(tensor.take_along_dim(indices, dim=-1).shape, (2, 3, 2))


def test_take_along_dim_broadcasting_and_flattening() -> None:
    tensor = torch.arange(12).reshape(3, 4)

    # TODO: BUG: Broadcast dimensions outside the selected axis.
    assert_shape(
        tensor.take_along_dim(torch.tensor([[0, 1]]), dim=1).shape,
        (1, 2),
        runtime=(3, 2),
    )

    # TODO: BUG: Support `dim=None` and return a flattened index shape.
    assert_shape(
        torch.take_along_dim(  # E: Missing argument `dim`
            tensor, torch.tensor([[0, 5], [1, 11]])
        ).shape,
        (2, 2),
        runtime=(4,),
    )


def test_take_along_dim_rejects_invalid_shapes() -> None:
    tensor = torch.zeros((3, 4))
    assert_shape(
        tensor.take_along_dim(torch.zeros((3, 2), dtype=torch.int64), dim=1).shape,
        (3, 2),
    )

    # TODO: BUG: Reject an out-of-range dimension statically.
    with assert_raises(IndexError):
        tensor.take_along_dim(torch.zeros((3, 2), dtype=torch.int64), dim=2)

    # TODO: BUG: The input and index ranks must match.
    with assert_raises(RuntimeError):
        torch.take_along_dim(tensor, torch.zeros(2, dtype=torch.int64), dim=1)

    # TODO: BUG: Dimensions outside the selected axis must be broadcastable.
    with assert_raises(RuntimeError):
        tensor.take_along_dim(torch.zeros((2, 2), dtype=torch.int64), dim=1)


if TYPE_CHECKING:

    def check_symbolic[N: IntVar, M: IntVar, K: IntVar](
        tensor: Tensor[[N, M]], indices: Tensor[[N, K]]
    ) -> None:
        assert_type(torch.take_along_dim(tensor, indices, dim=1), Tensor[[N, K]])
        assert_type(tensor.take_along_dim(indices, dim=1), Tensor[[N, K]])

    def check_gradual_inputs(
        tensor: Tensor[IntTuple], indices: Tensor[IntTuple]
    ) -> None:
        assert_type(tensor.take_along_dim(indices, dim=0), Tensor[IntTuple])
