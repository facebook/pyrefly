# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type, TYPE_CHECKING

import torch
from shape_extensions import assert_raises, assert_shape, IntTuple
from torch import Tensor


def test_cumulative_shapes() -> None:
    vector = torch.randn(4)
    assert_shape(torch.cumsum(vector, dim=0).shape, (4,))
    assert_shape(vector.cumprod(dim=-1).shape, (4,))

    tensor = torch.randn((2, 3, 4))
    assert_shape(torch.cumsum(tensor, dim=0).shape, (2, 3, 4))
    assert_shape(tensor.cumprod(dim=-2).shape, (2, 3, 4))

    scalar = torch.randn(())
    assert_shape(torch.cumsum(scalar, dim=0).shape, ())
    assert_shape(scalar.cumprod(dim=-1).shape, ())


def test_cumulative_extrema_shapes() -> None:
    matrix = torch.randn((3, 4))
    maximum = torch.cummax(matrix, dim=0)
    assert_shape(maximum.values.shape, (3, 4))
    assert_shape(maximum.indices.shape, (3, 4))

    tensor = torch.randn((2, 3, 4))
    minimum = tensor.cummin(dim=-1)
    assert_shape(minimum.values.shape, (2, 3, 4))
    assert_shape(minimum.indices.shape, (2, 3, 4))


def test_cumulative_operations_reject_invalid_dimensions() -> None:
    tensor = torch.randn((2, 3, 4))
    assert_shape(torch.cumsum(tensor, dim=1).shape, (2, 3, 4))

    with assert_raises(IndexError):
        # TODO: BUG: Reject out-of-range cumulative dimensions statically.
        torch.cumsum(tensor, dim=3)

    with assert_raises(IndexError):
        # TODO: BUG: Reject out-of-range cumulative dimensions statically.
        tensor.cummin(dim=-4)


if TYPE_CHECKING:

    def check_cumulative_shapes[Shape: IntTuple](
        tensor: Tensor[Shape], dim: int
    ) -> None:
        assert_type(torch.cumsum(tensor, dim), Tensor[Shape])
        assert_type(tensor.cumprod(dim), Tensor[Shape])
        assert_type(torch.cummax(tensor, dim), torch.return_types.cummax[Shape])
        assert_type(torch.cummax(tensor, dim).values, Tensor[Shape])
        assert_type(tensor.cummin(dim), torch.return_types.cummin[Shape])
        assert_type(tensor.cummin(dim).indices, Tensor[Shape])
