# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type, TYPE_CHECKING

import torch
from shape_extensions import assert_raises, assert_shape, Elements, IntTuple, IntVar
from torch import Tensor


def test_squeeze_shapes() -> None:
    tensor = torch.ones((1, 2, 1, 3))
    assert_shape(torch.squeeze(tensor).shape, (2, 3))
    assert_shape(tensor.squeeze(0).shape, (2, 1, 3))
    assert_shape(torch.squeeze(tensor, -2).shape, (1, 2, 3))
    assert_shape(tensor.squeeze(1).shape, (1, 2, 1, 3))

    scalar = torch.tensor(1)
    assert_shape(torch.squeeze(scalar, 0).shape, ())
    assert_shape(scalar.squeeze(-1).shape, ())


def test_squeeze_multiple_dimensions() -> None:
    tensor = torch.ones((1, 2, 1, 3))
    assert_shape(torch.squeeze(tensor, (0, 2)).shape, (2, 3))
    assert_shape(tensor.squeeze((-4, -2)).shape, (2, 3))
    assert_shape(torch.squeeze(tensor, ()).shape, (1, 2, 1, 3))

    with assert_raises(RuntimeError):
        torch.squeeze(tensor, (0, 0))  # E: duplicate squeeze dimension

    with assert_raises(IndexError):
        tensor.squeeze((0, 4))  # E: squeeze dimension out of range

    scalar = torch.tensor(1)
    with assert_raises(RuntimeError):
        scalar.squeeze((0, -1))  # E: duplicate squeeze dimension


def test_squeeze_rejects_invalid_dimensions() -> None:
    matrix = torch.ones((2, 3))
    assert_shape(matrix.squeeze(0).shape, (2, 3))

    with assert_raises(IndexError):
        torch.squeeze(matrix, 2)  # E: squeeze dimension out of range

    scalar = torch.tensor(1)
    with assert_raises(IndexError):
        scalar.squeeze(1)  # E: squeeze dimension out of range


if TYPE_CHECKING:

    def check_symbolic_suffix[Shape: IntTuple](
        x: Tensor[[*Elements[Shape], 3]],
    ) -> None:
        assert_type(torch.squeeze(x, -1), Tensor[[*Elements[Shape], 3]])

    def check_symbolic_last_extent[N: IntVar](x: Tensor[[2, N]]) -> None:
        # The result rank depends on whether `N` is one, so it is gradual.
        assert_type(torch.squeeze(x, -1), Tensor[IntTuple])

    def check_gradual_boundaries(x: Tensor[[2, 1, 3]], dim: int, bare: Tensor) -> None:
        assert_type(torch.squeeze(x, dim), Tensor[IntTuple])
        assert_type(bare.squeeze(), Tensor[IntTuple])
