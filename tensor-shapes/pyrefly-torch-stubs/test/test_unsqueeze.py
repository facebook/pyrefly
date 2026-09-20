# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type, TYPE_CHECKING

import torch
from shape_extensions import assert_raises, assert_shape, Elements, IntTuple
from torch import Tensor


def test_unsqueeze_shapes() -> None:
    matrix = torch.ones((2, 3))
    assert_shape(torch.unsqueeze(matrix, 0).shape, (1, 2, 3))
    assert_shape(matrix.unsqueeze(1).shape, (2, 1, 3))
    assert_shape(torch.unsqueeze(matrix, 2).shape, (2, 3, 1))
    assert_shape(matrix.unsqueeze(-1).shape, (2, 3, 1))
    assert_shape(torch.unsqueeze(matrix, -3).shape, (1, 2, 3))

    scalar = torch.tensor(1)
    assert_shape(scalar.unsqueeze(0).shape, (1,))
    assert_shape(torch.unsqueeze(scalar, -1).shape, (1,))


def test_unsqueeze_chaining() -> None:
    vector = torch.ones(3)
    assert_shape(vector.unsqueeze(0).unsqueeze(0).shape, (1, 1, 3))


def test_unsqueeze_rejects_invalid_dimensions() -> None:
    matrix = torch.ones((2, 3))
    assert_shape(matrix.unsqueeze(0).shape, (1, 2, 3))

    with assert_raises(IndexError):
        torch.unsqueeze(matrix, 3)  # E: unsqueeze dimension out of range

    with assert_raises(IndexError):
        matrix.unsqueeze(-4)  # E: unsqueeze dimension out of range


if TYPE_CHECKING:

    def check_symbolic_suffix[Shape: IntTuple](
        x: Tensor[[*Elements[Shape], 3]],
    ) -> None:
        assert_type(x.unsqueeze(-1), Tensor[[*Elements[Shape], 3, 1]])
        assert_type(torch.unsqueeze(x, -1), Tensor[[*Elements[Shape], 3, 1]])

    def check_gradual_boundaries(x: Tensor[[2, 3]], dim: int, bare: Tensor) -> None:
        assert_type(x.unsqueeze(dim), Tensor[IntTuple])
        assert_type(torch.unsqueeze(bare, 0), Tensor[IntTuple])
