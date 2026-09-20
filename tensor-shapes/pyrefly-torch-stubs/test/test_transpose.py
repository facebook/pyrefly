# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type, TYPE_CHECKING

import torch
from shape_extensions import assert_raises, assert_shape, Elements, IntTuple, IntVar
from torch import Tensor


def test_transpose_shapes() -> None:
    matrix = torch.ones((2, 3))
    assert_shape(torch.transpose(matrix, 0, 1).shape, (3, 2))
    assert_shape(matrix.transpose(1, 0).shape, (3, 2))

    tensor = torch.ones((2, 3, 4))
    assert_shape(torch.transpose(tensor, -1, 0).shape, (4, 3, 2))
    assert_shape(tensor.transpose(1, 1).shape, (2, 3, 4))


def test_transpose_scalar() -> None:
    scalar = torch.tensor(1)
    assert_shape(torch.transpose(scalar, 0, -1).shape, ())
    assert_shape(scalar.transpose(-1, 0).shape, ())


def test_transpose_rejects_invalid_dimensions() -> None:
    tensor = torch.ones((2, 3))
    assert_shape(tensor.transpose(0, 1).shape, (3, 2))

    with assert_raises(IndexError):
        # E: Cannot evaluate type-level shape DSL call: transpose dimension out of range
        torch.transpose(tensor, 0, 2)


if TYPE_CHECKING:

    def check_symbolic[N: IntVar, M: IntVar](x: Tensor[[N, M]]) -> None:
        assert_type(torch.transpose(x, 0, 1), Tensor[[M, N]])
        assert_type(x.transpose(-1, 0), Tensor[[M, N]])

    def check_gradual_axis(x: Tensor[[2, 3]], dim: int) -> None:
        assert_type(x.transpose(dim, 0), Tensor[IntTuple])

    def check_open_shape[Ts: IntTuple](x: Tensor[[*Elements[Ts], 3]]) -> None:
        assert_type(x.transpose(-1, -1), Tensor[[*Elements[Ts], 3]])
        assert_type(torch.transpose(x, 0, 0), Tensor[[*Elements[Ts], 3]])
        assert_type(x.transpose(-1, 0), Tensor[IntTuple])
