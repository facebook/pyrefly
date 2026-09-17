# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type, TYPE_CHECKING

import torch
import torch.nn as nn
from shape_extensions import assert_raises, assert_shape, IntTuple, IntVar
from torch import Tensor


def test_flatten_function_method_and_module_shapes() -> None:
    tensor = torch.ones((2, 3, 4, 5))
    assert_shape(tensor.flatten().shape, (120,))
    assert_shape(torch.flatten(tensor, start_dim=1, end_dim=2).shape, (2, 12, 5))
    assert_shape(tensor.flatten(-2).shape, (2, 3, 20))
    assert_shape(tensor.flatten(2, 2).shape, (2, 3, 4, 5))

    assert_shape(nn.Flatten()(tensor).shape, (2, 60))
    assert_shape(nn.Flatten(0, 1)(tensor).shape, (6, 4, 5))
    assert_shape(nn.Flatten(start_dim=2, end_dim=3)(tensor).shape, (2, 3, 20))


def test_flatten_scalars_and_vectors() -> None:
    scalar = torch.tensor(1)
    assert_shape(scalar.flatten().shape, (1,))
    assert_shape(torch.flatten(scalar, 0, -1).shape, (1,))
    assert_shape(nn.Flatten(0)(scalar).shape, (1,))

    vector = torch.ones(7)
    assert_shape(vector.flatten().shape, (7,))
    assert_shape(nn.Flatten(0)(vector).shape, (7,))


def test_flatten_module_reuse() -> None:
    flatten = nn.Flatten(1, -1)
    assert_shape(flatten(torch.ones((2, 3, 4))).shape, (2, 12))
    assert_shape(flatten(torch.ones((5, 6, 7, 8))).shape, (5, 336))


def test_flatten_rejects_invalid_dimensions() -> None:
    tensor = torch.ones((2, 3, 4))
    assert_shape(tensor.flatten(1).shape, (2, 12))

    with assert_raises(IndexError):
        tensor.flatten(3)  # E: flatten start_dim out of range

    with assert_raises(IndexError):
        tensor.flatten(0, -4)  # E: flatten end_dim out of range

    with assert_raises(RuntimeError):
        torch.flatten(tensor, 2, 1)  # E: flatten start_dim cannot come after end_dim

    scalar = torch.tensor(1)
    with assert_raises(IndexError):
        scalar.flatten(1)  # E: flatten dimension out of range for scalar input

    with assert_raises(IndexError):
        nn.Flatten()(scalar)  # E: flatten dimension out of range for scalar input


if TYPE_CHECKING:

    def check_symbolic[B: IntVar, C: IntVar, H: IntVar, W: IntVar](
        tensor: Tensor[[B, C, H, W]],
    ) -> None:
        assert_type(tensor.flatten(), Tensor[[B * C * H * W]])
        assert_type(torch.flatten(tensor, 1), Tensor[[B, C * H * W]])
        assert_type(nn.Flatten(1, 2)(tensor), Tensor[[B, C * H, W]])

    def check_gradual(tensor: Tensor[IntTuple], start_dim: int, end_dim: int) -> None:
        assert_type(tensor.flatten(), Tensor[IntTuple])
        assert_type(torch.flatten(tensor, start_dim, end_dim), Tensor[IntTuple])
        assert_type(nn.Flatten(start_dim, end_dim)(tensor), Tensor[IntTuple])
