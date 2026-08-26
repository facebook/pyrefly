# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch
from shape_extensions import assert_shape


def test_integer_and_slice_assignment() -> None:
    tensor = torch.randn((10, 20))
    tensor[0] = 1.0
    tensor[1] = torch.ones((20,))
    tensor[0:5] = 0.0
    tensor[:, 0:10] = torch.zeros((10, 10))
    assert_shape(tensor.shape, (10, 20))


def test_boolean_mask_assignment() -> None:
    tensor = torch.randn((32, 100))
    values = torch.randn((32, 1))
    tensor[tensor < values[:, [-1]]] = -float("inf")
    assert_shape(tensor.shape, (32, 100))


def test_multi_axis_assignment() -> None:
    tensor = torch.randn((10, 20, 30))
    tensor[0, :, 5:10] = 1.0
    tensor[:5, 10:, :15] = torch.zeros((5, 10, 15))
    assert_shape(tensor.shape, (10, 20, 30))
