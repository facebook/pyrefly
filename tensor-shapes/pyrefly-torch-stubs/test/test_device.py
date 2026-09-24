# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type, TYPE_CHECKING

import torch
from shape_extensions import assert_shape, IntTuple
from torch import Tensor


def test_device_context_manager() -> None:
    device = torch.device("cpu")
    with device as active_device:
        tensor = torch.zeros((3, 4))
    assert_type(active_device, torch.device)
    assert active_device is device
    assert tensor.device.type == "cpu"
    assert_shape(tensor.shape, (3, 4))


def test_type_preserves_shape() -> None:
    tensor = torch.ones((2, 3))
    assert tensor.type() == "torch.FloatTensor"
    assert tensor.type(dtype=None) == "torch.FloatTensor"
    assert tensor.type(non_blocking=True) == "torch.FloatTensor"
    assert_shape(tensor.type("torch.DoubleTensor").shape, (2, 3))


if TYPE_CHECKING:

    def check_device_context_preserves_shape[Shape: IntTuple](
        device: torch.device, tensor: Tensor[Shape], dtype: torch.dtype
    ) -> None:
        with device:
            result = tensor.relu()
        assert_type(result, Tensor[Shape])
        assert_type(tensor.type(), str)
        assert_type(tensor.type(dtype=None), str)
        assert_type(tensor.type(non_blocking=True), str)
        assert_type(tensor.type("torch.DoubleTensor"), Tensor[Shape])
        assert_type(tensor.type(dtype), Tensor[Shape])
