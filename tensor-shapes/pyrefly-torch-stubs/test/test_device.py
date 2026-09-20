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


if TYPE_CHECKING:

    def check_device_context_preserves_shape[Shape: IntTuple](
        device: torch.device, tensor: Tensor[Shape]
    ) -> None:
        with device:
            result = tensor.relu()
        assert_type(result, Tensor[Shape])
