# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test `torch.device` as a context manager."""

from typing import assert_type, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor


def test_device_context_manager() -> None:
    with torch.device("cpu"):
        x = torch.zeros(3, 4)
    assert_type(x, Tensor[[3, 4]])


def test_device_context_manager_binds_device() -> None:
    with torch.device("cuda", 0) as device:
        assert_type(device, torch.device)


def test_device_instance_as_context_manager(device: torch.device) -> None:
    with device:
        y = torch.ones(2)
    assert_type(y, Tensor[[2]])
