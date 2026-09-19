# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import assert_type

import torch


def test_device_context_manager() -> None:
    device = torch.device("cpu")
    with device as active_device:
        assert_type(active_device, torch.device)


def test_eq_scalar() -> None:
    assert_type(torch.eq(torch.ones(2, 3), 0), torch.Tensor[[2, 3]])
