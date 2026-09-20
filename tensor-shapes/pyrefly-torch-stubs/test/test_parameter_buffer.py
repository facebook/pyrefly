# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type, TYPE_CHECKING

import torch
import torch.nn as nn
from shape_extensions import assert_shape
from torch import Tensor


class ModuleWithState(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mask = nn.Buffer(torch.ones((3, 3)))
        self.weight = nn.Parameter(torch.randn((3, 3)))

    def forward(self, x: Tensor[[3, 3]]) -> Tensor[[3, 3]]:
        return x * self.mask + self.weight


class ConditionalBuffer(nn.Module):
    def __init__(self, enabled: bool) -> None:
        super().__init__()
        if enabled:
            self.bias = nn.Buffer(torch.zeros((10,)))

    def forward(self, x: Tensor[[10]]) -> Tensor[[10]]:
        return x + self.bias


def test_parameter_preserves_shape() -> None:
    parameter = nn.Parameter(torch.randn((10, 20)))
    assert_shape(parameter.shape, (10, 20))
    assert_type(parameter, Tensor[[10, 20]])


def test_module_state_attributes() -> None:
    module = ModuleWithState()
    assert_shape(module.mask.shape, (3, 3))
    assert_shape(module.weight.shape, (3, 3))
    assert_shape(module(torch.randn((3, 3))).shape, (3, 3))


def test_conditional_buffer_attribute() -> None:
    module = ConditionalBuffer(True)
    assert_shape(module(torch.randn((10,))).shape, (10,))


if TYPE_CHECKING:

    def check_parameter_with_runtime_extent(extent: int, bare: Tensor) -> None:
        tensor = torch.ones(extent)
        assert_type(tensor, Tensor[[int]])
        assert_type(nn.Parameter(tensor), Tensor[[int]])
        assert_type(nn.Parameter(bare), Tensor)
