# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type, TYPE_CHECKING

import torch
import torch.nn as nn
from shape_extensions import assert_shape, Int, IntTuple, IntVar
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


class LinearWithState[N: IntVar, M: IntVar](nn.Module):
    weight: Tensor[[M, N]]
    bias: Tensor[[M]]

    def __init__(self, input_features: Int[N], output_features: Int[M]) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn((output_features, input_features)))
        self.bias = nn.Buffer(torch.randn((output_features,)))

    def forward[B: IntVar](self, tensor: Tensor[[B, N]]) -> Tensor[[B, M]]:
        return torch.matmul(tensor, self.weight.transpose(0, 1)) + self.bias


def test_parameter_preserves_shape() -> None:
    parameter = nn.Parameter(torch.randn((10, 20)))
    assert_shape(parameter.shape, (10, 20))
    assert_type(parameter, Tensor[[10, 20]])

    bare: Tensor = torch.zeros(5)
    bare_parameter = nn.Parameter(bare)
    assert_type(bare_parameter, Tensor)
    # The explicit bare annotation intentionally erases the source shape.
    assert_shape(bare_parameter.shape, IntTuple, runtime=(5,))


def test_module_state_attributes() -> None:
    module = ModuleWithState()
    assert_shape(module.mask.shape, (3, 3))
    assert_shape(module.weight.shape, (3, 3))
    assert_shape(module(torch.randn((3, 3))).shape, (3, 3))


def test_conditional_buffer_attribute() -> None:
    module = ConditionalBuffer(True)
    assert_shape(module(torch.randn((10,))).shape, (10,))


def test_symbolic_parameter_and_buffer_shapes() -> None:
    module = LinearWithState(5, 10)
    assert_shape(module.weight.shape, (10, 5))
    assert_shape(module.bias.shape, (10,))
    assert_shape(module(torch.randn((16, 5))).shape, (16, 10))


if TYPE_CHECKING:

    def check_parameter_with_runtime_extent(extent: int, bare: Tensor) -> None:
        tensor = torch.ones(extent)
        assert_type(tensor, Tensor[[int]])
        assert_type(nn.Parameter(tensor), Tensor[[int]])
        assert_type(nn.Parameter(bare), Tensor)
