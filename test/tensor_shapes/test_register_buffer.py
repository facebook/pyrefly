# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Test that nn.Buffer and nn.Parameter create instance attributes

This tests that when Buffer or Parameter are assigned in __init__,
Pyrefly correctly tracks the instance attributes.
"""

from typing import assert_type, TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from torch import Tensor


class ModuleWithBuffer(nn.Module):
    """Module that uses Buffer"""

    def __init__(self, size: int) -> None:
        super().__init__()
        # Use nn.Buffer to create self.mask
        self.mask = nn.Buffer(torch.ones(size, size))

    def forward(self, x: Tensor[10, 20]) -> Tensor:
        """Forward pass that accesses the buffer"""
        # Should not error - self.mask exists
        mask: Tensor = self.mask
        return x * mask


class ModuleWithParameter(nn.Module):
    """Module that uses Parameter"""

    def __init__(self, dim: int) -> None:
        super().__init__()
        # Use nn.Parameter to create self.weight
        self.weight = nn.Parameter(torch.randn(dim, dim))

    def forward(self, x: Tensor[5, 5]) -> Tensor[5, 5]:
        """Forward pass that accesses the parameter"""
        # Should not error - self.weight exists
        weight: Tensor = self.weight
        return x @ weight


class ModuleWithMultipleBuffers(nn.Module):
    """Module with multiple buffers and parameters"""

    def __init__(self) -> None:
        super().__init__()
        self.buffer1 = nn.Buffer(torch.zeros(3, 3))
        self.buffer2 = nn.Buffer(torch.ones(3, 3))
        self.param1 = nn.Parameter(torch.randn(3, 3))

    def forward(self, x: Tensor[3, 3]) -> Tensor[3, 3]:
        """Use all attributes"""
        b1: Tensor = self.buffer1
        b2: Tensor = self.buffer2
        p1: Tensor = self.param1
        return x * b1 + b2 * p1


class ConditionalBuffer(nn.Module):
    """Module with conditionally created buffer"""

    def __init__(self, use_bias: bool) -> None:
        super().__init__()
        if use_bias:
            # Conditionally create buffer
            self.bias = nn.Buffer(torch.zeros(10))

    def forward(self, x: Tensor[10]) -> Tensor[10]:
        """Access conditionally created buffer"""
        # This should be allowed - Pyrefly tracks it
        bias: Tensor = self.bias
        return x + bias


def test_buffer():
    """Test basic Buffer"""
    module = ModuleWithBuffer(10)
    x: Tensor[10, 20] = torch.randn(10, 20)
    assert_type(module(x), Tensor)


def test_parameter():
    """Test basic Parameter"""
    module = ModuleWithParameter(5)
    x: Tensor[5, 5] = torch.randn(5, 5)
    assert_type(module(x), Tensor[5, 5])


def test_multiple():
    """Test multiple Buffer/Parameter"""
    module = ModuleWithMultipleBuffers()
    x: Tensor[3, 3] = torch.randn(3, 3)
    assert_type(module(x), Tensor[3, 3])


def test_conditional_buffer():
    """Test conditionally created buffer"""
    module = ConditionalBuffer(True)
    x: Tensor[10] = torch.randn(10)
    assert_type(module(x), Tensor[10])
