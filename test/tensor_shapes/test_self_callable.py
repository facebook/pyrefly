# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Test that Self type is callable in nn.Module subclasses

This tests that when a method in an nn.Module subclass calls self(x),
it properly redirects to self.forward(x) and type checks correctly.
"""

from typing import assert_type, TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from torch import Tensor


class SimpleModule(nn.Module):
    """Simple module that calls self() internally"""

    def __init__(self, size: int) -> None:
        super().__init__()
        self.size = size

    def forward(self, x: Tensor[10, 20]) -> Tensor[10, 20]:
        """Forward pass - just returns input"""
        return x

    def process(self, x: Tensor[10, 20]) -> Tensor[10, 20]:
        """Method that calls self() instead of self.forward()"""
        # This should work - self(x) redirects to self.forward(x)
        result: Tensor[10, 20] = self(x)
        return result


class RecursiveModule(nn.Module):
    """Module that calls itself recursively"""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor[5, 10], depth: int) -> Tensor[5, 10]:
        """Forward pass that recursively calls self"""
        if depth > 0:
            # Recursive self call
            intermediate: Tensor[5, 10] = self(x, depth - 1)
            return intermediate
        return x


class NestedCallModule(nn.Module):
    """Module with multiple methods that call self"""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor[3, 4]) -> Tensor[3, 4]:
        """Forward pass"""
        return x * 2

    def apply_twice(self, x: Tensor[3, 4]) -> Tensor[3, 4]:
        """Apply forward twice using self()"""
        once: Tensor[3, 4] = self(x)
        twice: Tensor[3, 4] = self(once)
        return twice

    def apply_and_add(self, x: Tensor[3, 4], y: Tensor[3, 4]) -> Tensor[3, 4]:
        """Apply forward to both inputs and add"""
        x_out: Tensor[3, 4] = self(x)
        y_out: Tensor[3, 4] = self(y)
        result: Tensor[3, 4] = x_out + y_out
        return result


def test_simple_self_call():
    """Test basic self() call in a method"""
    module = SimpleModule(10)
    x: Tensor[10, 20] = torch.randn(10, 20)

    # Call via process method which uses self()
    assert_type(module.process(x), Tensor[10, 20])


def test_recursive_self_call():
    """Test recursive self() calls"""
    module = RecursiveModule()
    x: Tensor[5, 10] = torch.randn(5, 10)

    # Recursive call
    assert_type(module(x, 3), Tensor[5, 10])


def test_nested_self_calls():
    """Test multiple self() calls in methods"""
    module = NestedCallModule()
    x: Tensor[3, 4] = torch.randn(3, 4)
    y: Tensor[3, 4] = torch.randn(3, 4)

    # Call methods that use self() internally
    assert_type(module.apply_twice(x), Tensor[3, 4])
    assert_type(module.apply_and_add(x, y), Tensor[3, 4])
