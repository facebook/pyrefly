# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test view/reshape validation errors"""

from typing import assert_type, reveal_type, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor


def test_multiple_minus_ones():
    """Multiple -1s should error"""
    x: Tensor[10, 20] = torch.randn(10, 20)
    y = x.view(-1, -1)  # ERROR: can only specify one unknown dimension as -1
    assert_type(y, Tensor)


def test_incompatible_shape():
    """Incompatible shape with literal dimensions should error"""
    x: Tensor[10, 20] = torch.randn(10, 20)  # 200 elements
    y = x.view(3, -1)  # ERROR: 200 / 3 is not an integer
    assert_type(y, Tensor)


def test_invalid_dimension_value():
    """Invalid dimension values like -2, -3 should error"""
    x: Tensor[100] = torch.randn(100)
    y = x.view(-2, 10)  # ERROR: invalid dimension value
    assert_type(y, Tensor)


def test_zero_dimension():
    """Zero dimension should error"""
    x: Tensor[100] = torch.randn(100)
    y = x.view(0, -1)  # ERROR: reshape dimensions cannot contain 0
    assert_type(y, Tensor)
