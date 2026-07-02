# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test view/reshape validation errors"""

from typing import assert_type, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor


def test_multiple_minus_ones():
    """Multiple -1s are rejected."""
    x: Tensor[10, 20] = torch.randn(10, 20)
    # E: can only specify one unknown dimension as -1
    y = x.view(-1, -1)
    assert_type(y, Tensor)


def test_incompatible_shape():
    """Incompatible shape with literal dimensions is rejected."""
    x: Tensor[10, 20] = torch.randn(10, 20)  # 200 elements
    # E: could not infer size for dimension -1:
    #    expected 200 to be divisible by 3
    y = x.view(3, -1)
    assert_type(y, Tensor)


def test_invalid_dimension_value():
    """Invalid dimension values like -2 and -3 are rejected."""
    x: Tensor[100] = torch.randn(100)
    # E: invalid negative dimension value (only -1 is allowed)
    y = x.view(-2, 10)
    assert_type(y, Tensor)


def test_zero_dimension():
    """Zero dimension is rejected."""
    x: Tensor[100] = torch.randn(100)
    # E: reshape dimensions cannot contain 0
    y = x.view(0, -1)
    assert_type(y, Tensor)
