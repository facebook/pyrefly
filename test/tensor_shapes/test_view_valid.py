# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test view with just -1"""

from typing import assert_type, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor


def test_view_negative_one_alone():
    """Test view with single -1"""
    x: Tensor[10, 20] = torch.randn(10, 20)
    y = x.view(-1)
    assert_type(y, Tensor[200])  # Should be Tensor[200]


def test_view_with_positive_and_negative():
    """Test view with mix of positive and negative"""
    x: Tensor[10, 20] = torch.randn(10, 20)
    y = x.view(10, -1)
    assert_type(y, Tensor[10, 20])  # Should be Tensor[10, 20]


def test_view_all_positive():
    """Test view with all positive"""
    x: Tensor[10, 20] = torch.randn(10, 20)
    y = x.view(4, 50)
    assert_type(y, Tensor[4, 50])  # Should be Tensor[4, 50]
