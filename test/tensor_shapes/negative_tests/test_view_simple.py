# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test if negative literals match int type"""

from typing import reveal_type, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor


def test_view_with_negative_one():
    """Test view with -1"""
    x: Tensor[10, 20] = torch.randn(10, 20)
    y = x.view(-1)  # Should work: -1 infers the dimension
    _ = y  # y should be Tensor[200]


def test_view_with_zero():
    """Test view with 0"""
    x: Tensor[10, 20] = torch.randn(10, 20)
    y = x.view(0, -1)  # Should ERROR but still return Tensor
    reveal_type(y)
    _ = y  # mark as used
