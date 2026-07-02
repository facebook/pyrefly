# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test that .shape property returns literal dimensions like .size()"""

from typing import assert_type, Literal

import torch
from torch import Tensor


def test_shape_returns_literal_tuple():
    """shape property returns tuple of literal ints"""
    x: Tensor[3, 4, 5] = torch.randn(3, 4, 5)
    shape = x.shape
    assert_type(shape, tuple[Literal[3], Literal[4], Literal[5]])


def test_shape_matches_size():
    """shape and size() return the same type"""
    x: Tensor[2, 7, 4] = torch.randn(2, 7, 4)
    assert_type(x.shape, tuple[Literal[2], Literal[7], Literal[4]])
    assert_type(x.size(), tuple[Literal[2], Literal[7], Literal[4]])


def test_shape_single_dimension():
    """shape works with single dimension tensor"""
    x: Tensor[10] = torch.randn(10)
    assert_type(x.shape, tuple[Literal[10]])


def test_shape_high_rank():
    """shape works with high rank tensors"""
    x: Tensor[2, 3, 4, 5, 6] = torch.randn(2, 3, 4, 5, 6)
    assert_type(
        x.shape, tuple[Literal[2], Literal[3], Literal[4], Literal[5], Literal[6]]
    )
