# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test relu shape preservation - for testing IdentityMetaShape removal"""

from typing import assert_type, TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    import torch.nn.functional as F
    from torch import Tensor


def test_relu_function[N, M](x: Tensor[N, M]):
    """Test torch.relu preserves shape"""
    y = torch.relu(x)
    assert_type(y, Tensor[N, M])


def test_relu_method[N, M](x: Tensor[N, M]):
    """Test Tensor.relu preserves shape"""
    y = x.relu()
    assert_type(y, Tensor[N, M])


def test_relu_method_literal(x: Tensor[2, 3, 4]):
    y = x.relu()
    assert_type(y, Tensor[2, 3, 4])


def test_relu_functional[N, M](x: Tensor[N, M]):
    """Test F.relu preserves shape"""
    y = F.relu(x)
    assert_type(y, Tensor[N, M])


def test_relu_with_literals(x: Tensor[2, 3, 4]):
    """Test relu with literal shapes"""
    y = torch.relu(x)
    assert_type(y, Tensor[2, 3, 4])
