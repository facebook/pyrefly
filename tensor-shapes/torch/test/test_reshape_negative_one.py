# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test .reshape() with -1 for automatic dimension"""

from typing import assert_type, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor


def test_reshape_with_minus_one():
    """Test .reshape() with -1"""
    x: Tensor[10, 20] = torch.randn(10, 20)
    y = x.reshape(-1)  # Flatten
    assert_type(y, Tensor[200])
    assert_type(y, Tensor[200])


def test_reshape_with_partial_minus_one():
    """Test .reshape() with partial -1"""
    x: Tensor[10, 20] = torch.randn(10, 20)
    y = x.reshape(2, -1)  # Should infer second dim as 100
    assert_type(y, Tensor[2, 100])
    assert_type(y, Tensor[2, 100])


def test_reshape_1d_to_4d[C](x: Tensor[C]) -> Tensor[1, C, 1, 1]:
    """Test reshaping 1D to 4D with -1"""
    y = x.reshape(1, -1, 1, 1)  # Should be [1, C, 1, 1]
    assert_type(y, Tensor[1, C, 1, 1])
    assert_type(y, Tensor[1, C, 1, 1])
    return y


# Test with concrete tensor
result = test_reshape_1d_to_4d(torch.randn(64))
assert_type(result, Tensor[1, 64, 1, 1])
