# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test .view() and .reshape() with symbolic dimensions"""

from typing import assert_type, reveal_type, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor


def test_view_literals():
    """Test .view() with literal dimensions"""
    x: Tensor[10, 20] = torch.randn(10, 20)
    y = x.view(2, 5, 20)
    assert_type(y, Tensor[2, 5, 20])
    assert_type(y, Tensor[2, 5, 20])


def test_reshape_literals():
    """Test .reshape() with literal dimensions"""
    x: Tensor[10, 20] = torch.randn(10, 20)
    y = x.reshape(2, 5, 20)
    assert_type(y, Tensor[2, 5, 20])
    assert_type(y, Tensor[2, 5, 20])


def test_view_symbolic[N, M](x: Tensor[N, M]) -> Tensor[2, N // 2, M]:
    """Test .view() with symbolic dimensions

    Takes a tensor with shape [N, M] where N is divisible by 2,
    reshapes it to [2, N//2, M]

    The .size() method returns Dim[N] and Dim[M], which can be used
    in arithmetic operations and passed to .view() for shape transformation.
    """
    # Get both dimensions from input - these return Dim[N] and Dim[M]
    n = x.size(0)
    m = x.size(1)
    reveal_type(n // 2)
    reveal_type(m)
    # Reshape: split N into 2 and N//2, keep M
    # The meta-shape system tracks the symbolic dimensions
    return x.view(2, n // 2, m)


# Test by calling with concrete tensor
result = test_view_symbolic(torch.randn(10, 20))
assert_type(result, Tensor[2, (10 // 2), 20])
assert_type(result, Tensor[2, 5, 20])
