# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Investigate @ operator bug with symbolic dimensions
"""

from typing import assert_type, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor


# Test 1: Does transpose preserve shapes?
def test_transpose[N, M](x: Tensor[N, M]):
    """Check if transpose returns correct shape"""
    y = x.transpose(0, 1)
    # Should be [M, N]
    assert_type(y, Tensor[M, N])


test_transpose(torch.randn(5, 10))


# Test 2: Does @ work with fully literal dimensions?
def test_matmul_literal():
    """Matmul with literal dimensions (baseline)"""
    a: Tensor[5, 10] = torch.randn(5, 10)
    b: Tensor[10, 7] = torch.randn(10, 7)
    c = a @ b
    assert_type(c, Tensor[5, 7])


# Test 3: Does @ work with symbolic dimensions?
def test_matmul_symbolic[N, M, K](a: Tensor[N, M], b: Tensor[M, K]):
    """Matmul with all symbolic dimensions"""
    c = a @ b
    # Should be [N, K]
    assert_type(c, Tensor[N, K])


test_matmul_symbolic(torch.randn(5, 10), torch.randn(10, 7))


# Test 4: Does @ work with mixed literal and symbolic?
def test_matmul_mixed[N](a: Tensor[N, 10]):
    """Matmul with mixed literal and symbolic"""
    b: Tensor[10, 7] = torch.randn(10, 7)
    c = a @ b
    # Should be [N, 7]
    assert_type(c, Tensor[N, 7])


test_matmul_mixed(torch.randn(5, 10))


# Test 5: Does torch.matmul work differently than @?
def test_torch_matmul[N, M, K](a: Tensor[N, M], b: Tensor[M, K]):
    """Test torch.matmul function"""
    c = torch.matmul(a, b)
    # Should be [N, K]
    assert_type(c, Tensor[N, K])


test_torch_matmul(torch.randn(5, 10), torch.randn(10, 7))
