# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Compare @ operator vs torch.matmul
"""

from typing import assert_type, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor


# Does torch.matmul work?
def test_matmul_function[N, M, K](a: Tensor[N, M], b: Tensor[M, K]):
    """Test torch.matmul with symbolic dimensions"""
    c = torch.matmul(a, b)
    assert_type(c, Tensor[N, K])


test_matmul_function(torch.randn(5, 10), torch.randn(10, 7))


# Does @ work?
def test_at_operator[N, M, K](a: Tensor[N, M], b: Tensor[M, K]):
    """Test @ operator with symbolic dimensions"""
    c = a @ b
    assert_type(c, Tensor[N, K])


test_at_operator(torch.randn(5, 10), torch.randn(10, 7))
