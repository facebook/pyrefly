# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test if .matmul() method works vs @ operator"""

from typing import assert_type, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor


def test_matmul_method[N, M, K](a: Tensor[N, M], b: Tensor[M, K]):
    """Test .matmul() method"""
    c = a.matmul(b)
    assert_type(c, Tensor[N, K])


test_matmul_method(torch.randn(5, 10), torch.randn(10, 7))
