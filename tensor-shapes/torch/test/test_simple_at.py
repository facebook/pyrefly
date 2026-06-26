# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Simple test for @ operator"""

from typing import assert_type, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor


def test_at_literal():
    """Test @ with literal dimensions"""
    a: Tensor[5, 10] = torch.randn(5, 10)
    b: Tensor[10, 7] = torch.randn(10, 7)
    c = a @ b
    assert_type(c, Tensor[5, 7])
