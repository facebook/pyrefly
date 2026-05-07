# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test nn.Parameter with unknown shapes"""

from typing import assert_type, TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from torch import Tensor

# Test 1: Parameter with known shape
x: Tensor[10, 20] = torch.randn(10, 20)
p1 = nn.Parameter(x)
assert_type(p1, Tensor[10, 20])


# Test 2: Parameter with unknown shape (from runtime value)
def create_param(ndim: int):
    # torch.ones returns Tensor (unknown shape)
    t = torch.ones(ndim)
    assert_type(t, Tensor)

    p = nn.Parameter(t)
    assert_type(p, Tensor)

    return p


# Test 3: What is the actual type of Tensor without args?
def test_bare_tensor():
    t: Tensor = torch.zeros(5)
    assert_type(t, Tensor)
