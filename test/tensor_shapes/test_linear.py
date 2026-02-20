# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test to understand bare Tensor type"""

from typing import assert_type, TYPE_CHECKING

import torch
from torch.nn import Linear

if TYPE_CHECKING:
    from torch import Tensor
    from torch_shapes import Dim


def test_linear():
    linear = Linear(3, 4)
    assert_type(linear, Linear[3, 4])


def test_linear_symbolic[N](n: Dim[N]):
    linear = Linear(n, n)
    assert_type(linear, Linear[N, N])


def test_linear_arith[N](n: Dim[N]):
    linear = Linear(n, n * 2)
    t = torch.randn(4, 3, n)
    d = linear(t)
    assert_type(d, Tensor[4, 3, N * 2])
