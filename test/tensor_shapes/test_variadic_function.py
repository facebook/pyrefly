# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test variadic generic function type argument inference"""

from typing import assert_type, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor


# Test case 1: Generic function with TypeVarTuple
def identity[*Ts](x: Tensor[*Ts]) -> Tensor[*Ts]:
    assert_type(x, Tensor[*Ts])
    return x


def test_identity():
    x: Tensor[10, 20] = torch.randn(10, 20)
    y = identity(x)
    assert_type(y, Tensor[10, 20])


# Test case 2: Compare with nn.Parameter
def test_parameter():
    import torch.nn as nn

    x: Tensor[10, 20] = torch.randn(10, 20)
    param = nn.Parameter(x)
    assert_type(param, Tensor[10, 20])
