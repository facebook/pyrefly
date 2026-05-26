# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type

import torch
from torch import Tensor

# Tensor types carry their shape as type parameters
x = torch.randn(3, 4)
assert_type(x, Tensor[3, 4])


# Shape-polymorphic function with variadic batch dims
def add_bias[*Batch, D](x: Tensor[*Batch, D], bias: Tensor[D]) -> Tensor[*Batch, D]:
    return x + bias


y = add_bias(torch.randn(2, 5, 8), torch.randn(8))
assert_type(y, Tensor[2, 5, 8])

# Transpose tracks dimension reordering
m = torch.randn(3, 5)
mt = m.transpose(0, 1)
assert_type(mt, Tensor[5, 3])


# ERROR: return type has wrong shape -- pyrefly catches it!
def broken[B, D](x: Tensor[B, D]) -> Tensor[D, B]:
    return x  # Tensor[B, D] is not Tensor[D, B]
