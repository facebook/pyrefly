# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Compare regular generics vs Tensor+Dim generics"""

from typing import assert_type, reveal_type, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor


# Test 1: Regular generic function (baseline)
def identity_regular[T](x: T) -> T:
    return x


result1 = identity_regular(5)
reveal_type(result1)  # Returns: int ✅


# Test 2: Generic function with Tensor using Dim
def identity_tensor[N](x: Tensor[N, 3]) -> Tensor[N, 3]:
    return x


import torch

x_concrete: Tensor[2, 3] = torch.randn(2, 3)
result2 = identity_tensor(x_concrete)
reveal_type(result2)  # Returns: Tensor[N, 3] or Tensor[2, 3] ??

# Test what assignment works
correct_assignment: Tensor[2, 3] = result2
wrong_assignment: Tensor[100, 3] = result2  # Should ERROR if result2 is Tensor[2, 3]
