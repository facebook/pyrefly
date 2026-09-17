# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type, TYPE_CHECKING

import microtorch as torch
from microtorch import Tensor

x = torch.randn((3, 4))
assert_type(x, Tensor[[3, 4]])
assert_type(torch.relu(x + x), Tensor[[3, 4]])

product = torch.randn((2, 3)) @ torch.randn((3, 5))
assert_type(product, Tensor[[2, 5]])

matrix = torch.diagonal(torch.ones((4,)), offset=1)
assert_type(matrix, Tensor[[5, 5]])

joined = torch.concatenate((torch.zeros((2, 3)), torch.ones((2, 4))), axis=1)
assert_type(joined, Tensor[[2, 7]])

if TYPE_CHECKING:
    torch.randn((2, 3)) @ torch.randn((4, 5))  # E: is not assignable
