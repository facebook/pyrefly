# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test that bare Tensor and nn.Parameter(bare_tensor) have consistent representations"""

from typing import assert_type, TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from torch import Tensor

# Test that all these are compatible and have the same representation
t1 = torch.ones(10)  # bare Tensor from runtime creation
t2: Tensor = torch.zeros(5)  # bare Tensor from annotation
p1 = nn.Parameter(t1)  # Should be bare Tensor after simplification
p2 = nn.Parameter(t2)  # Should be bare Tensor after simplification

# Test assignability: all should be assignable to each other
v1: Tensor = t1  # bare → bare
v2: Tensor = p1  # param(bare) → bare
v3: Tensor = t2  # bare → bare
v4: Tensor = p2  # param(bare) → bare


# Reverse direction
def takes_tensor(x: Tensor) -> None:
    pass


takes_tensor(t1)  # bare → bare
takes_tensor(t2)  # bare → bare
takes_tensor(p1)  # param(bare) → bare
takes_tensor(p2)  # param(bare) → bare

# All should display as bare Tensor or with known shape
assert_type(t1, Tensor[10])
assert_type(t2, Tensor)
assert_type(p1, Tensor[10])
assert_type(p2, Tensor)
