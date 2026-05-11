# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor
    from torch_shapes import Dim


# Step 1: Use Dim to capture dimensions at construction
class Actor[S, A]:
    def __init__(self, state_size: Dim[S], action_size: Dim[A]):
        self.w1: Tensor[S, 128] = torch.randn(state_size, 128)
        self.w2: Tensor[128, A] = torch.randn(128, action_size)

    # Step 2: Method-level type param B for batch size
    def forward[B](self, state: Tensor[B, S]) -> Tensor[B, A]:
        h: Tensor[B, 128] = torch.matmul(state, self.w1)
        h = torch.relu(h)
        return torch.matmul(h, self.w2)


# Step 3: Verify shapes with concrete dimensions
actor = Actor(24, 4)
state = torch.randn(8, 24)
action = actor.forward(state)
assert_type(action, Tensor[8, 4])

# ERROR: wrong input shape -- state is [8, 24], not [8, 10]
bad_state: Tensor[8, 10] = torch.randn(8, 10)
actor.forward(bad_state)
