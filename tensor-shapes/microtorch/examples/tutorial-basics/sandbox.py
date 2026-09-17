# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type, TYPE_CHECKING

import microtorch as torch
from microtorch import Tensor

if TYPE_CHECKING:
    from shape_extensions import Int, IntVar


class Actor[State: IntVar, Action: IntVar]:
    def __init__(self, state_size: Int[State], action_size: Int[Action]):
        self.w1: Tensor[[State, 128]] = torch.randn((state_size, 128))
        self.w2: Tensor[[128, Action]] = torch.randn((128, action_size))

    def forward[Batch: IntVar](
        self, state: Tensor[[Batch, State]]
    ) -> Tensor[[Batch, Action]]:
        return torch.relu(state @ self.w1) @ self.w2


actor = Actor(24, 4)
action = actor.forward(torch.randn((8, 24)))
assert_type(action, Tensor[[8, 4]])

if TYPE_CHECKING:
    bad_state: Tensor[[8, 10]] = torch.randn((8, 10))
    actor.forward(bad_state)  # E: is not assignable
