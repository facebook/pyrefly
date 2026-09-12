# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test `nn.Module.forward` declared as a callable attribute."""

from typing import Any, assert_type, override, TYPE_CHECKING

import torch
import torch.nn as nn
from shape_extensions import Int, IntVar

if TYPE_CHECKING:
    from torch import Tensor


class LinearLayer[N: IntVar, M: IntVar](nn.Module):
    def __init__(self, n: Int[N], m: Int[M]) -> None:
        super().__init__()
        self.linear = nn.Linear(n, m)

    @override
    def forward[B: IntVar](self, x: Tensor[[B, N]]) -> Tensor[[B, M]]:
        return self.linear(x)


class Passthrough(nn.Module):
    """Transform-style module, like torchvision's `Transform.forward(self, *inputs)`."""

    @override
    def forward(self, *inputs: Any) -> Any:
        return inputs


def test_forward_override_keeps_call_proxy() -> None:
    x: Tensor[[16, 6]] = torch.randn(16, 6)
    layer = LinearLayer(6, 9)
    assert_type(layer(x), Tensor[[16, 9]])
    assert_type(layer.forward(x), Tensor[[16, 9]])


def test_call_through_base_module_type() -> None:
    x: Tensor[[16, 6]] = torch.randn(16, 6)
    module: nn.Module = LinearLayer(6, 9)
    assert_type(module(x), Any)


def test_forward_override_with_star_args() -> None:
    module = Passthrough()
    module(torch.zeros(2), torch.zeros(3))
