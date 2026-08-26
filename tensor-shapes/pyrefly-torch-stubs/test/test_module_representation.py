# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type, override, TYPE_CHECKING

import torch
import torch.nn as nn
from shape_extensions import assert_shape, Int, IntVar

if TYPE_CHECKING:
    from torch import Tensor


class LinearLayer[N: IntVar, M: IntVar](nn.Module):
    def __init__(self, n: Int[N], m: Int[M]) -> None:
        super().__init__()
        self.linear = nn.Linear(n, m)

    def forward[B: IntVar](self, x: Tensor[[B, N]]) -> Tensor[[B, M]]:
        return self.linear(x)

    @override
    def extra_repr(self) -> str:
        return f"in={self.linear.weight.shape[1]}, out={self.linear.weight.shape[0]}"


class BadExtraRepr(nn.Module):
    @override
    def extra_repr(self) -> int:  # E: inconsistent manner
        return 0


def test_extra_repr_override() -> None:
    layer = LinearLayer(6, 9)
    assert_type(layer.extra_repr(), str)
    assert layer.extra_repr() == "in=6, out=9"
    assert_shape(layer(torch.randn((2, 6))).shape, (2, 9))
