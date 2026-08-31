# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test that view/reshape gracefully handle variadic (*Bs) tensor shapes.

The view DSL computes prod(self.shape) for -1 inference. When the tensor has
variadic batch dims (*Bs), it preserves the fixed target rank and known dimensions.
"""

from typing import TYPE_CHECKING, assert_type

import torch
import torch.nn as nn
from shape_extensions import Elements, IntTuple, IntVar

if TYPE_CHECKING:
    from shape_extensions import Int
    from torch import Tensor


# --- view on Linear output with variadic *Bs ---


class Reshaper[K: IntVar, D: IntVar](nn.Module):
    """Linear whose out_features is a Int expression, followed by view."""

    def __init__(self, k: Int[K], d: Int[D]) -> None:
        super().__init__()
        self.k = k
        self.d = d
        self.proj = nn.Linear(256, k * d)

    def forward[B: IntVar](self, x: Tensor[[B, 256]]) -> Tensor[[B, K, D]]:
        # proj(x) returns Tensor[[*Elements[Bs], K*D]] — *Elements[Bs] is unresolved variadic.
        p = self.proj(x)
        out: Tensor[[B, K, D]] = p.view(-1, self.k, self.d)
        return out


def test_view_on_variadic_linear():
    """view() on Linear output with Int expression doesn't crash."""
    m = Reshaper(16, 8)
    x: Tensor[[4, 256]] = torch.randn(4, 256)
    out = m(x)
    assert_type(out, Tensor[[4, 16, 8]])


# --- reshape on explicitly variadic function param ---


def reshape_variadic[Bs: IntTuple, C: IntVar](
    x: Tensor[[*Elements[Bs], C]], c: Int[C]
) -> Tensor[[int, C]]:
    """reshape on a variadic tensor should not crash."""
    y = x.reshape(-1, c)
    assert_type(y, Tensor[[int, C]])
    return y


def test_reshape_variadic_param():
    """reshape() on explicitly variadic tensor doesn't crash."""
    x: Tensor[[2, 3, 10]] = torch.randn(2, 3, 10)
    out = reshape_variadic(x, 10)
    assert_type(out, Tensor[[int, 10]])
