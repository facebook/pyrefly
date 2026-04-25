# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test that view/reshape gracefully handle variadic (*Bs) tensor shapes.

The view DSL computes prod(self.shape) for -1 inference. When the tensor has
variadic batch dims (*Bs), prod must return Unsupported rather than panicking.
"""

from typing import assert_type, TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from torch import Tensor
    from torch_shapes import Dim


# --- view on Linear output with variadic *Bs ---


class Reshaper[K, D](nn.Module):
    """Linear whose out_features is a Dim expression, followed by view."""

    def __init__(self, k: Dim[K], d: Dim[D]) -> None:
        super().__init__()
        self.k = k
        self.d = d
        self.proj = nn.Linear(256, k * d)

    def forward[B](self, x: Tensor[B, 256]) -> Tensor[B, K, D]:
        # proj(x) returns Tensor[*Bs, K*D] — *Bs is unresolved variadic.
        # view should fall back to bare rather than crashing.
        p = self.proj(x)
        # Annotation fallback: view can't infer -1 from variadic shape
        out: Tensor[B, K, D] = p.view(-1, self.k, self.d)
        return out


def test_view_on_variadic_linear():
    """view() on Linear output with Dim expression doesn't crash."""
    m = Reshaper(16, 8)
    x: Tensor[4, 256] = torch.randn(4, 256)
    out = m(x)
    assert_type(out, Tensor[4, 16, 8])


# --- reshape on explicitly variadic function param ---


def reshape_variadic[*Bs, C](x: Tensor[*Bs, C], c: Dim[C]) -> Tensor[*Bs, C]:
    """reshape on a variadic tensor should not crash."""
    y = x.reshape(-1, c)
    # y is bare (can't infer -1 from variadic); annotation fallback
    result: Tensor[*Bs, C] = y
    return result


def test_reshape_variadic_param():
    """reshape() on explicitly variadic tensor doesn't crash."""
    x: Tensor[2, 3, 10] = torch.randn(2, 3, 10)
    out = reshape_variadic(x, 10)
    assert_type(out, Tensor[2, 3, 10])
