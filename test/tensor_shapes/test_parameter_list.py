# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test nn.ParameterList generic typing."""

from typing import assert_type, TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from torch import Tensor
    from torch_shapes import Dim


class MultiHead[D](nn.Module):
    """Module using ParameterList to store per-head projection weights."""

    def __init__(self, d: Dim[D], n_heads: int) -> None:
        super().__init__()
        self.weights = nn.ParameterList(
            [nn.Parameter(torch.randn(d, d)) for _ in range(n_heads)]
        )

    def forward[B](self, x: Tensor[B, D]) -> Tensor[B, D]:
        return x


def test_parameter_list_len():
    """ParameterList supports len()."""
    m = MultiHead(64, 4)
    x: Tensor[8, 64] = torch.randn(8, 64)
    out = m(x)
    assert_type(out, Tensor[8, 64])
