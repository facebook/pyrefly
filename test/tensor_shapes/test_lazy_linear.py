# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test nn.LazyLinear shape preservation.

LazyLinear accepts any input features (in_features inferred at first forward)
but preserves out_features in the output shape.
"""

from typing import assert_type, TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from torch import Tensor
    from torch_shapes import Dim


def test_lazy_linear_preserves_out_features():
    """LazyLinear output has known out_features dim."""
    proj = nn.LazyLinear(128)
    x: Tensor[4, 256] = torch.randn(4, 256)
    out = proj(x)
    assert_type(out, Tensor[4, 128])


def test_lazy_linear_batched():
    """LazyLinear preserves batch dims via variadic *Bs."""
    proj = nn.LazyLinear(64)
    x: Tensor[2, 8, 512] = torch.randn(2, 8, 512)
    out = proj(x)
    assert_type(out, Tensor[2, 8, 64])
