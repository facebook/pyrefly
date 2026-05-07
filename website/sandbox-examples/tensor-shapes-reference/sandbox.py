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


# Dim arithmetic: +, -, *, //
def dim_math[A, B](a: Dim[A], b: Dim[B]) -> None:
    assert_type(a + b, Dim[A + B])
    assert_type(a * b, Dim[A * B])
    assert_type(a - b, Dim[A - B])
    assert_type(a // b, Dim[A // B])


# Tensor with arithmetic in shape annotations
def concat_channels[B, C1, C2, H, W](
    x: Tensor[B, C1, H, W], y: Tensor[B, C2, H, W]
) -> Tensor[B, C1 + C2, H, W]:
    return torch.cat([x, y], dim=1)


out = concat_channels(torch.randn(1, 3, 8, 8), torch.randn(1, 5, 8, 8))
assert_type(out, Tensor[1, 8, 8, 8])


# Transpose produces correct reordered shape
def identity_like[M, N](x: Tensor[M, N]) -> Tensor[N, M]:
    return x.transpose(0, 1)


flipped = identity_like(torch.randn(3, 7))
assert_type(flipped, Tensor[7, 3])


# ERROR: simplification catches mismatches
def bad_math[N](x: Tensor[N * 2]) -> Tensor[N]:
    return x  # Tensor[N * 2] is not Tensor[N]
