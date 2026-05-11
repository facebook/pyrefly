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


# Dim arithmetic: compute dimensions at the type level
def split_and_combine[D](x: Tensor[D], half: Dim[D // 2]) -> Tensor[D // 2]:
    return torch.randn(half)


a = torch.randn(8)
result = split_and_combine(a, 4)
assert_type(result, Tensor[4])


# Dim values compose through functions
def double_dim[N](n: Dim[N]) -> Dim[N * 2]:
    return n * 2


doubled = double_dim(5)
assert_type(doubled, Dim[10])


# Use Dim to build tensors with matching shapes
def make_pair[D](d: Dim[D]) -> tuple[Tensor[D], Tensor[D, D]]:
    return torch.randn(d), torch.randn(d, d)


vec, mat = make_pair(4)
assert_type(vec, Tensor[4])
assert_type(mat, Tensor[4, 4])

# ERROR: wrong assert_type -- doubled is Dim[10], not Dim[20]
assert_type(doubled, Dim[20])
