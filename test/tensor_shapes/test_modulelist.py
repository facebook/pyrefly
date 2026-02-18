# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test to understand bare Tensor type"""

from typing import assert_type, TYPE_CHECKING

from torch.nn import Module, ModuleList

if TYPE_CHECKING:
    from torch import Tensor
    from torch_shapes import Dim


class Block[N](Module):
    """Transformer block with self-attention and MLP. Generic over embedding dim, num heads, and block size."""

    def __init__(self, x: Dim[N]):
        super().__init__()
        self.x = x

    def forward[B, T](self, x: Tensor[B, T, N]) -> Tensor[B, T, N]:
        return x


def test_modulelist(modules: ModuleList[Block[4]], x: Tensor[2, 3, 4]):
    y = modules[0](x)
    assert_type(y, Tensor[2, 3, 4])


def test_modulelist_symbolic[B, T, N](
    modules: ModuleList[Block[N]], x: Tensor[B, T, N]
):
    y = modules[0](x)
    assert_type(y, Tensor[B, T, N])


def test_modulelist_symbolic_loop[B, T, N](
    modules: ModuleList[Block[N]], x: Tensor[B, T, N]
):
    assert_type(x, Tensor[B, T, N])
    y = x
    for block in modules:
        y = block(y)
    assert_type(y, Tensor[B, T, N])
