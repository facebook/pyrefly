# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Test overload-based narrowing as an alternative to generic parameter narrowing.
"""

from typing import assert_type, overload, TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from torch import Tensor
    from torch_shapes import Dim


# ============================================================================
# Test 1: Regular generics with overload (the user's example)
# ============================================================================


@overload
def foo(x: int) -> int: ...


@overload
def foo[T](x: T) -> T: ...


def foo[T](x: int | T) -> int | T:
    if isinstance(x, int):
        return x + 1
    return x


def test_foo():
    x = foo("hello")
    assert_type(x, str)
    y = foo(3)
    assert_type(y, int)


# ============================================================================
# Test 2: dense_chain with overloads
# ============================================================================


class GenericDenseLayer(nn.Module):
    """DenseNet layer: adds 32 channels via concatenation."""

    def forward[B, C, H, W](self, x: Tensor[B, C, H, W]) -> Tensor[B, C + 32, H, W]: ...  # type: ignore[return-type]


@overload
def dense_chain[B, C, H, W](
    x: Tensor[B, C, H, W],
    layer: GenericDenseLayer,
    depth: Dim[1],
) -> Tensor[B, C + 32, H, W]: ...


@overload
def dense_chain[I, B, C, H, W](
    x: Tensor[B, C, H, W],
    layer: GenericDenseLayer,
    depth: Dim[I],
) -> Tensor[B, C + I * 32, H, W]: ...


def dense_chain[I, B, C, H, W](
    x: Tensor[B, C, H, W],
    layer: GenericDenseLayer,
    depth: Dim[I],
) -> Tensor[B, C + 32, H, W] | Tensor[B, C + I * 32, H, W]:
    if depth == 1:
        return layer(x)
    y = layer(x)
    return dense_chain(y, layer, depth - 1)


def test_dense_chain():
    """Test DenseNet-style linear channel accumulation with overloads."""
    layer = GenericDenseLayer()
    x: Tensor[2, 64, 32, 32] = torch.randn(2, 64, 32, 32)
    y = dense_chain(x, layer, 6)
    assert_type(y, Tensor[2, 256, 32, 32])  # 64 + 6*32 = 256


def test_dense_chain_one():
    """Test base case: depth=1 applies one layer."""
    layer = GenericDenseLayer()
    x: Tensor[2, 64, 32, 32] = torch.randn(2, 64, 32, 32)
    y = dense_chain(x, layer, 1)
    assert_type(y, Tensor[2, 96, 32, 32])  # 64 + 32 = 96
