# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test Tensor .shape property returns tuple of literal dimensions.

The .shape property on a shaped Tensor should return a tuple type
where each element is Literal[n] for the corresponding dimension.
"""

from typing import assert_type, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor
    from torch_shapes import Dim


def test_shape_2d(x: Tensor[3, 4]) -> tuple[Literal[3], Literal[4]]:
    """Shape of 2D tensor is tuple of two literals"""
    return x.shape


def test_shape_3d(x: Tensor[2, 3, 4]) -> tuple[Literal[2], Literal[3], Literal[4]]:
    """Shape of 3D tensor is tuple of three literals"""
    return x.shape


def test_shape_1d(x: Tensor[10]) -> tuple[Literal[10]]:
    """Shape of 1D tensor is tuple with one literal"""
    return x.shape


def test_shape_5d(
    x: Tensor[1, 2, 3, 4, 5],
) -> tuple[Literal[1], Literal[2], Literal[3], Literal[4], Literal[5]]:
    """Shape of high-rank tensor"""
    return x.shape


# ============================================================================
# Symbolic Dimensions in Shape
# ============================================================================


def test_shape_symbolic[N, M](x: Tensor[N, M]) -> tuple[Dim[N], Dim[M]]:
    """Shape with symbolic dimensions returns Dim types"""
    return x.shape


def test_shape_mixed[N](x: Tensor[N, 3, 4]) -> tuple[Dim[N], Literal[3], Literal[4]]:
    """Shape with mix of symbolic and literal dimensions"""
    return x.shape


def test_shape_arithmetic[N](x: Tensor[N + 1, N * 2]) -> tuple[Dim[N + 1], Dim[N * 2]]:
    """Shape with arithmetic expressions in dimensions"""
    return x.shape
