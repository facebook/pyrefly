# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test shape_extensions.SymVar for tensor shape dimensions.

shape_extensions.SymVar is treated identically to typing.TypeVar in pyrefly.
This test verifies that:
1. SymVar("N") works for shape annotations
2. SizeTuple carriers work for variadic shapes
3. Generic works with shape_extensions.SymVar for class-level type parameters
4. Shape arithmetic (N+1, N*2) works in annotations
"""

from typing import assert_type, Generic, TYPE_CHECKING

from shape_extensions import Elements, SizeTuple, SymVar

if TYPE_CHECKING:
    from torch import Tensor

N = SymVar("N")
M = SymVar("M")


# ============================================================================
# Basic TypeVar usage in function signatures
# ============================================================================


def test_typevar_identity(x: Tensor[[N, M]]) -> Tensor[[N, M]]:
    """TypeVar in input and output — same shape"""
    return x


def test_typevar_single(x: Tensor[[N]]) -> Tensor[[N]]:
    """Single TypeVar dimension"""
    return x


def test_typevar_inference():
    """TypeVar binds to concrete dims via inference"""
    import torch

    t: Tensor[[3, 4]] = torch.randn(3, 4)
    result = test_typevar_identity(t)
    assert_type(result, Tensor[[3, 4]])


# ============================================================================
# TypeVar with arithmetic in shapes
# ============================================================================


def test_typevar_add(x: Tensor[[N, M]]) -> Tensor[[N + 1, M]]:
    """N + 1 in return type"""
    return x  # type: ignore[bad-return]


def test_typevar_mul(x: Tensor[[N, M]]) -> Tensor[[N * 2, M]]:
    """N * 2 in return type"""
    return x  # type: ignore[bad-return]


def test_typevar_sub(x: Tensor[[N, M]]) -> Tensor[[N - 1, M]]:
    """N - 1 in return type"""
    return x  # type: ignore[bad-return]


def test_typevar_two_vars(x: Tensor[[N, M]]) -> Tensor[[N + M, 3]]:
    """N + M in return type"""
    return x  # type: ignore[bad-return]


# ============================================================================
# Generic with TypeVar for class-level type parameters
# ============================================================================


class SameShapeLayer(Generic[N]):
    """Class generic over single TypeVar"""

    def forward(self, x: Tensor[[N]]) -> Tensor[[N]]:
        return x


def test_class_generic():
    """Generic class with method call — N binds from input"""
    layer = SameShapeLayer()
    import torch

    x: Tensor[[3]] = torch.randn(3)
    result = layer.forward(x)
    assert_type(result, Tensor[[3]])


# ============================================================================
# SizeTuple carrier in function signatures
# ============================================================================


def test_sizetuple_identity[Ns: SizeTuple](x: Tensor[Ns]) -> Tensor[Ns]:
    """SizeTuple carrier preserves shape"""
    return x


def test_sizetuple_inference():
    """SizeTuple carrier binds to concrete dims via inference"""
    import torch

    t: Tensor[[10, 20]] = torch.randn(10, 20)
    result = test_sizetuple_identity(t)
    assert_type(result, Tensor[[10, 20]])


def test_sizetuple_with_fixed_dim[Ns: SizeTuple, N: SymVar](
    x: Tensor[[*Elements[Ns], N]],
) -> Tensor[[*Elements[Ns], N]]:
    """SizeTuple carrier mixed with TypeVar"""
    return x


def test_sizetuple_with_arithmetic[Ns: SizeTuple, N: SymVar](
    x: Tensor[[*Elements[Ns], N]],
) -> Tensor[[*Elements[Ns], N + 1]]:
    """SizeTuple carrier with TypeVar arithmetic"""
    return x  # type: ignore[bad-return]


# ============================================================================
# SizeTuple carrier with Generic for class-level shape parameters
# ============================================================================


class VariadicLayer:
    """Layer with a generic SizeTuple carrier method"""

    def forward[Shape: SizeTuple](self, x: Tensor[Shape]) -> Tensor[Shape]:
        return x


def test_class_sizetuple_carrier():
    """Generic class with SizeTuple carrier — shape preserved"""
    layer = VariadicLayer()
    import torch

    x: Tensor[[2, 3, 4]] = torch.randn(2, 3, 4)
    result = layer.forward(x)
    assert_type(result, Tensor[[2, 3, 4]])
