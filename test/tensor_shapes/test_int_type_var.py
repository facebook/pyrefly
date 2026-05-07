# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test torch_shapes.TypeVar for tensor shape dimensions.

torch_shapes.TypeVar is treated identically to typing.TypeVar in pyrefly.
This test verifies that:
1. TypeVar("N") works for shape annotations
2. TypeVarTuple("Ns") works for variadic shapes
3. Generic works with torch_shapes.TypeVar for class-level type parameters
4. Shape arithmetic (N+1, N*2) works in annotations
"""

from typing import assert_type, Generic, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor
    from torch_shapes import TypeVar, TypeVarTuple

N = TypeVar("N")
M = TypeVar("M")
Ns = TypeVarTuple("Ns")


# ============================================================================
# Basic TypeVar usage in function signatures
# ============================================================================


def test_typevar_identity(x: Tensor[N, M]) -> Tensor[N, M]:
    """TypeVar in input and output — same shape"""
    return x


def test_typevar_single(x: Tensor[N]) -> Tensor[N]:
    """Single TypeVar dimension"""
    return x


def test_typevar_inference():
    """TypeVar binds to concrete dims via inference"""
    import torch

    t: Tensor[3, 4] = torch.randn(3, 4)
    result = test_typevar_identity(t)
    assert_type(result, Tensor[3, 4])


# ============================================================================
# TypeVar with arithmetic in shapes
# ============================================================================


def test_typevar_add(x: Tensor[N, M]) -> Tensor[N + 1, M]:
    """N + 1 in return type"""
    return x  # type: ignore[bad-return]


def test_typevar_mul(x: Tensor[N, M]) -> Tensor[N * 2, M]:
    """N * 2 in return type"""
    return x  # type: ignore[bad-return]


def test_typevar_sub(x: Tensor[N, M]) -> Tensor[N - 1, M]:
    """N - 1 in return type"""
    return x  # type: ignore[bad-return]


def test_typevar_two_vars(x: Tensor[N, M]) -> Tensor[N + M, 3]:
    """N + M in return type"""
    return x  # type: ignore[bad-return]


# ============================================================================
# Generic with TypeVar for class-level type parameters
# ============================================================================


class SameShapeLayer(Generic[N]):
    """Class generic over single TypeVar"""

    def forward(self, x: Tensor[N]) -> Tensor[N]:
        return x


def test_class_generic():
    """Generic class with method call — N binds from input"""
    layer = SameShapeLayer()
    import torch

    x: Tensor[3] = torch.randn(3)
    result = layer.forward(x)
    assert_type(result, Tensor[3])


# ============================================================================
# TypeVarTuple in function signatures
# ============================================================================


def test_typevartuple_identity(x: Tensor[*Ns]) -> Tensor[*Ns]:
    """TypeVarTuple preserves shape"""
    return x


def test_typevartuple_inference():
    """TypeVarTuple binds to concrete dims via inference"""
    import torch

    t: Tensor[10, 20] = torch.randn(10, 20)
    result = test_typevartuple_identity(t)
    assert_type(result, Tensor[10, 20])


def test_typevartuple_with_fixed_dim(x: Tensor[*Ns, N]) -> Tensor[*Ns, N]:
    """TypeVarTuple mixed with TypeVar"""
    return x


def test_typevartuple_with_arithmetic(x: Tensor[*Ns, N]) -> Tensor[*Ns, N + 1]:
    """TypeVarTuple with TypeVar arithmetic"""
    return x  # type: ignore[bad-return]


# ============================================================================
# TypeVarTuple with Generic for class-level variadic type parameters
# ============================================================================


class VariadicLayer(Generic[*Ns]):
    """Class generic over TypeVarTuple"""

    def forward(self, x: Tensor[*Ns]) -> Tensor[*Ns]:
        return x


def test_class_typevartuple():
    """Generic class with TypeVarTuple — shape preserved"""
    layer = VariadicLayer()
    import torch

    x: Tensor[2, 3, 4] = torch.randn(2, 3, 4)
    result = layer.forward(x)
    assert_type(result, Tensor[2, 3, 4])
