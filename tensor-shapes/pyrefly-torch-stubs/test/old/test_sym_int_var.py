# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test shape_extensions.IntVar for tensor shape dimensions.

shape_extensions.IntVar marks symbolic integer dimensions in pyrefly.
This test verifies that:
1. IntVar("N") works for shape annotations
2. IntTuple parameters work for variadic shapes
3. Generic works with shape_extensions.IntVar for class-level type parameters
4. Shape arithmetic (N+1, N*2) works in annotations
"""

from typing import assert_type, Generic, TYPE_CHECKING

from shape_extensions import Elements, IntTuple, IntVar

if TYPE_CHECKING:
    from torch import Tensor

N = IntVar("N")
M = IntVar("M")


# ============================================================================
# Basic IntVar usage in function signatures
# ============================================================================


def test_intvar_identity(x: Tensor[[N, M]]) -> Tensor[[N, M]]:
    """IntVar in input and output: same shape"""
    return x


def test_intvar_single(x: Tensor[[N]]) -> Tensor[[N]]:
    """Single IntVar dimension"""
    return x


def test_intvar_inference():
    """IntVar binds to concrete dims via inference"""
    import torch

    t: Tensor[[3, 4]] = torch.randn(3, 4)
    result = test_intvar_identity(t)
    assert_type(result, Tensor[[3, 4]])


# ============================================================================
# IntVar with arithmetic in shapes
# ============================================================================


def test_intvar_add(x: Tensor[[N, M]]) -> Tensor[[N + 1, M]]:
    """N + 1 in return type"""
    return x  # type: ignore[pyrefly:bad-return]


def test_intvar_mul(x: Tensor[[N, M]]) -> Tensor[[N * 2, M]]:
    """N * 2 in return type"""
    return x  # type: ignore[pyrefly:bad-return]


def test_intvar_sub(x: Tensor[[N, M]]) -> Tensor[[N - 1, M]]:
    """N - 1 in return type"""
    return x  # type: ignore[pyrefly:bad-return]


def test_intvar_two_vars(x: Tensor[[N, M]]) -> Tensor[[N + M, 3]]:
    """N + M in return type"""
    return x  # type: ignore[pyrefly:bad-return]


# ============================================================================
# Generic with IntVar for class-level type parameters
# ============================================================================


class SameShapeLayer(Generic[N]):
    """Class generic over single IntVar"""

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
# IntTuple in function signatures
# ============================================================================


def test_inttuple_identity[Ns: IntTuple](x: Tensor[Ns]) -> Tensor[Ns]:
    """An IntTuple parameter preserves the whole shape."""
    return x


def test_inttuple_inference():
    """An IntTuple parameter binds to concrete dimensions through inference."""
    import torch

    t: Tensor[[10, 20]] = torch.randn(10, 20)
    result = test_inttuple_identity(t)
    assert_type(result, Tensor[[10, 20]])


def test_inttuple_with_fixed_dim[Ns: IntTuple, N: IntVar](
    x: Tensor[[*Elements[Ns], N]],
) -> Tensor[[*Elements[Ns], N]]:
    """An IntTuple parameter can be combined with an IntVar."""
    return x


def test_inttuple_with_arithmetic[Ns: IntTuple, N: IntVar](
    x: Tensor[[*Elements[Ns], N]],
) -> Tensor[[*Elements[Ns], N + 1]]:
    """An IntTuple parameter composes with IntVar arithmetic."""
    return x  # type: ignore[pyrefly:bad-return]


# ============================================================================
# IntTuple with Generic for class-level shape parameters
# ============================================================================


class VariadicLayer:
    """Layer with a generic whole-shape method."""

    def forward[Shape: IntTuple](self, x: Tensor[Shape]) -> Tensor[Shape]:
        return x


def test_class_inttuple_parameter():
    """A generic class preserves an IntTuple shape parameter."""
    layer = VariadicLayer()
    import torch

    x: Tensor[[2, 3, 4]] = torch.randn(2, 3, 4)
    result = layer.forward(x)
    assert_type(result, Tensor[[2, 3, 4]])
