# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test Tensor subtyping rules.

These tests verify that Tensor types follow proper subtyping:
- Tensor[[2, 3]] is NOT a subtype of Tensor[[4, 3]] (different dimensions)
- Tensor[[N, 3]] with N=2 substitutes correctly
- Shapeless Tensor is compatible with any shaped Tensor
- Shape dimensions and expressions must be compatible
"""

from typing import TYPE_CHECKING

from shape_extensions import IntVar


if TYPE_CHECKING:
    from torch import Tensor


# ============================================================================
# Basic Tensor Subtyping - Literal Dimensions
# ============================================================================


def tensor_identity_2_3(x: Tensor[[2, 3]]) -> Tensor[[2, 3]]:
    """Identity - same shape in and out"""
    return x


def tensor_wrong_first_dim(x: Tensor[[2, 3]]) -> Tensor[[4, 3]]:
    """First dimension mismatch."""
    # E: Returned type `Tensor[[2, 3]]` is not assignable
    #    to declared return type `Tensor[[4, 3]]`
    return x


def tensor_wrong_second_dim(x: Tensor[[2, 3]]) -> Tensor[[2, 5]]:
    """Second dimension mismatch."""
    # E: Returned type `Tensor[[2, 3]]` is not assignable
    #    to declared return type `Tensor[[2, 5]]`
    return x


def tensor_wrong_rank(x: Tensor[[2, 3]]) -> Tensor[[2, 3, 4]]:
    """Rank mismatch."""
    # E: Returned type `Tensor[[2, 3]]` is not assignable
    #    to declared return type `Tensor[[2, 3, 4]]`
    return x


def tensor_to_shapeless(x: Tensor[[2, 3]]) -> Tensor:
    """Any shaped tensor is subtype of shapeless."""
    return x


def shapeless_to_shaped(x: Tensor) -> Tensor[[2, 3]]:
    """Shapeless Tensor is accepted where a specific shape is expected."""
    return x


# ============================================================================
# Generic Tensor Functions - Type Variable Substitution
# ============================================================================


def tensor_generic_identity[N: IntVar, M: IntVar](
    x: Tensor[[N, M]],
) -> Tensor[[N, M]]:
    """Generic identity preserves shape"""
    return x


def tensor_generic_wrong_order[N: IntVar, M: IntVar](
    x: Tensor[[N, M]],
) -> Tensor[[M, N]]:
    """Swapped dimensions."""
    # E: Returned type `Tensor[[N, M]]` is not assignable
    #    to declared return type `Tensor[[M, N]]`
    return x


def tensor_generic_first_dim[N: IntVar](x: Tensor[[N, 3]]) -> Tensor[[N, 3]]:
    """Generic in first dimension only"""
    return x


def tensor_generic_first_dim_wrong[N: IntVar](x: Tensor[[N, 3]]) -> Tensor[[N, 5]]:
    """Second dimension mismatch even with generic first."""
    # E: Returned type `Tensor[[N, 3]]` is not assignable
    #    to declared return type `Tensor[[N, 5]]`
    return x


# ============================================================================
# Arithmetic Expressions in Dimensions
# ============================================================================


def tensor_add_dims[N: IntVar, M: IntVar](x: Tensor[[N, M]]) -> Tensor[[N + M]]:
    """Cannot return a 2D tensor as 1D with sum dimension."""
    # E: Returned type `Tensor[[N, M]]` is not assignable
    #    to declared return type `Tensor[[(N + M)]]`
    return x


def tensor_same_arithmetic[N: IntVar](x: Tensor[[N + 1, 3]]) -> Tensor[[N + 1, 3]]:
    """Same arithmetic expression."""
    return x


def tensor_different_arithmetic[N: IntVar](
    x: Tensor[[N + 1, 3]],
) -> Tensor[[N + 2, 3]]:
    """Different arithmetic expression."""
    # E: Returned type `Tensor[[(1 + N), 3]]` is not assignable
    #    to declared return type `Tensor[[(2 + N), 3]]`
    return x


def tensor_mul_dims[N: IntVar, M: IntVar](
    x: Tensor[[N * M, 3]],
) -> Tensor[[N * M, 3]]:
    """Same multiplication expression."""
    return x


def tensor_add_vs_mul[N: IntVar, M: IntVar](
    x: Tensor[[N + M, 3]],
) -> Tensor[[N * M, 3]]:
    """Addition vs multiplication."""
    # E: Returned type `Tensor[[(N + M), 3]]` is not assignable
    #    to declared return type `Tensor[[(N * M), 3]]`
    return x


# ============================================================================
# Nested Generic Calls - Substitution Through Calls
# ============================================================================


def call_generic_identity(x: Tensor[[2, 3]]) -> Tensor[[2, 3]]:
    """Call generic identity - should substitute N=2, M=3"""
    return tensor_generic_identity(x)


def call_generic_wrong_return(x: Tensor[[2, 3]]) -> Tensor[[4, 3]]:
    """Generic identity returns Tensor[[2, 3]], not Tensor[[4, 3]]."""
    # E: Returned type `Tensor[[2, 3]]` is not assignable
    #    to declared return type `Tensor[[4, 3]]`
    return tensor_generic_identity(x)
