# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test Tensor subtyping rules.

These tests verify that Tensor types follow proper subtyping:
- Tensor[2, 3] is NOT a subtype of Tensor[4, 3] (different dimensions)
- Tensor[N, 3] with N=2 substitutes correctly
- Shapeless Tensor is compatible with any shaped Tensor
- Variance in dimensions is covariant
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor


# ============================================================================
# Basic Tensor Subtyping - Literal Dimensions
# ============================================================================


def tensor_identity_2_3(x: Tensor[2, 3]) -> Tensor[2, 3]:
    """Identity - same shape in and out"""
    return x  # OK


def tensor_wrong_first_dim(x: Tensor[2, 3]) -> Tensor[4, 3]:
    """ERROR: First dimension mismatch"""
    return x  # ERROR: Tensor[2, 3] not assignable to Tensor[4, 3]


def tensor_wrong_second_dim(x: Tensor[2, 3]) -> Tensor[2, 5]:
    """ERROR: Second dimension mismatch"""
    return x  # ERROR: Tensor[2, 3] not assignable to Tensor[2, 5]


def tensor_wrong_rank(x: Tensor[2, 3]) -> Tensor[2, 3, 4]:
    """ERROR: Rank mismatch"""
    return x  # ERROR: Tensor[2, 3] not assignable to Tensor[2, 3, 4]


def tensor_to_shapeless(x: Tensor[2, 3]) -> Tensor:
    """OK: Any shaped tensor is subtype of shapeless"""
    return x  # OK


def shapeless_to_shaped(x: Tensor) -> Tensor[2, 3]:
    """OK: Shapeless is compatible with any shape (widening)"""
    return x  # OK


# ============================================================================
# Generic Tensor Functions - Type Variable Substitution
# ============================================================================


def tensor_generic_identity[N, M](x: Tensor[N, M]) -> Tensor[N, M]:
    """Generic identity preserves shape"""
    return x  # OK


def tensor_generic_wrong_order[N, M](x: Tensor[N, M]) -> Tensor[M, N]:
    """ERROR: Swapped dimensions"""
    return x  # ERROR: Tensor[N, M] not assignable to Tensor[M, N]


def tensor_generic_first_dim[N](x: Tensor[N, 3]) -> Tensor[N, 3]:
    """Generic in first dimension only"""
    return x  # OK


def tensor_generic_first_dim_wrong[N](x: Tensor[N, 3]) -> Tensor[N, 5]:
    """ERROR: Second dimension mismatch even with generic first"""
    return x  # ERROR: Tensor[N, 3] not assignable to Tensor[N, 5]


# ============================================================================
# Arithmetic Expressions in Dimensions
# ============================================================================


def tensor_add_dims[N, M](x: Tensor[N, M]) -> Tensor[N + M]:
    """ERROR: Can't just return 2D tensor as 1D with sum dimension"""
    return x  # ERROR: Tensor[N, M] not assignable to Tensor[N + M]


def tensor_same_arithmetic[N](x: Tensor[N + 1, 3]) -> Tensor[N + 1, 3]:
    """OK: Same arithmetic expression"""
    return x  # OK


def tensor_different_arithmetic[N](x: Tensor[N + 1, 3]) -> Tensor[N + 2, 3]:
    """ERROR: Different arithmetic expression"""
    return x  # ERROR: N + 1 not equal to N + 2


def tensor_mul_dims[N, M](x: Tensor[N * M, 3]) -> Tensor[N * M, 3]:
    """OK: Same multiplication expression"""
    return x  # OK


def tensor_add_vs_mul[N, M](x: Tensor[N + M, 3]) -> Tensor[N * M, 3]:
    """ERROR: Addition vs multiplication"""
    return x  # ERROR: N + M not equal to N * M


# ============================================================================
# Nested Generic Calls - Substitution Through Calls
# ============================================================================


def call_generic_identity(x: Tensor[2, 3]) -> Tensor[2, 3]:
    """Call generic identity - should substitute N=2, M=3"""
    return tensor_generic_identity(x)  # OK: returns Tensor[2, 3]


def call_generic_wrong_return(x: Tensor[2, 3]) -> Tensor[4, 3]:
    """ERROR: Generic identity returns Tensor[2, 3], not Tensor[4, 3]"""
    return tensor_generic_identity(x)  # ERROR
