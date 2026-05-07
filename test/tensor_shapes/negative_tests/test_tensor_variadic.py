# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test variadic shape patterns with prefix/middle/suffix.

Generic functions can use TypeVarTuple with prefix and suffix dimensions
to capture variable-length shapes and return derived types.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor


# ============================================================================
# Variadic Identity Functions
# ============================================================================


def variadic_identity[*Ts](x: Tensor[*Ts]) -> Tensor[*Ts]:
    """Identity preserving variadic shape"""
    return x


def test_variadic_identity_2d(x: Tensor[10, 20]) -> Tensor[10, 20]:
    """2D tensor through variadic identity"""
    return variadic_identity(x)


def test_variadic_identity_4d(x: Tensor[1, 2, 3, 4]) -> Tensor[1, 2, 3, 4]:
    """4D tensor through variadic identity"""
    return variadic_identity(x)


# ============================================================================
# Prefix + Middle + Suffix Patterns
# ============================================================================


def with_prefix_suffix[P, *Qs, R, S](x: Tensor[P, *Qs, R, S]) -> Tensor[P, *Qs, R, S]:
    """Function with prefix P, middle *Qs, and suffix R, S"""
    return x


def test_prefix_suffix_6d(x: Tensor[1, 2, 3, 4, 5, 6]) -> Tensor[1, 2, 3, 4, 5, 6]:
    """6D: P=1, Qs=[2,3,4], R=5, S=6"""
    return with_prefix_suffix(x)


def test_prefix_suffix_4d(x: Tensor[10, 20, 30, 40]) -> Tensor[10, 20, 30, 40]:
    """4D: P=10, Qs=[20], R=30, S=40"""
    return with_prefix_suffix(x)


# ============================================================================
# Extract Parts - Return Tuple of Tensors
# ============================================================================


def split_first_rest[N, *Rest](x: Tensor[N, *Rest]) -> tuple[Tensor[N], Tensor[*Rest]]:
    """Split into first dimension and rest"""
    ...


def test_split_first_rest_4d(
    x: Tensor[1, 2, 3, 4],
) -> tuple[Tensor[1], Tensor[2, 3, 4]]:
    """4D: first=1, rest=[2,3,4]"""
    return split_first_rest(x)


def test_split_first_rest_2d(x: Tensor[10, 20]) -> tuple[Tensor[10], Tensor[20]]:
    """2D: first=10, rest=[20]"""
    return split_first_rest(x)


def split_init_last[*Init, N](x: Tensor[*Init, N]) -> tuple[Tensor[*Init], Tensor[N]]:
    """Split into init dimensions and last"""
    ...


def test_split_init_last_4d(x: Tensor[1, 2, 3, 4]) -> tuple[Tensor[1, 2, 3], Tensor[4]]:
    """4D: init=[1,2,3], last=4"""
    return split_init_last(x)


# ============================================================================
# Error Cases
# ============================================================================


def test_variadic_identity_wrong(x: Tensor[10, 20]) -> Tensor[10, 30]:
    """ERROR: shape should be preserved"""
    return variadic_identity(x)


def test_split_first_wrong_rest(
    x: Tensor[1, 2, 3, 4],
) -> tuple[Tensor[1], Tensor[2, 3]]:
    """ERROR: rest should be [2,3,4] not [2,3]"""
    return split_first_rest(x)
