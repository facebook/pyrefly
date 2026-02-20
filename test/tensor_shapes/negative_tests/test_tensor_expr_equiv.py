# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test subtyping compatibility between equivalent shape expressions.

Shape expressions that simplify to the same value should be compatible.
For example, Tensor[2 + 3] should be assignable to Tensor[5].
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor


# ============================================================================
# Literal Expression Equivalence
# ============================================================================


def literal_add_equals_literal(x: Tensor[2 + 3, 4]) -> Tensor[5, 4]:
    """2 + 3 simplifies to 5"""
    return x


def literal_mul_equals_literal(x: Tensor[2 * 3, 4]) -> Tensor[6, 4]:
    """2 * 3 simplifies to 6"""
    return x


def literal_complex_expr(x: Tensor[2 * 3 + 1, 4]) -> Tensor[7, 4]:
    """2 * 3 + 1 = 7"""
    return x


def literal_sub_equals_literal(x: Tensor[10 - 3, 4]) -> Tensor[7, 4]:
    """10 - 3 = 7"""
    return x


# ============================================================================
# Symbolic Expression Equivalence (Commutativity)
# ============================================================================


def add_commutative[N, M](x: Tensor[N + M]) -> Tensor[M + N]:
    """Addition is commutative: N + M = M + N"""
    return x


def mul_commutative[N, M](x: Tensor[N * M]) -> Tensor[M * N]:
    """Multiplication is commutative: N * M = M * N"""
    return x


# ============================================================================
# Expression with Concrete and Symbolic
# ============================================================================


def concrete_plus_symbolic[N](x: Tensor[N + 0]) -> Tensor[N]:
    """N + 0 = N (additive identity)"""
    return x


def concrete_times_one[N](x: Tensor[N * 1]) -> Tensor[N]:
    """N * 1 = N (multiplicative identity)"""
    return x


def double_is_times_two[N](x: Tensor[N + N]) -> Tensor[N * 2]:
    """N + N = N * 2"""
    return x


# ============================================================================
# Expression Equivalence Errors (Non-equivalent expressions)
# ============================================================================


def add_not_equal_mul[N, M](x: Tensor[N + M]) -> Tensor[N * M]:
    """ERROR: N + M != N * M in general"""
    return x


def different_constants[N](x: Tensor[N + 1]) -> Tensor[N + 2]:
    """ERROR: N + 1 != N + 2"""
    return x


def wrong_literal_simplification(x: Tensor[2 + 3, 4]) -> Tensor[6, 4]:
    """ERROR: 2 + 3 = 5, not 6"""
    return x
