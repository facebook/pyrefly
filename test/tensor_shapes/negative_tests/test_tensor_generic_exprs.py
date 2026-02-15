# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test generic function substitution with tensor shape expressions.

When calling a generic function with concrete tensor shapes, type variables
should be substituted and expressions should be evaluated.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor


# ============================================================================
# Generic Functions with Expression Return Types
# ============================================================================


def sum_dims[N, M](x: Tensor[N, M]) -> Tensor[N + M]:
    """Returns 1D tensor with dimension N + M"""
    ...


def product_dims[N, M](x: Tensor[N, M]) -> Tensor[N * M]:
    """Returns 1D tensor with dimension N * M"""
    ...


def double_first[N, M](x: Tensor[N, M]) -> Tensor[N * 2, M]:
    """Returns tensor with first dimension doubled"""
    ...


def add_one[N, M](x: Tensor[N, M]) -> Tensor[N + 1, M + 1]:
    """Returns tensor with both dimensions increased by 1"""
    ...


# ============================================================================
# Concrete Substitution Tests
# ============================================================================


def test_sum_dims_concrete(x: Tensor[2, 3]) -> Tensor[5]:
    """N=2, M=3 -> N+M = 5"""
    return sum_dims(x)


def test_product_dims_concrete(x: Tensor[2, 3]) -> Tensor[6]:
    """N=2, M=3 -> N*M = 6"""
    return product_dims(x)


def test_double_first_concrete(x: Tensor[4, 5]) -> Tensor[8, 5]:
    """N=4, M=5 -> N*2=8, M=5"""
    return double_first(x)


def test_add_one_concrete(x: Tensor[3, 4]) -> Tensor[4, 5]:
    """N=3, M=4 -> N+1=4, M+1=5"""
    return add_one(x)


# ============================================================================
# Symbolic Substitution - Preserving Expressions
# ============================================================================


def test_sum_dims_symbolic[A, B](x: Tensor[A, B]) -> Tensor[A + B]:
    """Symbolic input -> symbolic output with expression"""
    return sum_dims(x)


def test_product_dims_symbolic[A, B](x: Tensor[A, B]) -> Tensor[A * B]:
    """Symbolic input -> symbolic output with expression"""
    return product_dims(x)


def test_double_first_symbolic[A, B](x: Tensor[A, B]) -> Tensor[A * 2, B]:
    """Symbolic input -> symbolic output with expression"""
    return double_first(x)


# ============================================================================
# Chained Generic Calls
# ============================================================================


def flatten_to_1d[N, M](x: Tensor[N, M]) -> Tensor[N * M]:
    """Flatten 2D to 1D"""
    ...


def duplicate[K](x: Tensor[K]) -> Tensor[K * 2]:
    """Duplicate the dimension"""
    ...


def test_chained_concrete(x: Tensor[3, 4]) -> Tensor[24]:
    """3*4=12, 12*2=24"""
    flat = flatten_to_1d(x)
    return duplicate(flat)


def test_chained_symbolic[N, M](x: Tensor[N, M]) -> Tensor[N * M * 2]:
    """N*M -> (N*M)*2"""
    flat = flatten_to_1d(x)
    return duplicate(flat)


# ============================================================================
# Wrong Return Type Errors
# ============================================================================


def test_sum_dims_wrong(x: Tensor[2, 3]) -> Tensor[6]:
    """ERROR: N+M=5, not 6"""
    return sum_dims(x)  # ERROR


def test_product_dims_wrong(x: Tensor[2, 3]) -> Tensor[5]:
    """ERROR: N*M=6, not 5"""
    return product_dims(x)  # ERROR


def test_double_first_wrong(x: Tensor[4, 5]) -> Tensor[4, 5]:
    """ERROR: First dim should be 8, not 4"""
    return double_first(x)  # ERROR
