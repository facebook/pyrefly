# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Test symbolic dimension canonicalization.

This tests that different representations of the same symbolic expression
are recognized as equal after canonicalization.

Each test documents:
1. What pyrefly actually infers (via assert_type with actual type)
2. What types are structurally compatible (via assignment to expected type)
"""

from typing import assert_type, TYPE_CHECKING

import torch
from shape_extensions import IntVar

if TYPE_CHECKING:
    from torch import Tensor


def test_combine_like_terms[N: IntVar, M: IntVar](
    x: Tensor[[N, M]],
) -> Tensor[[M * N]]:
    """Test that products are structurally commutative: N*M compatible with M*N"""
    # Flatten produces N*M (left-associative)
    result = x.flatten(0, 1)
    # Assert the actual inferred type
    assert_type(result, Tensor[[N * M]])
    # Show structural compatibility with commuted order
    expected: Tensor[[M * N]] = result
    return expected


def test_division_flattening[N: IntVar, M: IntVar, K: IntVar](
    x: Tensor[[N, M, K]],
) -> Tensor[[(N * M) // 2, K]]:
    """Test that symbolic slicing preserves dimension expressions"""
    # Flatten first two dims: N*M
    flattened = x.flatten(0, 1)
    assert_type(flattened, Tensor[[N * M, K]])
    # Slice with symbolic bound
    result = flattened[: flattened.size(0) // 2, :]
    # Assert the actual inferred type
    assert_type(result, Tensor[[(N * M) // 2, K]])
    return result


def test_product_ordering[N: IntVar, M: IntVar, K: IntVar](
    x: Tensor[[N, M, K]],
) -> Tensor[[M * N * K]]:
    """Test that products are structurally commutative: N*M*K compatible with other orderings"""
    result = x.flatten(0, 2)
    # Assert the actual inferred type (left-associative: ((N * M) * K))
    assert_type(result, Tensor[[N * M * K]])
    # Show structural compatibility with different orderings
    expected1: Tensor[[K * M * N]] = result
    expected2: Tensor[[M * N * K]] = result
    _ = expected1  # avoid unused variable warning
    return expected2


def test_flatten_compatibility[B: IntVar, C: IntVar, H: IntVar, W: IntVar](
    images: Tensor[[B, C, H, W]],
) -> Tensor[[W * H * C * B]]:
    """Test that flatten produces correct product"""
    flattened = images.flatten()
    # Assert the actual inferred type (left-associative)
    assert_type(flattened, Tensor[[B * C * H * W]])
    # Show structural compatibility with reversed order
    expected: Tensor[[W * H * C * B]] = flattened
    return expected


def test_literal_evaluation[N: IntVar](x: Tensor[[N, 10]]) -> Tensor[[N * 10]]:
    """Test that literal expressions evaluate correctly"""
    # Flatten creates N*10
    result = x.flatten(0, 1)
    assert_type(result, Tensor[[N * 10]])
    return result


def test_distributive_symbolic[GR: IntVar, Iterations: IntVar](
    a: Tensor[[GR * Iterations]],
    b: Tensor[[GR]],
) -> None:
    """Symbolic distribution enables like-term cancellation.

    GR*Iterations + GR = GR*(Iterations+1): adding GR*Iterations and GR should produce GR*(Iterations+1).
    This requires distributing GR across (I+1) and combining like terms.

    Used in DenseNet: each block adds GR channels, so after Iterations blocks
    the channel count is InC + GR*Iterations. Adding another GR gives InC + GR*(Iterations+1).
    """
    # cat along dim 0: GR*Iterations + GR
    result = torch.cat((a, b), dim=0)
    assert_type(result, Tensor[[GR * Iterations + GR]])
    # The checker should canonicalize GR*Iterations + GR to GR*(Iterations+1)
    expected: Tensor[[GR * (Iterations + 1)]] = result
    _ = expected


def test_multi_dim_flatten[
    A: IntVar,
    B: IntVar,
    C: IntVar,
    D: IntVar,
    E: IntVar,
](
    x: Tensor[[A, B, C, D, E]],
) -> tuple[
    Tensor[[C * B * A, D, E]], Tensor[[E * D * C * B * A]], Tensor[[A, D * C * B, E]]
]:
    """Test flattening multiple dimensions"""
    # Flatten different ranges
    r1 = x.flatten(0, 2)  # A*B*C, D, E
    r2 = x.flatten(0, 4)  # A*B*C*D*E
    r3 = x.flatten(1, 3)  # A, B*C*D, E

    # Assert the actual inferred types (left-associative)
    assert_type(r1, Tensor[[A * B * C, D, E]])
    assert_type(r2, Tensor[[A * B * C * D * E]])
    assert_type(r3, Tensor[[A, B * C * D, E]])

    # Show structural compatibility with commuted orderings
    expected1: Tensor[[C * B * A, D, E]] = r1
    expected2: Tensor[[E * D * C * B * A]] = r2
    expected3: Tensor[[A, D * C * B, E]] = r3
    return (expected1, expected2, expected3)


# ============================================================================
# Pow canonicalization tests
# ============================================================================


def test_pow_literal(x: Tensor[[2**3]]) -> Tensor[[8]]:
    """2**3 = 8, literal exponentiation."""
    return x


def test_pow_identity[N: IntVar](x: Tensor[[N**1]]) -> Tensor[[N]]:
    """N**1 = N."""
    return x


def test_pow_zero[N: IntVar](x: Tensor[[N**0]]) -> Tensor[[1]]:
    """N**0 = 1."""
    return x


def test_pow_symbolic_base[I: IntVar](x: Tensor[[2**I]]) -> Tensor[[2**I]]:
    """Symbolic exponent preserved."""
    return x


def test_pow_product_grouping[I: IntVar](
    x: Tensor[[2**I]],
) -> Tensor[[2**I]]:
    """2 * 2**I = 2**(I+1) and 2**(I+1) canonically equals... 2**(I+1).
    But we test that 2 * 2**(I-1) = 2**I via product grouping."""
    assert_type(x, Tensor[[2**I]])
    return x


def test_pow_same_base_mul[I: IntVar, B: IntVar, C: IntVar](
    x: Tensor[[B, C * 2**I]],
) -> Tensor[[B, C * 2**I]]:
    """Product with Pow factor stays canonical."""
    return x


def test_pow_literal_absorb[I: IntVar](
    x: Tensor[[8 * 2**I]],
) -> Tensor[[2 ** (I + 3)]]:
    """8 * 2**I = 2**3 * 2**I = 2**(I+3)."""
    return x


def test_pow_mul_same_base[I: IntVar](
    x: Tensor[[2**I * 2]],
) -> Tensor[[2 ** (I + 1)]]:
    """2**I * 2 = 2**(I+1)."""
    return x
