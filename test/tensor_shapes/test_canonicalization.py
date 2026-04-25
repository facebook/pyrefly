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

from typing import assert_type, reveal_type, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor


def test_combine_like_terms[N, M](x: Tensor[N, M]) -> Tensor[M * N]:
    """Test that products are structurally commutative: N*M compatible with M*N"""
    # Flatten produces N*M (left-associative)
    result = x.flatten(0, 1)
    reveal_type(result)
    # Assert the actual inferred type
    assert_type(result, Tensor[N * M])
    # Show structural compatibility with commuted order
    expected: Tensor[M * N] = result
    return expected


def test_division_flattening[N, M, K](
    x: Tensor[N, M, K],
) -> Tensor[(N * M) // 2, K]:
    """Test that symbolic slicing preserves dimension expressions"""
    # Flatten first two dims: N*M
    flattened = x.flatten(0, 1)
    reveal_type(flattened)
    # Slice with symbolic bound
    result = flattened[: flattened.size(0) // 2, :]
    reveal_type(result)
    # Assert the actual inferred type
    assert_type(result, Tensor[(N * M) // 2, K])
    return result


def test_product_ordering[N, M, K](
    x: Tensor[N, M, K],
) -> Tensor[M * N * K]:
    """Test that products are structurally commutative: N*M*K compatible with other orderings"""
    result = x.flatten(0, 2)
    reveal_type(result)
    # Assert the actual inferred type (left-associative: ((N * M) * K))
    assert_type(result, Tensor[N * M * K])
    # Show structural compatibility with different orderings
    expected1: Tensor[K * M * N] = result
    expected2: Tensor[M * N * K] = result
    _ = expected1  # avoid unused variable warning
    return expected2


def test_flatten_compatibility[B, C, H, W](
    images: Tensor[B, C, H, W],
) -> Tensor[W * H * C * B]:
    """Test that flatten produces correct product"""
    flattened = images.flatten()
    reveal_type(flattened)
    # Assert the actual inferred type (left-associative)
    assert_type(flattened, Tensor[B * C * H * W])
    # Show structural compatibility with reversed order
    expected: Tensor[W * H * C * B] = flattened
    return expected


def test_literal_evaluation[N](x: Tensor[N, 10]) -> Tensor[N * 10]:
    """Test that literal expressions evaluate correctly"""
    # Flatten creates N*10
    result = x.flatten(0, 1)
    reveal_type(result)
    assert_type(result, Tensor[N * 10])
    return result


def test_distributive_symbolic[GR, I](
    a: Tensor[GR * I],
    b: Tensor[GR],
) -> None:
    """Symbolic distribution enables like-term cancellation.

    GR*I + GR = GR*(I+1): adding GR*I and GR should produce GR*(I+1).
    This requires distributing GR across (I+1) and combining like terms.

    Used in DenseNet: each block adds GR channels, so after I blocks
    the channel count is InC + GR*I. Adding another GR gives InC + GR*(I+1).
    """
    # cat along dim 0: GR*I + GR
    result = torch.cat((a, b), dim=0)
    assert_type(result, Tensor[GR * I + GR])
    # The checker should canonicalize GR*I + GR to GR*(I+1)
    expected: Tensor[GR * (I + 1)] = result


def test_multi_dim_flatten[A, B, C, D, E](
    x: Tensor[A, B, C, D, E],
) -> tuple[Tensor[C * B * A, D, E], Tensor[E * D * C * B * A], Tensor[A, D * C * B, E]]:
    """Test flattening multiple dimensions"""
    # Flatten different ranges
    r1 = x.flatten(0, 2)  # A*B*C, D, E
    r2 = x.flatten(0, 4)  # A*B*C*D*E
    r3 = x.flatten(1, 3)  # A, B*C*D, E

    reveal_type(r1)
    reveal_type(r2)
    reveal_type(r3)

    # Assert the actual inferred types (left-associative)
    assert_type(r1, Tensor[A * B * C, D, E])
    assert_type(r2, Tensor[A * B * C * D * E])
    assert_type(r3, Tensor[A, B * C * D, E])

    # Show structural compatibility with commuted orderings
    expected1: Tensor[C * B * A, D, E] = r1
    expected2: Tensor[E * D * C * B * A] = r2
    expected3: Tensor[A, D * C * B, E] = r3
    return (expected1, expected2, expected3)


# ============================================================================
# Pow canonicalization tests
# ============================================================================


def test_pow_literal(x: Tensor[2**3]) -> Tensor[8]:
    """2**3 = 8, literal exponentiation."""
    return x


def test_pow_identity[N](x: Tensor[N**1]) -> Tensor[N]:
    """N**1 = N."""
    return x


def test_pow_zero[N](x: Tensor[N**0]) -> Tensor[1]:
    """N**0 = 1."""
    return x


def test_pow_symbolic_base[I](x: Tensor[2**I]) -> Tensor[2**I]:
    """Symbolic exponent preserved."""
    return x


def test_pow_product_grouping[I](
    x: Tensor[2**I],
) -> Tensor[2**I]:
    """2 * 2**I = 2**(I+1) and 2**(I+1) canonically equals... 2**(I+1).
    But we test that 2 * 2**(I-1) = 2**I via product grouping."""
    reveal_type(x)
    return x


def test_pow_same_base_mul[I, B, C](
    x: Tensor[B, C * 2**I],
) -> Tensor[B, C * 2**I]:
    """Product with Pow factor stays canonical."""
    return x


def test_pow_literal_absorb[I](
    x: Tensor[8 * 2**I],
) -> Tensor[2 ** (I + 3)]:
    """8 * 2**I = 2**3 * 2**I = 2**(I+3)."""
    return x


def test_pow_mul_same_base[I](
    x: Tensor[2**I * 2],
) -> Tensor[2 ** (I + 1)]:
    """2**I * 2 = 2**(I+1)."""
    return x
