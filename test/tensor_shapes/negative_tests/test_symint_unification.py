# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test type variable unification in Dim expressions"""

from typing import assert_type, reveal_type, TYPE_CHECKING

if TYPE_CHECKING:
    from torch_shapes import Dim


# Test 1: Top-level type var unification
# When passing Dim[A * B] to a function expecting Dim[X],
# X should be unified with A * B
def identity_symint[X](x: Dim[X]) -> Dim[X]:
    return x


def test_top_level_unification[A, B](a: Dim[A], b: Dim[B]):
    expr = a * b  # Dim[A * B]
    reveal_type(expr)  # Should be Dim[A * B]
    result = identity_symint(expr)
    reveal_type(result)  # Should be Dim[A * B] if X is unified
    # X should be unified with A * B, so result should be Dim[A * B]
    assert_type(result, Dim[A * B])


# Test 2: Nested type var - SHOULD FAIL (X not bound)
# When passing Dim[(A * B) // 2] to a function expecting Dim[X // 2],
# X cannot be inferred from a nested position - this is an error
def half_symint[X](x: Dim[X // 2]) -> Dim[X]:
    return x * 2  # type: ignore


def test_nested_unification_fails[A, B](a: Dim[A], b: Dim[B]):
    expr = (a * b) // 2  # Dim[(A * B) // 2]
    # This should fail - X is in a nested position and cannot be inferred
    half_symint(expr)


# Test 3: Nested type var with prior binding - SHOULD PASS
# If X is bound from the first argument, then the second argument can use X in a nested position
def two_args[X](first: Dim[X], second: Dim[X // 2]) -> Dim[X]:
    return first


def test_nested_with_prior_binding[N](n: Dim[N]):
    half_n = n // 2  # Dim[N // 2]
    # First arg binds X = N, second arg checks N // 2 = N // 2 (should pass)
    result = two_args(n, half_n)
    reveal_type(result)  # Should be Dim[N]
    assert_type(result, Dim[N])


# Test 4: Nested type var with prior binding - complex expression
# X is bound to A + A from first arg, second arg uses X // 2 = (A + A) // 2 = A
def test_nested_with_simplification[A](a: Dim[A]):
    double_a = a + a  # Dim[A + A]
    # X = A + A from first arg
    # Second arg: X // 2 = (A + A) // 2 = A (after simplification)
    result = two_args(double_a, a)
    reveal_type(result)  # Should be Dim[A + A]
    assert_type(result, Dim[A + A])
