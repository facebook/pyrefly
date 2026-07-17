# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test type variable unification in Int expressions"""

from typing import assert_type, TYPE_CHECKING

from shape_extensions import IntVar

if TYPE_CHECKING:
    from shape_extensions import Int


# Test 1: Top-level type var unification
# When passing Int[A * B] to a function expecting Int[X],
# X should be unified with A * B
def identity_int[X: IntVar](x: Int[X]) -> Int[X]:
    return x


def test_top_level_unification[A: IntVar, B: IntVar](a: Int[A], b: Int[B]):
    expr = a * b  # Int[A * B]
    assert_type(expr, Int[A * B])
    result = identity_int(expr)
    assert_type(result, Int[A * B])
    # X should be unified with A * B, so result should be Int[A * B]
    assert_type(result, Int[A * B])


# Test 2: Nested type var without prior binding
# When passing Int[(A * B) // 2] to a function expecting Int[X // 2],
# X cannot be inferred from a nested position - this is an error
def half_int[X: IntVar](x: Int[X // 2]) -> Int[X]:
    return x * 2  # type: ignore


def test_nested_unification_fails[A: IntVar, B: IntVar](a: Int[A], b: Int[B]):
    expr = (a * b) // 2  # Int[(A * B) // 2]
    # X is in a nested position and cannot be inferred.
    # E: Type variable cannot be inferred from a nested position
    half_int(expr)


# Test 3: Nested type var with prior binding
# If X is bound from the first argument, then the second argument can use X in a nested position
def two_args[X: IntVar](first: Int[X], second: Int[X // 2]) -> Int[X]:
    return first


def test_nested_with_prior_binding[N: IntVar](n: Int[N]):
    half_n = n // 2  # Int[N // 2]
    # First arg binds X = N, second arg checks N // 2 = N // 2.
    result = two_args(n, half_n)
    assert_type(result, Int[N])


# Test 4: Nested type var with prior binding - complex expression
# X is bound to A + A from first arg, second arg uses X // 2 = (A + A) // 2 = A
def test_nested_with_simplification[A: IntVar](a: Int[A]):
    double_a = a + a  # Int[A + A]
    # X = A + A from first arg
    # Second arg: X // 2 = (A + A) // 2 = A (after simplification)
    result = two_args(double_a, a)
    assert_type(result, Int[A + A])
