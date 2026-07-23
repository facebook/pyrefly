# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Comprehensive tests for Int type: parsing, arithmetic, and subtyping.

This file tests the Int/Int type system independent of Tensor shapes.
"""

from typing import Any, assert_type, Literal, TYPE_CHECKING

from shape_extensions import IntVar


if TYPE_CHECKING:
    from shape_extensions import Int


# ============================================================================
# Basic Int Parsing
# ============================================================================


def test_dim_literal_parsing() -> None:
    """Int[n] with literal integers"""
    x: Int[3] = 3
    assert_type(x, Int[3])

    y: Int[42] = 42
    assert_type(y, Int[42])


def test_dim_typevar_parsing[N: IntVar](n: Int[N]) -> None:
    """Int[N] with type variable"""
    assert_type(n, Int[N])


def test_dim_expression_parsing[N: IntVar](n: Int[N]) -> None:
    """Int[N+1] with expression"""
    x = n + 1
    assert_type(x, Int[N + 1])


# ============================================================================
# Int Arithmetic - All Operators
# ============================================================================


def test_dim_add_literal[N: IntVar](n: Int[N]) -> None:
    """Addition: Int + literal"""
    result = n + 2
    assert_type(result, Int[N + 2])


def test_dim_add_dim[A: IntVar, B: IntVar](a: Int[A], b: Int[B]) -> None:
    """Addition: Int + Int"""
    result = a + b
    assert_type(result, Int[A + B])


def test_dim_sub_literal[N: IntVar](n: Int[N]) -> None:
    """Subtraction: Int - literal"""
    result = n - 1
    assert_type(result, Int[N - 1])


def test_dim_sub_dim[A: IntVar, B: IntVar](a: Int[A], b: Int[B]) -> None:
    """Subtraction: Int - Int"""
    result = a - b
    assert_type(result, Int[A - B])


def test_dim_mul_literal[N: IntVar](n: Int[N]) -> None:
    """Multiplication: Int * literal"""
    result = n * 2
    assert_type(result, Int[N * 2])


def test_dim_mul_dim[A: IntVar, B: IntVar](a: Int[A], b: Int[B]) -> None:
    """Multiplication: Int * Int"""
    result = a * b
    assert_type(result, Int[A * B])


def test_dim_floordiv_literal[N: IntVar](n: Int[N]) -> None:
    """Floor division: Int // literal"""
    result = n // 2
    assert_type(result, Int[N // 2])


def test_dim_floordiv_dim[A: IntVar, B: IntVar](a: Int[A], b: Int[B]) -> None:
    """Floor division: Int // Int"""
    result = a // b
    assert_type(result, Int[A // B])


def test_dim_complex_expression[N: IntVar](n: Int[N]) -> None:
    """Complex arithmetic expression"""
    # (N + N) // 2 expression - simplification happens during subtyping
    double = n + n
    half_double = double // 2
    assert_type(half_double, Int[(N + N) // 2])


def test_dim_nested_expression[A: IntVar, B: IntVar](a: Int[A], b: Int[B]) -> None:
    """Nested arithmetic"""
    result = (a + b) * 2
    assert_type(result, Int[(A + B) * 2])


# ============================================================================
# Int Subtyping - Using def f(x: T1) -> T2: return x pattern
# ============================================================================


def dim_to_dim_same[N: IntVar](x: Int[N]) -> Int[N]:
    """Int[N] <: Int[N]"""
    return x


def dim_literal_to_same(x: Int[3]) -> Int[3]:
    """Int[3] <: Int[3]"""
    return x


def dim_to_int[N: IntVar](x: Int[N]) -> int:
    """Int[N] <: int - Int values are subtypes of int"""
    return x


def dim_literal_to_int(x: Int[5]) -> int:
    """Int[5] <: int"""
    return x


def int_to_dim_any(x: int) -> Int[Any]:
    """int <: Int[Any] - plain int can be used where Int expected"""
    return x


def literal_to_dim(x: Literal[7]) -> Int[7]:
    """Literal[7] <: Int[7] - literal ints are subtypes of Int"""
    return x


def dim_expression_subtype[N: IntVar](x: Int[N + N]) -> Int[N * 2]:
    """Int[N + N] <: Int[N * 2] - after simplification these are equal"""
    return x


def dim_double_half[N: IntVar](x: Int[(N + N) // 2]) -> Int[N]:
    """Int[(N + N) // 2] <: Int[N] - simplifies to N"""
    return x


# ============================================================================
# Int Type Variable Binding
# ============================================================================


def identity_dim[X: IntVar](x: Int[X]) -> Int[X]:
    """Identity function for Int - X binds to the dimension"""
    return x


def test_dim_binding_literal() -> None:
    """Binding type var to literal"""
    result = identity_dim(4)
    assert_type(result, Int[4])


def test_dim_binding_typevar[N: IntVar](n: Int[N]) -> None:
    """Binding type var to another type var"""
    result = identity_dim(n)
    assert_type(result, Int[N])


def test_dim_binding_expression[A: IntVar, B: IntVar](a: Int[A], b: Int[B]) -> None:
    """Binding type var to expression"""
    expr = a * b
    result = identity_dim(expr)
    assert_type(result, Int[A * B])


# ============================================================================
# Int Unification - Type Var in Result Position
# ============================================================================


def double_dim[X: IntVar](x: Int[X]) -> Int[X * 2]:
    """Return doubled dimension"""
    return x * 2


def test_double_dim_literal() -> None:
    """Double a literal dimension"""
    result = double_dim(5)
    assert_type(result, Int[10])


def test_double_dim_typevar[N: IntVar](n: Int[N]) -> None:
    """Double a symbolic dimension"""
    result = double_dim(n)
    assert_type(result, Int[N * 2])


def half_dim[X: IntVar](x: Int[X]) -> Int[X // 2]:
    """Return halved dimension"""
    return x // 2


def test_half_dim_literal() -> None:
    """Half a literal dimension"""
    result = half_dim(10)
    assert_type(result, Int[5])


def test_half_dim_typevar[N: IntVar](n: Int[N]) -> None:
    """Half a symbolic dimension"""
    result = half_dim(n)
    assert_type(result, Int[N // 2])


# ============================================================================
# Multi-Argument Int Functions
# ============================================================================


def add_dims[A: IntVar, B: IntVar](a: Int[A], b: Int[B]) -> Int[A + B]:
    """Add two dimensions"""
    return a + b


def test_add_dims_literals() -> None:
    """Add literal dimensions"""
    result = add_dims(3, 4)
    assert_type(result, Int[7])


def test_add_dims_typevars[X: IntVar, Y: IntVar](x: Int[X], y: Int[Y]) -> None:
    """Add symbolic dimensions"""
    result = add_dims(x, y)
    assert_type(result, Int[X + Y])


def test_add_dims_mixed[N: IntVar](n: Int[N]) -> None:
    """Add symbolic and literal"""
    result = add_dims(n, 5)
    assert_type(result, Int[N + 5])


# ============================================================================
# Int with Prior Binding
# ============================================================================


def two_dims_same_var[X: IntVar](first: Int[X], second: Int[X]) -> Int[X]:
    """Both arguments must have same dimension"""
    return first


def test_two_dims_same_literal() -> None:
    """Same literal for both"""
    result = two_dims_same_var(5, 5)
    assert_type(result, Int[5])


def test_two_dims_same_typevar[N: IntVar](n: Int[N]) -> None:
    """Same typevar for both"""
    result = two_dims_same_var(n, n)
    assert_type(result, Int[N])


def with_derived[X: IntVar](first: Int[X], second: Int[X // 2]) -> Int[X]:
    """Second arg uses derived dimension"""
    return first


def test_derived_binding[N: IntVar](n: Int[N]) -> None:
    """Bind X from first, check X // 2 in second"""
    half = n // 2
    result = with_derived(n, half)
    assert_type(result, Int[N])


def test_derived_with_simplification[A: IntVar](a: Int[A]) -> None:
    """Bind X = A + A, check X // 2 = A"""
    double_a = a + a  # Int[A + A]
    # X = A + A, X // 2 = (A + A) // 2 = A
    result = with_derived(double_a, a)
    assert_type(result, Int[A + A])
