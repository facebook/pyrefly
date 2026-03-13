# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Comprehensive tests for Dim type: parsing, arithmetic, and subtyping.

This file tests the Size/Dim type system independent of Tensor shapes.
"""

from typing import Any, assert_type, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from torch_shapes import Dim


# ============================================================================
# Basic Dim Parsing
# ============================================================================


def test_dim_literal_parsing() -> None:
    """Dim[n] with literal integers"""
    x: Dim[3] = 3
    assert_type(x, Dim[3])

    y: Dim[42] = 42
    assert_type(y, Dim[42])


def test_dim_typevar_parsing[N](n: Dim[N]) -> None:
    """Dim[N] with type variable"""
    assert_type(n, Dim[N])


def test_dim_expression_parsing[N](n: Dim[N]) -> None:
    """Dim[N+1] with expression"""
    x = n + 1
    assert_type(x, Dim[N + 1])


# ============================================================================
# Dim Arithmetic - All Operators
# ============================================================================


def test_dim_add_literal[N](n: Dim[N]) -> None:
    """Addition: Dim + literal"""
    result = n + 2
    assert_type(result, Dim[N + 2])


def test_dim_add_dim[A, B](a: Dim[A], b: Dim[B]) -> None:
    """Addition: Dim + Dim"""
    result = a + b
    assert_type(result, Dim[A + B])


def test_dim_sub_literal[N](n: Dim[N]) -> None:
    """Subtraction: Dim - literal"""
    result = n - 1
    assert_type(result, Dim[N - 1])


def test_dim_sub_dim[A, B](a: Dim[A], b: Dim[B]) -> None:
    """Subtraction: Dim - Dim"""
    result = a - b
    assert_type(result, Dim[A - B])


def test_dim_mul_literal[N](n: Dim[N]) -> None:
    """Multiplication: Dim * literal"""
    result = n * 2
    assert_type(result, Dim[N * 2])


def test_dim_mul_dim[A, B](a: Dim[A], b: Dim[B]) -> None:
    """Multiplication: Dim * Dim"""
    result = a * b
    assert_type(result, Dim[A * B])


def test_dim_floordiv_literal[N](n: Dim[N]) -> None:
    """Floor division: Dim // literal"""
    result = n // 2
    assert_type(result, Dim[N // 2])


def test_dim_floordiv_dim[A, B](a: Dim[A], b: Dim[B]) -> None:
    """Floor division: Dim // Dim"""
    result = a // b
    assert_type(result, Dim[A // B])


def test_dim_complex_expression[N](n: Dim[N]) -> None:
    """Complex arithmetic expression"""
    # (N + N) // 2 expression - simplification happens during subtyping
    double = n + n
    half_double = double // 2
    assert_type(half_double, Dim[(N + N) // 2])


def test_dim_nested_expression[A, B](a: Dim[A], b: Dim[B]) -> None:
    """Nested arithmetic"""
    result = (a + b) * 2
    assert_type(result, Dim[(A + B) * 2])


# ============================================================================
# Dim Subtyping - Using def f(x: T1) -> T2: return x pattern
# ============================================================================


def dim_to_dim_same[N](x: Dim[N]) -> Dim[N]:
    """Dim[N] <: Dim[N]"""
    return x


def dim_literal_to_same(x: Dim[3]) -> Dim[3]:
    """Dim[3] <: Dim[3]"""
    return x


def dim_to_int[N](x: Dim[N]) -> int:
    """Dim[N] <: int - Dim values are subtypes of int"""
    return x


def dim_literal_to_int(x: Dim[5]) -> int:
    """Dim[5] <: int"""
    return x


def int_to_dim_any(x: int) -> Dim[Any]:
    """int <: Dim[Any] - plain int can be used where Dim expected"""
    return x


def literal_to_dim(x: Literal[7]) -> Dim[7]:
    """Literal[7] <: Dim[7] - literal ints are subtypes of Dim"""
    return x


def dim_expression_subtype[N](x: Dim[N + N]) -> Dim[N * 2]:
    """Dim[N + N] <: Dim[N * 2] - after simplification these are equal"""
    return x


def dim_double_half[N](x: Dim[(N + N) // 2]) -> Dim[N]:
    """Dim[(N + N) // 2] <: Dim[N] - simplifies to N"""
    return x


# ============================================================================
# Dim Type Variable Binding
# ============================================================================


def identity_dim[X](x: Dim[X]) -> Dim[X]:
    """Identity function for Dim - X binds to the dimension"""
    return x


def test_dim_binding_literal() -> None:
    """Binding type var to literal"""
    result = identity_dim(4)
    assert_type(result, Dim[4])


def test_dim_binding_typevar[N](n: Dim[N]) -> None:
    """Binding type var to another type var"""
    result = identity_dim(n)
    assert_type(result, Dim[N])


def test_dim_binding_expression[A, B](a: Dim[A], b: Dim[B]) -> None:
    """Binding type var to expression"""
    expr = a * b
    result = identity_dim(expr)
    assert_type(result, Dim[A * B])


# ============================================================================
# Dim Unification - Type Var in Result Position
# ============================================================================


def double_dim[X](x: Dim[X]) -> Dim[X * 2]:
    """Return doubled dimension"""
    return x * 2


def test_double_dim_literal() -> None:
    """Double a literal dimension"""
    result = double_dim(5)
    assert_type(result, Dim[10])


def test_double_dim_typevar[N](n: Dim[N]) -> None:
    """Double a symbolic dimension"""
    result = double_dim(n)
    assert_type(result, Dim[N * 2])


def half_dim[X](x: Dim[X]) -> Dim[X // 2]:
    """Return halved dimension"""
    return x // 2


def test_half_dim_literal() -> None:
    """Half a literal dimension"""
    result = half_dim(10)
    assert_type(result, Dim[5])


def test_half_dim_typevar[N](n: Dim[N]) -> None:
    """Half a symbolic dimension"""
    result = half_dim(n)
    assert_type(result, Dim[N // 2])


# ============================================================================
# Multi-Argument Dim Functions
# ============================================================================


def add_dims[A, B](a: Dim[A], b: Dim[B]) -> Dim[A + B]:
    """Add two dimensions"""
    return a + b


def test_add_dims_literals() -> None:
    """Add literal dimensions"""
    result = add_dims(3, 4)
    assert_type(result, Dim[7])


def test_add_dims_typevars[X, Y](x: Dim[X], y: Dim[Y]) -> None:
    """Add symbolic dimensions"""
    result = add_dims(x, y)
    assert_type(result, Dim[X + Y])


def test_add_dims_mixed[N](n: Dim[N]) -> None:
    """Add symbolic and literal"""
    result = add_dims(n, 5)
    assert_type(result, Dim[N + 5])


# ============================================================================
# Dim with Prior Binding
# ============================================================================


def two_dims_same_var[X](first: Dim[X], second: Dim[X]) -> Dim[X]:
    """Both arguments must have same dimension"""
    return first


def test_two_dims_same_literal() -> None:
    """Same literal for both"""
    result = two_dims_same_var(5, 5)
    assert_type(result, Dim[5])


def test_two_dims_same_typevar[N](n: Dim[N]) -> None:
    """Same typevar for both"""
    result = two_dims_same_var(n, n)
    assert_type(result, Dim[N])


def with_derived[X](first: Dim[X], second: Dim[X // 2]) -> Dim[X]:
    """Second arg uses derived dimension"""
    return first


def test_derived_binding[N](n: Dim[N]) -> None:
    """Bind X from first, check X // 2 in second"""
    half = n // 2
    result = with_derived(n, half)
    assert_type(result, Dim[N])


def test_derived_with_simplification[A](a: Dim[A]) -> None:
    """Bind X = A + A, check X // 2 = A"""
    double_a = a + a  # Dim[A + A]
    # X = A + A, X // 2 = (A + A) // 2 = A
    result = with_derived(double_a, a)
    assert_type(result, Dim[A + A])
