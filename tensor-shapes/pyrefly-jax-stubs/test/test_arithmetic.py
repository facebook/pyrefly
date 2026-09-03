# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import jax.numpy as jnp
from shape_extensions import assert_shape, Scalar


def test_elementwise_operators_preserve_shape() -> None:
    a = jnp.ones((3, 4))
    b = jnp.full((3, 4), 2.0)

    assert_shape(a + b, (3, 4))
    assert_shape(b - a, (3, 4))
    assert_shape(a * b, (3, 4))
    assert_shape(b / a, (3, 4))
    assert_shape(b**2, (3, 4))


def test_scalar_operands_preserve_shape() -> None:
    a = jnp.ones((3, 4))
    v = jnp.full(5, 2.0)

    assert_shape(a + 1.0, (3, 4))
    assert_shape(1.0 + a, (3, 4))
    assert_shape(a - 1.0, (3, 4))
    assert_shape(1.0 - a, (3, 4))
    assert_shape(a * 2.0, (3, 4))
    assert_shape(2.0 * a, (3, 4))
    assert_shape(a / 2.0, (3, 4))
    assert_shape(v**2, (5,))


def test_complex_scalars_preserve_shape() -> None:
    a = jnp.ones((3, 4))

    assert_shape(a + 1j, (3, 4))
    assert_shape(1j + a, (3, 4))
    assert_shape(a * 2j, (3, 4))
    assert_shape(jnp.add(a, 1j), (3, 4))
    assert_shape(jnp.multiply(2j, a), (3, 4))


def test_comparisons_produce_boolean_arrays() -> None:
    a = jnp.ones((3, 4))

    # These are elementwise: the result is an array, not a `bool`.
    assert_shape(a == a, (3, 4))
    assert_shape(a != jnp.ones((1, 4)), (3, 4))
    assert_shape(a > 0, (3, 4))
    assert_shape(a <= 1.0, (3, 4))
    assert_shape(a >= jnp.ones((1, 4)), (3, 4))
    assert_shape(a < jnp.ones((3, 1)), (3, 4))


def test_unary_operators_preserve_shape() -> None:
    a = jnp.full((3, 4), -1.0)

    assert_shape(-a, (3, 4))
    assert_shape(+a, (3, 4))
    assert_shape(abs(a), (3, 4))
    assert_shape(jnp.abs(a), (3, 4))
    assert_shape(jnp.negative(a), (3, 4))


def test_broadcasting_expands_singleton_dimensions() -> None:
    column = jnp.ones((3, 1))
    row = jnp.ones((1, 4))

    assert_shape(column + row, (3, 4))
    assert_shape(row * column, (3, 4))
    assert_shape(jnp.add(column, row), (3, 4))
    assert_shape(jnp.multiply(row, column), (3, 4))
    assert_shape(jnp.maximum(column, row), (3, 4))
    assert_shape(jnp.minimum(column, row), (3, 4))


def test_binary_functions_accept_scalars() -> None:
    a = jnp.ones((3, 4))

    assert_shape(jnp.add(a, 1), (3, 4))
    assert_shape(jnp.add(1, a), (3, 4))
    assert_shape(jnp.subtract(a, 1.0), (3, 4))
    assert_shape(jnp.multiply(2.0, a), (3, 4))
    assert_shape(jnp.divide(a, 2.0), (3, 4))
    assert_shape(jnp.maximum(a, 0), (3, 4))
    assert_shape(jnp.minimum(0, a), (3, 4))


def test_elementwise_unary_functions_preserve_shape() -> None:
    a = jnp.full((2, 3), 0.5)

    assert_shape(jnp.abs(a), (2, 3))
    assert_shape(jnp.absolute(a), (2, 3))
    assert_shape(jnp.acos(a), (2, 3))
    assert_shape(jnp.acosh(a + 1.0), (2, 3))
    assert_shape(jnp.angle(a), (2, 3))
    assert_shape(jnp.arccos(a), (2, 3))
    assert_shape(jnp.arccosh(a + 1.0), (2, 3))
    assert_shape(jnp.arcsin(a), (2, 3))
    assert_shape(jnp.arcsinh(a), (2, 3))
    assert_shape(jnp.arctan(a), (2, 3))
    assert_shape(jnp.arctanh(a), (2, 3))
    assert_shape(jnp.around(a), (2, 3))
    assert_shape(jnp.asin(a), (2, 3))
    assert_shape(jnp.asinh(a), (2, 3))
    assert_shape(jnp.atan(a), (2, 3))
    assert_shape(jnp.atanh(a), (2, 3))
    assert_shape(jnp.cbrt(a), (2, 3))
    assert_shape(jnp.ceil(a), (2, 3))
    assert_shape(jnp.conj(a), (2, 3))
    assert_shape(jnp.conjugate(a), (2, 3))
    assert_shape(jnp.cos(a), (2, 3))
    assert_shape(jnp.cosh(a), (2, 3))
    assert_shape(jnp.deg2rad(a), (2, 3))
    assert_shape(jnp.degrees(a), (2, 3))
    assert_shape(jnp.exp(a), (2, 3))
    assert_shape(jnp.exp2(a), (2, 3))
    assert_shape(jnp.expm1(a), (2, 3))
    assert_shape(jnp.fabs(a), (2, 3))
    assert_shape(jnp.floor(a), (2, 3))
    assert_shape(jnp.i0(a), (2, 3))
    assert_shape(jnp.imag(a), (2, 3))
    assert_shape(jnp.log(a), (2, 3))
    assert_shape(jnp.log10(a), (2, 3))
    assert_shape(jnp.log1p(a), (2, 3))
    assert_shape(jnp.log2(a), (2, 3))
    assert_shape(jnp.negative(a), (2, 3))
    assert_shape(jnp.positive(a), (2, 3))
    assert_shape(jnp.rad2deg(a), (2, 3))
    assert_shape(jnp.radians(a), (2, 3))
    assert_shape(jnp.real(a), (2, 3))
    assert_shape(jnp.reciprocal(a), (2, 3))
    assert_shape(jnp.rint(a), (2, 3))
    assert_shape(jnp.round(a), (2, 3))
    assert_shape(jnp.sign(a), (2, 3))
    assert_shape(jnp.signbit(a), (2, 3))
    assert_shape(jnp.sin(a), (2, 3))
    assert_shape(jnp.sinc(a), (2, 3))
    assert_shape(jnp.sinh(a), (2, 3))
    assert_shape(jnp.spacing(a), (2, 3))
    assert_shape(jnp.sqrt(a), (2, 3))
    assert_shape(jnp.square(a), (2, 3))
    assert_shape(jnp.tan(a), (2, 3))
    assert_shape(jnp.tanh(a), (2, 3))
    assert_shape(jnp.trunc(a), (2, 3))
    assert_shape(jnp.unwrap(a), (2, 3))


def test_tuple_returning_functions_preserve_shape() -> None:
    a = jnp.full((2, 3), 1.5)
    b = jnp.full((1, 3), 0.5)

    f1, f2 = jnp.frexp(a)
    assert_shape(f1, (2, 3))
    assert_shape(f2, (2, 3))

    m1, m2 = jnp.modf(a)
    assert_shape(m1, (2, 3))
    assert_shape(m2, (2, 3))

    d1, d2 = jnp.divmod(a, b)
    assert_shape(d1, (2, 3))
    assert_shape(d2, (2, 3))


def test_integer_elementwise_functions() -> None:
    a = jnp.ones((2, 3), dtype=jnp.int32)
    b = jnp.full((1, 3), 2, dtype=jnp.int32)

    assert_shape(jnp.bitwise_count(a), (2, 3))
    assert_shape(jnp.bitwise_invert(a), (2, 3))
    assert_shape(jnp.bitwise_not(a), (2, 3))
    assert_shape(jnp.invert(a), (2, 3))

    assert_shape(jnp.bitwise_and(a, b), (2, 3))
    assert_shape(jnp.bitwise_or(a, b), (2, 3))
    assert_shape(jnp.bitwise_xor(a, b), (2, 3))
    assert_shape(jnp.bitwise_left_shift(a, b), (2, 3))
    assert_shape(jnp.bitwise_right_shift(a, b), (2, 3))
    assert_shape(jnp.left_shift(a, b), (2, 3))
    assert_shape(jnp.right_shift(a, b), (2, 3))
    assert_shape(jnp.gcd(a, b), (2, 3))
    assert_shape(jnp.lcm(a, b), (2, 3))


def test_additional_binary_broadcasting_functions() -> None:
    col = jnp.ones((3, 1))
    row = jnp.ones((1, 4))

    assert_shape(jnp.arctan2(col, row), (3, 4))
    assert_shape(jnp.atan2(col, row), (3, 4))
    assert_shape(jnp.copysign(col, row), (3, 4))
    assert_shape(jnp.float_power(col, row), (3, 4))
    assert_shape(jnp.floor_divide(col, row), (3, 4))
    assert_shape(jnp.fmod(col, row), (3, 4))
    assert_shape(jnp.heaviside(col, row), (3, 4))
    assert_shape(jnp.hypot(col, row), (3, 4))
    assert_shape(jnp.logaddexp(col, row), (3, 4))
    assert_shape(jnp.logaddexp2(col, row), (3, 4))
    assert_shape(jnp.mod(col, row), (3, 4))
    assert_shape(jnp.nextafter(col, row), (3, 4))
    assert_shape(jnp.pow(col, row), (3, 4))
    assert_shape(jnp.power(col, row), (3, 4))
    assert_shape(jnp.remainder(col, row), (3, 4))
    assert_shape(jnp.true_divide(col, row), (3, 4))


def test_subtraction_rejects_incompatible_broadcast() -> None:
    a = jnp.ones((3, 4))
    b = jnp.ones(5)

    assert_shape(a - jnp.ones((3, 4)), (3, 4))
    try:
        # E: Cannot evaluate type-level shape DSL call
        a - b
    except (TypeError, ValueError):
        pass
    else:
        raise AssertionError("expected JAX to reject incompatible shapes")


def test_binary_functions_reject_incompatible_broadcast() -> None:
    a = jnp.ones((3, 4))
    b = jnp.ones(5)

    assert_shape(jnp.maximum(a, jnp.ones((3, 4))), (3, 4))
    try:
        jnp.maximum(  # E: Cannot evaluate type-level shape DSL call
            a, b
        )
    except (TypeError, ValueError):
        pass
    else:
        raise AssertionError("expected JAX to reject incompatible shapes")


def test_add_arraylike_annotations() -> None:
    # Scalar / scalar operations produce 0-D arrays.
    assert_shape(jnp.add(1, 2), ())
    assert_shape(jnp.add(1.5, 2.5), ())
    assert_shape(jnp.add(1, 2.0), ())
    assert_shape(jnp.add(True, 1), ())
    assert_shape(jnp.add(True, False), ())
    assert_shape(jnp.add(1j, 2.0), ())
    assert_shape(jnp.add(1j, 2j), ())

    # Scalar / array operations preserve or broadcast shape.
    scalar_0d = jnp.ones(())
    assert_shape(jnp.add(1, scalar_0d), ())
    assert_shape(jnp.add(scalar_0d, 1), ())
    assert_shape(jnp.add(scalar_0d, scalar_0d), ())

    vector = jnp.ones(4)
    assert_shape(jnp.add(vector, 1), (4,))
    assert_shape(jnp.add(1, vector), (4,))
    assert_shape(jnp.add(vector, 2.5), (4,))
    assert_shape(jnp.add(2.5, vector), (4,))

    matrix = jnp.ones((3, 4))
    assert_shape(jnp.add(matrix, 1), (3, 4))
    assert_shape(jnp.add(1, matrix), (3, 4))
    assert_shape(jnp.add(matrix, 2.5), (3, 4))
    assert_shape(jnp.add(2.5, matrix), (3, 4))
    assert_shape(jnp.add(matrix, 1j), (3, 4))
    assert_shape(jnp.add(1j, matrix), (3, 4))
    assert_shape(jnp.add(matrix, True), (3, 4))
    assert_shape(jnp.add(True, matrix), (3, 4))

    tensor = jnp.ones((2, 3, 4))
    assert_shape(jnp.add(tensor, 1), (2, 3, 4))
    assert_shape(jnp.add(1, tensor), (2, 3, 4))
    assert_shape(jnp.add(tensor, 1.0), (2, 3, 4))
    assert_shape(jnp.add(1.0, tensor), (2, 3, 4))

    # Scalar type only allows empty shapes:
    _: Scalar[()] = 1
    _bare: Scalar = 1
    # E: `shape_extensions.Scalar` only accepts an empty shape `[]`
    _bad: Scalar[[3, 4]]
