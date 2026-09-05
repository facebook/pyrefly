# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import jax.numpy as jnp
from shape_extensions import assert_shape


def test_elementwise_operators_preserve_shape() -> None:
    a = jnp.ones((3, 4))
    b = jnp.full((3, 4), 2.0)

    assert_shape((a + b).shape, (3, 4))
    assert_shape((b - a).shape, (3, 4))
    assert_shape((a * b).shape, (3, 4))
    assert_shape((b / a).shape, (3, 4))
    assert_shape((b**2).shape, (3, 4))


def test_scalar_operands_preserve_shape() -> None:
    a = jnp.ones((3, 4))
    v = jnp.full(5, 2.0)

    assert_shape((a + 1.0).shape, (3, 4))
    assert_shape((1.0 + a).shape, (3, 4))
    assert_shape((a - 1.0).shape, (3, 4))
    assert_shape((1.0 - a).shape, (3, 4))
    assert_shape((a * 2.0).shape, (3, 4))
    assert_shape((2.0 * a).shape, (3, 4))
    assert_shape((a / 2.0).shape, (3, 4))
    assert_shape((v**2).shape, (5,))


def test_complex_scalars_preserve_shape() -> None:
    a = jnp.ones((3, 4))

    assert_shape((a + 1j).shape, (3, 4))
    assert_shape((1j + a).shape, (3, 4))
    assert_shape((a * 2j).shape, (3, 4))
    assert_shape(jnp.add(a, 1j).shape, (3, 4))
    assert_shape(jnp.multiply(2j, a).shape, (3, 4))


def test_comparisons_produce_boolean_arrays() -> None:
    a = jnp.ones((3, 4))

    # These are elementwise: the result is an array, not a `bool`.
    assert_shape((a == a).shape, (3, 4))
    assert_shape((a != jnp.ones((1, 4))).shape, (3, 4))
    assert_shape((a > 0).shape, (3, 4))
    assert_shape((a <= 1.0).shape, (3, 4))
    assert_shape((a >= jnp.ones((1, 4))).shape, (3, 4))
    assert_shape((a < jnp.ones((3, 1))).shape, (3, 4))


def test_unary_operators_preserve_shape() -> None:
    a = jnp.full((3, 4), -1.0)

    assert_shape((-a).shape, (3, 4))
    assert_shape((+a).shape, (3, 4))
    assert_shape(abs(a).shape, (3, 4))
    assert_shape(jnp.abs(a).shape, (3, 4))
    assert_shape(jnp.negative(a).shape, (3, 4))


def test_broadcasting_expands_singleton_dimensions() -> None:
    column = jnp.ones((3, 1))
    row = jnp.ones((1, 4))

    assert_shape((column + row).shape, (3, 4))
    assert_shape((row * column).shape, (3, 4))
    assert_shape(jnp.add(column, row).shape, (3, 4))
    assert_shape(jnp.multiply(row, column).shape, (3, 4))
    assert_shape(jnp.maximum(column, row).shape, (3, 4))
    assert_shape(jnp.minimum(column, row).shape, (3, 4))


def test_binary_functions_accept_scalars() -> None:
    a = jnp.ones((3, 4))

    assert_shape(jnp.add(a, 1).shape, (3, 4))
    assert_shape(jnp.add(1, a).shape, (3, 4))
    assert_shape(jnp.subtract(a, 1.0).shape, (3, 4))
    assert_shape(jnp.multiply(2.0, a).shape, (3, 4))
    assert_shape(jnp.divide(a, 2.0).shape, (3, 4))
    assert_shape(jnp.maximum(a, 0).shape, (3, 4))
    assert_shape(jnp.minimum(0, a).shape, (3, 4))


def test_elementwise_unary_functions_preserve_shape() -> None:
    a = jnp.full((2, 3), 0.5)

    assert_shape(jnp.abs(a).shape, (2, 3))
    assert_shape(jnp.absolute(a).shape, (2, 3))
    assert_shape(jnp.acos(a).shape, (2, 3))
    assert_shape(jnp.acosh(a + 1.0).shape, (2, 3))
    assert_shape(jnp.angle(a).shape, (2, 3))
    assert_shape(jnp.arccos(a).shape, (2, 3))
    assert_shape(jnp.arccosh(a + 1.0).shape, (2, 3))
    assert_shape(jnp.arcsin(a).shape, (2, 3))
    assert_shape(jnp.arcsinh(a).shape, (2, 3))
    assert_shape(jnp.arctan(a).shape, (2, 3))
    assert_shape(jnp.arctanh(a).shape, (2, 3))
    assert_shape(jnp.around(a).shape, (2, 3))
    assert_shape(jnp.asin(a).shape, (2, 3))
    assert_shape(jnp.asinh(a).shape, (2, 3))
    assert_shape(jnp.atan(a).shape, (2, 3))
    assert_shape(jnp.atanh(a).shape, (2, 3))
    assert_shape(jnp.cbrt(a).shape, (2, 3))
    assert_shape(jnp.ceil(a).shape, (2, 3))
    assert_shape(jnp.conj(a).shape, (2, 3))
    assert_shape(jnp.conjugate(a).shape, (2, 3))
    assert_shape(jnp.cos(a).shape, (2, 3))
    assert_shape(jnp.cosh(a).shape, (2, 3))
    assert_shape(jnp.deg2rad(a).shape, (2, 3))
    assert_shape(jnp.degrees(a).shape, (2, 3))
    assert_shape(jnp.exp(a).shape, (2, 3))
    assert_shape(jnp.exp2(a).shape, (2, 3))
    assert_shape(jnp.expm1(a).shape, (2, 3))
    assert_shape(jnp.fabs(a).shape, (2, 3))
    assert_shape(jnp.floor(a).shape, (2, 3))
    assert_shape(jnp.i0(a).shape, (2, 3))
    assert_shape(jnp.imag(a).shape, (2, 3))
    assert_shape(jnp.log(a).shape, (2, 3))
    assert_shape(jnp.log10(a).shape, (2, 3))
    assert_shape(jnp.log1p(a).shape, (2, 3))
    assert_shape(jnp.log2(a).shape, (2, 3))
    assert_shape(jnp.negative(a).shape, (2, 3))
    assert_shape(jnp.positive(a).shape, (2, 3))
    assert_shape(jnp.rad2deg(a).shape, (2, 3))
    assert_shape(jnp.radians(a).shape, (2, 3))
    assert_shape(jnp.real(a).shape, (2, 3))
    assert_shape(jnp.reciprocal(a).shape, (2, 3))
    assert_shape(jnp.rint(a).shape, (2, 3))
    assert_shape(jnp.round(a).shape, (2, 3))
    assert_shape(jnp.sign(a).shape, (2, 3))
    assert_shape(jnp.signbit(a).shape, (2, 3))
    assert_shape(jnp.sin(a).shape, (2, 3))
    assert_shape(jnp.sinc(a).shape, (2, 3))
    assert_shape(jnp.sinh(a).shape, (2, 3))
    assert_shape(jnp.spacing(a).shape, (2, 3))
    assert_shape(jnp.sqrt(a).shape, (2, 3))
    assert_shape(jnp.square(a).shape, (2, 3))
    assert_shape(jnp.tan(a).shape, (2, 3))
    assert_shape(jnp.tanh(a).shape, (2, 3))
    assert_shape(jnp.trunc(a).shape, (2, 3))
    assert_shape(jnp.unwrap(a).shape, (2, 3))


def test_tuple_returning_functions_preserve_shape() -> None:
    a = jnp.full((2, 3), 1.5)
    b = jnp.full((1, 3), 0.5)

    f1, f2 = jnp.frexp(a)
    assert_shape(f1.shape, (2, 3))
    assert_shape(f2.shape, (2, 3))

    m1, m2 = jnp.modf(a)
    assert_shape(m1.shape, (2, 3))
    assert_shape(m2.shape, (2, 3))

    d1, d2 = jnp.divmod(a, b)
    assert_shape(d1.shape, (2, 3))
    assert_shape(d2.shape, (2, 3))


def test_integer_elementwise_functions() -> None:
    a = jnp.ones((2, 3), dtype=jnp.int32)
    b = jnp.full((1, 3), 2, dtype=jnp.int32)

    assert_shape(jnp.bitwise_count(a).shape, (2, 3))
    assert_shape(jnp.bitwise_invert(a).shape, (2, 3))
    assert_shape(jnp.bitwise_not(a).shape, (2, 3))
    assert_shape(jnp.invert(a).shape, (2, 3))

    assert_shape(jnp.bitwise_and(a, b).shape, (2, 3))
    assert_shape(jnp.bitwise_or(a, b).shape, (2, 3))
    assert_shape(jnp.bitwise_xor(a, b).shape, (2, 3))
    assert_shape(jnp.bitwise_left_shift(a, b).shape, (2, 3))
    assert_shape(jnp.bitwise_right_shift(a, b).shape, (2, 3))
    assert_shape(jnp.left_shift(a, b).shape, (2, 3))
    assert_shape(jnp.right_shift(a, b).shape, (2, 3))
    assert_shape(jnp.gcd(a, b).shape, (2, 3))
    assert_shape(jnp.lcm(a, b).shape, (2, 3))


def test_additional_binary_broadcasting_functions() -> None:
    col = jnp.ones((3, 1))
    row = jnp.ones((1, 4))

    assert_shape(jnp.arctan2(col, row).shape, (3, 4))
    assert_shape(jnp.atan2(col, row).shape, (3, 4))
    assert_shape(jnp.copysign(col, row).shape, (3, 4))
    assert_shape(jnp.float_power(col, row).shape, (3, 4))
    assert_shape(jnp.floor_divide(col, row).shape, (3, 4))
    assert_shape(jnp.fmod(col, row).shape, (3, 4))
    assert_shape(jnp.heaviside(col, row).shape, (3, 4))
    assert_shape(jnp.hypot(col, row).shape, (3, 4))
    assert_shape(jnp.logaddexp(col, row).shape, (3, 4))
    assert_shape(jnp.logaddexp2(col, row).shape, (3, 4))
    assert_shape(jnp.mod(col, row).shape, (3, 4))
    assert_shape(jnp.nextafter(col, row).shape, (3, 4))
    assert_shape(jnp.pow(col, row).shape, (3, 4))
    assert_shape(jnp.power(col, row).shape, (3, 4))
    assert_shape(jnp.remainder(col, row).shape, (3, 4))
    assert_shape(jnp.true_divide(col, row).shape, (3, 4))


def test_subtraction_rejects_incompatible_broadcast() -> None:
    a = jnp.ones((3, 4))
    b = jnp.ones(5)

    assert_shape((a - jnp.ones((3, 4))).shape, (3, 4))
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

    assert_shape(jnp.maximum(a, jnp.ones((3, 4))).shape, (3, 4))
    try:
        jnp.maximum(  # E: Cannot evaluate type-level shape DSL call
            a, b
        )
    except (TypeError, ValueError):
        pass
    else:
        raise AssertionError("expected JAX to reject incompatible shapes")
