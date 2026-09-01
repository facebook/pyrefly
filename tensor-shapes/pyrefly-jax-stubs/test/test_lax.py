# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import jax
import jax.lax as lax
import jax.numpy as jnp
from shape_extensions import assert_shape, IntTuple


def generic_unary_preserves_shape[Shape: IntTuple](
    x: jax.Array[Shape],
) -> jax.Array[Shape]:
    return lax.sin(x)


def test_unary_elementwise() -> None:
    vec = jnp.ones(4)
    mat = jnp.ones((2, 3))
    tensor = jnp.ones((2, 3, 4))

    assert_shape(lax.abs(vec), (4,))
    assert_shape(lax.neg(mat), (2, 3))
    assert_shape(lax.exp(tensor), (2, 3, 4))
    assert_shape(lax.log(mat), (2, 3))
    assert_shape(lax.sqrt(vec), (4,))
    assert_shape(lax.sin(mat), (2, 3))
    assert_shape(lax.cos(tensor), (2, 3, 4))
    assert_shape(lax.tan(vec), (4,))
    assert_shape(lax.tanh(mat), (2, 3))
    assert_shape(lax.ceil(vec), (4,))
    assert_shape(lax.floor(mat), (2, 3))
    assert_shape(lax.round(tensor), (2, 3, 4))
    assert_shape(lax.sign(vec), (4,))
    assert_shape(lax.square(mat), (2, 3))
    assert_shape(lax.rsqrt(tensor), (2, 3, 4))
    assert_shape(lax.is_finite(vec), (4,))
    assert_shape(lax.integer_pow(mat, 3), (2, 3))

    c_vec = vec * 1j
    c_mat = mat * 1j
    c_tensor = tensor * 1j
    assert_shape(lax.real(c_vec), (4,))
    assert_shape(lax.imag(c_mat), (2, 3))
    assert_shape(lax.conj(c_tensor), (2, 3, 4))


def test_binary_broadcasting_with_scalars() -> None:
    scalar_arr = jnp.ones(())
    vec = jnp.ones(4)
    mat = jnp.ones((2, 3))

    # Python scalar with Array
    assert_shape(lax.add(1.0, vec), (4,))
    assert_shape(lax.add(vec, 1.0), (4,))
    assert_shape(lax.sub(2.0, mat), (2, 3))
    assert_shape(lax.mul(mat, 3.0), (2, 3))

    # 0-D Array with N-D Array
    assert_shape(lax.add(scalar_arr, vec), (4,))
    assert_shape(lax.add(vec, scalar_arr), (4,))
    assert_shape(lax.add(scalar_arr, mat), (2, 3))
    assert_shape(lax.add(mat, scalar_arr), (2, 3))
    assert_shape(lax.add(scalar_arr, scalar_arr), ())


def test_binary_broadcasting_same_rank() -> None:
    row = jnp.ones((1, 4))
    col = jnp.ones((3, 1))
    mat = jnp.ones((3, 4))

    assert_shape(lax.add(row, col), (3, 4))
    assert_shape(lax.add(col, row), (3, 4))
    assert_shape(lax.add(row, mat), (3, 4))
    assert_shape(lax.add(mat, col), (3, 4))

    # 3-D same rank broadcasting
    a3d = jnp.ones((2, 1, 4))
    b3d = jnp.ones((1, 3, 4))
    assert_shape(lax.add(a3d, b3d), (2, 3, 4))
    assert_shape(lax.sub(a3d, b3d), (2, 3, 4))
    assert_shape(lax.mul(a3d, b3d), (2, 3, 4))
    assert_shape(lax.div(a3d, b3d), (2, 3, 4))
    assert_shape(lax.max(a3d, b3d), (2, 3, 4))
    assert_shape(lax.min(a3d, b3d), (2, 3, 4))
    assert_shape(lax.atan2(a3d, b3d), (2, 3, 4))
    assert_shape(lax.pow(a3d, b3d), (2, 3, 4))
    assert_shape(lax.rem(a3d, b3d), (2, 3, 4))


def test_binary_bitwise_and_comparison() -> None:
    a = jnp.ones((2, 3), dtype=jnp.int32)
    b = jnp.ones((1, 3), dtype=jnp.int32)

    assert_shape(lax.bitwise_and(a, b), (2, 3))
    assert_shape(lax.bitwise_or(a, b), (2, 3))
    assert_shape(lax.bitwise_xor(a, b), (2, 3))
    assert_shape(lax.shift_left(a, b), (2, 3))
    assert_shape(lax.shift_right_arithmetic(a, b), (2, 3))
    assert_shape(lax.shift_right_logical(a, b), (2, 3))

    assert_shape(lax.eq(a, b), (2, 3))
    assert_shape(lax.ne(a, b), (2, 3))
    assert_shape(lax.lt(a, b), (2, 3))
    assert_shape(lax.le(a, b), (2, 3))
    assert_shape(lax.gt(a, b), (2, 3))
    assert_shape(lax.ge(a, b), (2, 3))


def test_binary_rejects_differing_non_scalar_ranks() -> None:
    vec = jnp.ones(4)
    row = jnp.ones((1, 4))
    mat = jnp.ones((3, 1))

    # Same rank broadcasting works
    assert_shape(lax.add(row, mat), (3, 4))

    try:
        # E: Cannot evaluate type-level shape DSL call: arrays must have the same number of dimensions
        lax.add(vec, mat)
    except TypeError:
        pass
    else:
        raise AssertionError(
            "expected JAX to reject differing-rank non-scalar operands in lax.add"
        )


def test_binary_rejects_incompatible_dimensions() -> None:
    a = jnp.ones((2, 3))
    b = jnp.ones((2, 4))

    assert_shape(lax.add(a, jnp.ones((2, 3))), (2, 3))
    try:
        # E: Cannot evaluate type-level shape DSL call: incompatible shapes for broadcasting
        lax.add(a, b)
    except TypeError:
        pass
    else:
        raise AssertionError(
            "expected JAX to reject incompatible dimensions in lax.add"
        )
