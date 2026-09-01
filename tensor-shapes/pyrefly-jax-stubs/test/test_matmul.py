# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type

import jax
import jax.numpy as jnp
from shape_extensions import assert_shape, IntTuple, IntVar

N = IntVar("N")
M = IntVar("M")
P = IntVar("P")
B = IntVar("B")


# A non-tuple sequence of axes is accepted but gradual; see `test_reductions.py`.
GRADUAL_SHAPE_RUNTIME_TESTS = {
    "test_transpose_accepts_any_axis_sequence",
    "test_transpose_method_variadic_and_sequence_forms",
}


def generic_matmul[N: IntVar, M: IntVar, P: IntVar](
    left: jax.Array[[N, M]],
    right: jax.Array[[M, P]],
) -> jax.Array[[N, P]]:
    """Symbolic dimensions compose, so this checks without concrete sizes."""

    return left @ right


def symbolic_matmul_dimensions_are_not_rejected(
    left: jax.Array[[N, M]],
    unrelated: jax.Array[[P, M]],
) -> None:
    left @ unrelated


def batched_matmul_is_not_rejected(
    x: jax.Array[[N, M, P]], y: jax.Array[[P, M]]
) -> None:
    """Batched matmul with matching contracting dimension."""

    jnp.matmul(x, y)


def gufunc_matmul_shapes(
    left_vector: jax.Array[[M]],
    right_vector: jax.Array[[P]],
    matrix: jax.Array[[M, P]],
    batched: jax.Array[[B, N, M]],
    broadcasted: jax.Array[[1, M, P]],
) -> None:
    assert_type(jnp.matmul(left_vector, matrix), jax.Array[[P]])
    assert_type(jnp.matmul(matrix, right_vector), jax.Array[[M]])
    assert_type(jnp.matmul(left_vector, left_vector), jax.Array[[]])
    assert_type(jnp.matmul(batched, broadcasted), jax.Array[[B, N, P]])


def gradual_matmul(
    left: jax.Array[IntTuple], right: jax.Array[IntTuple]
) -> jax.Array[IntTuple]:
    return jnp.matmul(left, right)


def reject_invalid_gufunc_matmul_shapes(
    bad_core: jax.Array[[2, 5, 6]],
    bad_batch: jax.Array[[3, 4, 6]],
) -> None:
    # E: Cannot evaluate type-level shape DSL call: gufunc: core dimension 'n' has conflicting extents 4 and 5
    jnp.matmul(jnp.ones((2, 3, 4)), bad_core)
    # E: Cannot evaluate type-level shape DSL call: Cannot broadcast dimension Int[2] with dimension Int[3] at position 0
    jnp.matmul(jnp.ones((2, 3, 4)), bad_batch)


def reject_scalar_matmul(scalar: jax.Array[[]], vector: jax.Array[[M]]) -> None:
    # E: Cannot evaluate type-level shape DSL call: matmul expects at least 1-D arrays
    jnp.matmul(scalar, vector)


def test_operator_matmul() -> None:
    left = jnp.ones((3, 4))
    right = jnp.ones((4, 5))

    assert_shape(left @ right, (3, 5))


def test_function_matmul() -> None:
    left = jnp.ones((2, 3))
    right = jnp.ones((3, 7))

    assert_shape(jnp.matmul(left, right), (2, 7))
    # JAX names both operands, unlike the ufuncs, which are positional-only.
    assert_shape(jnp.matmul(a=left, b=right), (2, 7))


def test_batched_matmul() -> None:
    vec4 = jnp.ones(4)
    mat34 = jnp.ones((3, 4))
    mat45 = jnp.ones((4, 5))
    batch_234 = jnp.ones((2, 3, 4))
    batch_245 = jnp.ones((2, 4, 5))

    # 1D @ 1D -> ()
    assert_shape(jnp.matmul(vec4, vec4), ())

    # 1D @ ND -> (*batch, m)
    assert_shape(jnp.matmul(vec4, mat45), (5,))
    assert_shape(jnp.matmul(vec4, batch_245), (2, 5))

    # ND @ 1D -> (*batch, n)
    assert_shape(jnp.matmul(mat34, vec4), (3,))
    assert_shape(jnp.matmul(batch_234, vec4), (2, 3))

    # ND @ ND -> (*broadcast(batch_left, batch_right), n, m)
    assert_shape(jnp.matmul(mat34, mat45), (3, 5))
    assert_shape(jnp.matmul(batch_234, mat45), (2, 3, 5))
    assert_shape(jnp.matmul(batch_234, batch_245), (2, 3, 5))
    assert_shape(batch_234 @ mat45, (2, 3, 5))
    assert_shape(batch_234 @ batch_245, (2, 3, 5))


def test_matmul_contracts_vector_operands() -> None:
    a = jnp.ones((3, 4))
    v = jnp.ones(4)

    # A 1-D operand contributes no axis to the result.
    assert_shape(a @ v, (3,))
    assert_shape(jnp.matmul(a, v), (3,))
    assert_shape(v @ a.T, (3,))
    assert_shape(v @ v, ())


def test_function_matmul_rejects_mismatched_inner_dimension() -> None:
    a = jnp.ones((3, 4))

    assert_shape(jnp.matmul(a, jnp.ones((4, 5))), (3, 5))
    try:
        # E: Cannot evaluate type-level shape DSL call: gufunc: core dimension 'n' has conflicting extents 4 and 7
        jnp.matmul(a, jnp.ones((7, 5)))
    except TypeError:
        pass
    else:
        raise AssertionError("expected JAX to reject mismatched inner dimensions")


def test_transpose_method() -> None:
    a = jnp.ones((3, 4))
    c = jnp.ones((2, 3, 4))

    assert_shape(a.transpose(), (4, 3))
    assert_shape(a.transpose((1, 0)), (4, 3))
    assert_shape(c.transpose((0, 2, 1)), (2, 4, 3))


def test_transpose_method_variadic_and_sequence_forms() -> None:
    c = jnp.ones((2, 3, 4))

    # Both are gradual: a variadic argument list cannot be captured as a Flag,
    # and a list is not a Flag domain.
    assert c.transpose(0, 2, 1).shape == (2, 4, 3)
    assert c.transpose([2, 0, 1]).shape == (4, 2, 3)


def test_transpose_rejects_non_sequence_and_wrong_length_axes() -> None:
    a = jnp.ones((3, 4))
    c = jnp.ones((2, 3, 4))

    assert_shape(jnp.transpose(a, (1, 0)), (4, 3))
    try:
        # E: Cannot evaluate type-level shape DSL call: transpose axes must be a sequence
        jnp.transpose(c, 1)
    except TypeError:
        pass
    else:
        raise AssertionError("expected JAX to reject a scalar axis")
    try:
        # E: Cannot evaluate type-level shape DSL call: transpose axes must cover every dimension
        jnp.transpose(a, (1, 0, 2))
    except ValueError:
        pass
    else:
        raise AssertionError("expected JAX to reject a wrong-length permutation")


def test_transpose() -> None:
    a = jnp.ones((3, 4))

    assert_shape(a.T, (4, 3))
    assert_shape(jnp.transpose(a), (4, 3))
    assert_shape(a.T @ a, (4, 4))


def test_transpose_reverses_every_axis() -> None:
    c = jnp.ones((2, 3, 4))

    assert_shape(c.T, (4, 3, 2))
    assert_shape(jnp.transpose(c), (4, 3, 2))


def test_transpose_with_explicit_axes() -> None:
    c = jnp.ones((2, 3, 4))

    assert_shape(jnp.transpose(c, (0, 2, 1)), (2, 4, 3))
    assert_shape(jnp.transpose(c, (2, 0, 1)), (4, 2, 3))
    assert_shape(jnp.transpose(jnp.ones((3, 4)), axes=(1, 0)), (4, 3))
    assert_shape(jnp.transpose(c, (0, -1, 1)), (2, 4, 3))


def test_transpose_accepts_any_axis_sequence() -> None:
    c = jnp.ones((2, 3, 4))

    assert jnp.transpose(c, [2, 0, 1]).shape == (4, 2, 3)
    assert jnp.transpose(c, range(3)).shape == (2, 3, 4)


def test_transpose_rejects_bad_axes() -> None:
    c = jnp.ones((2, 3, 4))

    assert_shape(jnp.transpose(c, (1, 0, 2)), (3, 2, 4))
    try:
        # E: Cannot evaluate type-level shape DSL call: duplicate axis
        jnp.transpose(c, (0, 0, 1))
    except TypeError:
        pass
    else:
        raise AssertionError("expected JAX to reject a duplicate axis")
    try:
        # E: Cannot evaluate type-level shape DSL call: axis out of bounds
        jnp.transpose(c, (0, 1, 3))
    except ValueError:
        pass
    else:
        raise AssertionError("expected JAX to reject an out-of-bounds axis")


def test_matmul_rejects_mismatched_inner_dimension() -> None:
    left = jnp.ones((3, 4))
    right = jnp.ones((7, 5))

    assert_shape(left @ jnp.ones((4, 5)), (3, 5))
    try:
        left @ right  # E: `@` is not supported
    except TypeError:
        pass
    else:
        raise AssertionError("expected JAX to reject mismatched inner dimensions")
