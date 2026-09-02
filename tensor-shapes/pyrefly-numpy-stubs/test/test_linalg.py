# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type

import numpy as np
from shape_extensions import assert_shape, Int, IntTuple, IntVar

N = IntVar("N")
M = IntVar("M")


def square_svd_components[N: IntVar, DType](
    x: np.ndarray[[N, N], DType],
) -> np.ndarray[[N, N], DType]:
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    assert_type(u, np.ndarray[[N, N], DType])
    assert_type(s, np.ndarray[[N], DType])
    assert_type(vt, np.ndarray[[N, N], DType])
    return vt


def unrelated_svd_components[M: IntVar, N: IntVar, DType](
    x: np.ndarray[[M, N], DType],
) -> tuple[
    np.ndarray[[M, int], DType],
    np.ndarray[[int], DType],
    np.ndarray[[int, N], DType],
]:
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    assert_type(u, np.ndarray[[M, int], DType])
    assert_type(s, np.ndarray[[int], DType])
    assert_type(vt, np.ndarray[[int, N], DType])
    return u, s, vt


def reject_non_2d_svd(x: np.ndarray[[M, N, 3]]) -> None:
    np.linalg.svd(x, full_matrices=False)  # E: not assignable


def reject_unsupported_svd_options(x: np.ndarray[[M, N]]) -> None:
    np.linalg.svd(x, full_matrices=True)  # E: not assignable
    np.linalg.svd(x, full_matrices=False, compute_uv=False)  # E: not assignable
    np.linalg.svd(x, full_matrices=False, hermitian=True)  # E: not assignable


def generic_matmul[
    N: IntVar,
    M: IntVar,
    K: IntVar,
    P: IntVar,
    DType,
](
    left: np.ndarray[[N, M], DType],
    same_inner: np.ndarray[[M, P]],
    unrelated_inner: np.ndarray[[K, P]],
    gradual_inner: np.ndarray[[int, P]],
) -> None:
    # Symbolic inner dimensions cannot prove a mismatch, so the DSL preserves
    # the known outer dimensions in both the shared and unrelated cases.
    assert_type(np.matmul(left, same_inner), np.ndarray[[N, P]])
    assert_type(left @ same_inner, np.ndarray[[N, P], DType])
    assert_type(np.matmul(left, unrelated_inner), np.ndarray[[N, P]])
    left @ unrelated_inner  # E: `@` is not supported
    assert_type(np.matmul(left, gradual_inner), np.ndarray[[N, P]])
    assert_type(left @ gradual_inner, np.ndarray[[N, P], DType])


def concrete_inner_matmul[N: IntVar, P: IntVar, DType](
    left: np.ndarray[[N, 4], DType],
    same_inner: np.ndarray[[4, P]],
    different_inner: np.ndarray[[5, P]],
) -> None:
    assert_type(np.matmul(left, same_inner), np.ndarray[[N, P]])
    assert_type(left @ same_inner, np.ndarray[[N, P], DType])
    # E: Cannot evaluate type-level shape DSL call: matmul inner dimensions must match
    np.matmul(left, different_inner)
    left @ different_inner  # E: Shape dimension mismatch


def symbolic_concrete_inner_matmul[N: IntVar, M: IntVar, P: IntVar](
    symbolic_left: np.ndarray[[N, M]],
    concrete_right: np.ndarray[[4, P]],
    concrete_left: np.ndarray[[N, 4]],
    symbolic_right: np.ndarray[[M, P]],
) -> None:
    assert_type(np.matmul(symbolic_left, concrete_right), np.ndarray[[N, P]])
    assert_type(np.matmul(concrete_left, symbolic_right), np.ndarray[[N, P]])


def gradual_matmul[DType](
    left: np.ndarray[IntTuple, DType], right: np.ndarray[IntTuple]
) -> None:
    assert_type(np.matmul(left, right), np.ndarray[IntTuple])
    assert_type(left @ right, np.ndarray[[int, int], DType])


def bare_matmul(left: np.ndarray, right: np.ndarray) -> None:
    assert_type(np.matmul(left, right), np.ndarray)


def same_dtype_matmul(
    left: np.ndarray[[3, 4], np.dtype[np.float64]],
    right: np.ndarray[[4, 5], np.dtype[np.float64]],
) -> None:
    assert_type(np.matmul(left, right), np.ndarray[[3, 5]])


def mixed_dtype_matmul(
    left: np.ndarray[[3, 4], np.dtype[np.int32]],
    right: np.ndarray[[4, 5], np.dtype[np.float64]],
) -> None:
    # Shape tracking does not model NumPy's dtype-promotion rules.
    assert_type(np.matmul(left, right), np.ndarray[[3, 5]])


def known_limitation_matmul_rejects_valid_non_2d_inputs(
    batched: np.ndarray[[2, 3, 4]],
) -> None:
    # NumPy supports vector and batched matmul, but the shape-stub MVP models
    # only the 2-D contract.
    vector = np.ones(4)
    matrix = np.ones((4, 5))
    vector @ matrix  # E: `@` is not supported
    matrix @ vector  # E: `@` is not supported
    # E: Cannot evaluate type-level shape DSL call: matmul expects 2-D arrays
    np.matmul(vector, matrix)
    # E: Cannot evaluate type-level shape DSL call: matmul expects 2-D arrays
    np.matmul(matrix, vector)
    # E: Cannot evaluate type-level shape DSL call: matmul expects 2-D arrays
    np.matmul(batched, matrix)


def reject_invalid_rank_with_gradual_other(
    gradual: np.ndarray[IntTuple], batched: np.ndarray[[2, 3, 4]]
) -> None:
    # E: Cannot evaluate type-level shape DSL call: matmul expects 2-D arrays
    np.matmul(gradual, batched)
    # E: Cannot evaluate type-level shape DSL call: matmul expects 2-D arrays
    np.matmul(batched, gradual)


def test_matmul_function_2d() -> None:
    a = np.ones((3, 4))
    b = np.ones((4, 5))

    assert_shape(np.matmul(a, b), (3, 5))


def test_matmul_operator_2d() -> None:
    a = np.ones((3, 4))
    b = np.ones((4, 5))

    assert_shape(a @ b, (3, 5))


def test_transpose_property_2d() -> None:
    x = np.ones((3, 4))
    y = np.ones((3, 1))

    assert_shape(x.T, (4, 3))
    assert_shape(x.T.T, (3, 4))
    assert_shape(x.T @ x, (4, 4))
    assert_shape(x.T @ y, (4, 1))


def test_solve_vector_rhs() -> None:
    a = np.eye(3)
    b = np.ones(3)

    assert_shape(np.linalg.solve(a, b), (3,))


def test_solve_matrix_rhs() -> None:
    a = np.eye(3)
    b = np.ones((3, 2))

    assert_shape(np.linalg.solve(a, b), (3, 2))


def test_solve_column_rhs_regression_composition() -> None:
    x = np.random.randn(5, 3)
    y = np.random.randn(5, 1)

    assert_shape(np.linalg.solve(x.T @ x, x.T @ y), (3, 1))


def test_eigh_square_matrix() -> None:
    hamiltonian = np.eye(5)
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)

    assert_shape(eigenvalues, (5,))
    assert_shape(eigenvectors, (5, 5))


def particle_in_box_shape_path(
    n_points: Int[N],
) -> tuple[np.ndarray[[N]], np.ndarray[[N, N]]]:
    dx = 1.0 / (n_points + 1)
    diagonal = np.full(n_points, 2.0 / dx**2)
    off_diagonal = np.full(n_points - 1, -1.0 / dx**2)
    hamiltonian = (
        np.diag(diagonal) + np.diag(off_diagonal, 1) + np.diag(off_diagonal, -1)
    )
    return np.linalg.eigh(hamiltonian)


def test_particle_in_box_shape_path() -> None:
    energies, wavefunctions = particle_in_box_shape_path(5)

    assert_shape(energies, (5,))
    assert_shape(wavefunctions, (5, 5))


def test_norm_3d_axis_keepdims_for_nbody() -> None:
    positions = np.ones((5, 3))
    pairwise_deltas = positions[:, None, :] - positions[None, :, :]

    assert_shape(np.linalg.norm(pairwise_deltas, axis=-1, keepdims=True), (5, 5, 1))


def gravitational_force_shape_path(
    pos: np.ndarray[[N, 3]],
    mass: np.ndarray[[N]],
) -> np.ndarray[[N, 3]]:
    diff = pos[None, :, :] - pos[:, None, :]
    dist = np.linalg.norm(diff, axis=-1, keepdims=True)
    np.fill_diagonal(dist[:, :, 0], 1.0)
    forces = mass[:, None, None] * diff * (mass[None, :, None] / dist**3)
    return forces.sum(axis=1)


def test_nbody_force_shape_path() -> None:
    pos = np.ones((5, 3))
    mass = np.ones(5)

    assert_shape(gravitational_force_shape_path(pos, mass), (5, 3))


def test_svd_reduced_wide_matrix() -> None:
    x = np.ones((3, 5))

    u, s, vt = np.linalg.svd(x, full_matrices=False)

    assert_shape(u, (3, 3))
    assert_shape(s, (3,))
    assert_shape(vt, (3, 5))


def test_svd_reduced_tall_matrix() -> None:
    x = np.ones((5, 3))

    u, s, vt = np.linalg.svd(x, full_matrices=False)

    assert_shape(u, (5, 3))
    assert_shape(s, (3,))
    assert_shape(vt, (3, 3))


def test_svd_reduced_square_matrix() -> None:
    x = np.ones((4, 4))

    u, s, vt = np.linalg.svd(x, full_matrices=False)

    assert_shape(u, (4, 4))
    assert_shape(s, (4,))
    assert_shape(vt, (4, 4))
    assert_shape(square_svd_components(x), (4, 4))


def test_svd_reduced_dtype_preserved() -> None:
    x = np.ones((5, 3), dtype=np.float32)

    u, s, vt = np.linalg.svd(x, full_matrices=False)

    assert_shape(u, (5, 3))
    assert_shape(s, (3,))
    assert_shape(vt, (3, 3))
    assert_type(u.dtype, np.dtype[np.float32])
    assert_type(s.dtype, np.dtype[np.float32])
    assert_type(vt.dtype, np.dtype[np.float32])


def test_svd_all_component_pca_projection() -> None:
    x = np.random.randn(5, 3)
    x_centered = x - x.mean(axis=0)
    u, s, vt = np.linalg.svd(x_centered, full_matrices=False)
    projection = x_centered @ vt.T

    assert_shape(u, (5, 3))
    assert_shape(s, (3,))
    assert_shape(vt, (3, 3))
    assert_shape(projection, (5, 3))


def test_matmul_operator_rejects_mismatched_inner_dimension() -> None:
    a = np.ones((3, 4))
    b = np.ones((6, 5))

    # The mismatched matmul below raises before assert_shape runs, so anchor the
    # well-formed shape here to satisfy run_runtime_tests' "every test asserts a
    # shape" invariant.
    assert_shape(np.ones((3, 4)) @ np.ones((4, 5)), (3, 5))
    try:
        # `a @ b` is rejected statically (mismatched inner dims) and raises at
        # runtime; the well-formed anchor above satisfies the shape-assertion
        # invariant, so this line only needs to exercise the rejection.
        a @ b  # E: `@` is not supported
    except ValueError:
        pass
    else:
        raise AssertionError("expected NumPy to reject mismatched inner dimensions")


def test_matmul_rejects_mismatched_inner_dimension() -> None:
    a = np.ones((3, 4))
    b = np.ones((6, 5))

    # The mismatched matmul below raises before assert_shape runs, so anchor the
    # well-formed shape here to satisfy run_runtime_tests' "every test asserts a
    # shape" invariant.
    assert_shape(np.matmul(np.ones((3, 4)), np.ones((4, 5))), (3, 5))
    try:
        # E: Cannot evaluate type-level shape DSL call: matmul inner dimensions must match
        np.matmul(a, b)
    except ValueError:
        pass
    else:
        raise AssertionError("expected NumPy to reject mismatched inner dimensions")
