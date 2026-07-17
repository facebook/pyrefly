# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import numpy as np
from shape_extensions import assert_shape, Int, IntVar

N = IntVar("N")


def square_svd_components(
    x: np.ndarray[tuple[Int[N], Int[N]]],
) -> np.ndarray[tuple[Int[N], Int[N]]]:
    _u, _s, vt = np.linalg.svd(x, full_matrices=False)
    return vt


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
) -> tuple[np.ndarray[tuple[Int[N]]], np.ndarray[tuple[Int[N], Int[N]]]]:
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
    pos: np.ndarray[tuple[Int[N], Int[3]]],
    mass: np.ndarray[tuple[Int[N]]],
) -> np.ndarray[tuple[Int[N], Int[3]]]:
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


def test_svd_reduced_square_matrix() -> None:
    x = np.ones((4, 4))

    u, s, vt = np.linalg.svd(x, full_matrices=False)

    assert_shape(u, (4, 4))
    assert_shape(s, (4,))
    assert_shape(vt, (4, 4))
    assert_shape(square_svd_components(x), (4, 4))


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
        # E: `@` is not supported
        assert_shape(a @ b, (3, 4))
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
        # E: matmul inner dimensions must match
        np.matmul(a, b)
    except ValueError:
        pass
    else:
        raise AssertionError("expected NumPy to reject mismatched inner dimensions")
