# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import numpy as np
from shape_extensions import assert_shape, Int, IntVar

N = IntVar("N")
D = IntVar("D")


def harmonic_oscillator_energy(
    position: np.ndarray[[N, D]],
    velocity: np.ndarray[[N, D]],
    stiffness: float,
    mass: float,
) -> np.ndarray[[N]]:
    """Compute the total oscillator energy for a batch of states.

    Each row is one independent oscillator state, and each column is a spatial
    coordinate. The potential energy is proportional to squared displacement
    from equilibrium, while the kinetic energy is proportional to squared
    velocity. Reducing over the coordinate axis keeps one scalar energy per
    state, so the result has shape `(N,)`.
    """
    potential_energy = 0.5 * stiffness * np.sum(position**2, axis=-1)
    kinetic_energy = 0.5 * mass * np.sum(velocity**2, axis=-1)
    return potential_energy + kinetic_energy


def linear_elastic_displacement(
    stiffness: np.ndarray[[N, N]],
    force: np.ndarray[[N, 1]],
) -> np.ndarray[[N, 1]]:
    """Solve a linear elastic equilibrium system.

    In small-displacement linear elasticity, the discretized equilibrium
    equation is `K u = f`: the stiffness matrix `K` maps unknown displacements
    `u` to applied forces `f`. Solving the square linear system preserves the
    force vector's column shape, producing a displacement vector with shape
    `(N, 1)`.
    """
    return np.linalg.solve(stiffness, force)


def gravitational_forces(
    position: np.ndarray[[N, 3]],
    mass: np.ndarray[[N]],
) -> np.ndarray[[N, 3]]:
    """Compute Newtonian gravitational forces for an n-body system.

    Each row of `position` is a particle's 3-D location, and `mass` stores one
    mass per particle. Inserting singleton axes creates all target-source
    particle pairs: `(1, N, 3) - (N, 1, 3)` broadcasts to `(N, N, 3)` vectors
    pointing from each target particle toward each source particle. The norm
    keeps a singleton final axis so the `(N, N, 1)` distance scale broadcasts
    back over the 3-D vector components. In units where the gravitational
    constant is `G = 1`, multiplying by target and source masses gives pairwise
    forces, and summing over source particles leaves one total force vector per
    target particle, with shape `(N, 3)`.
    """
    diff = position[None, :, :] - position[:, None, :]
    distance = np.linalg.norm(diff, axis=-1, keepdims=True)
    np.fill_diagonal(distance[:, :, 0], 1.0)
    forces = mass[:, None, None] * diff * (mass[None, :, None] / distance**3)
    return forces.sum(axis=1)


def particle_in_box(
    n_points: Int[N],
) -> tuple[np.ndarray[[N]], np.ndarray[[N, N]]]:
    """Solve a finite-difference quantum particle-in-a-box Hamiltonian.

    A one-dimensional particle in a box is a standard quantum mechanics model:
    the particle is confined to an interval, and the Hamiltonian's eigenvalues
    are the allowed energies while its eigenvectors are the wavefunctions. This
    dimensionless finite-difference version samples the interior of the interval
    at `N` points, so the negative second-derivative kinetic operator becomes an
    `N x N` tridiagonal matrix. The main diagonal has length `N`, while each
    off-diagonal has length `N - 1`; placing those off-diagonal vectors at
    offsets `+1` and `-1` still produces `N x N` matrices. Solving the symmetric
    eigenproblem returns `N` energy levels and an `N x N` matrix whose columns
    are the corresponding wavefunctions.
    """
    dx = 1.0 / (n_points + 1)
    diagonal = np.full(n_points, 2.0 / dx**2)
    off_diagonal = np.full(n_points - 1, -1.0 / dx**2)
    hamiltonian = (
        np.diag(diagonal) + np.diag(off_diagonal, 1) + np.diag(off_diagonal, -1)
    )
    return np.linalg.eigh(hamiltonian)


def test_harmonic_oscillator_energy() -> None:
    position = np.random.randn(5, 3)
    velocity = np.random.randn(5, 3)
    potential_energy = 0.5 * 2.0 * np.sum(position**2, axis=-1)
    kinetic_energy = 0.5 * 4.0 * np.sum(velocity**2, axis=-1)
    energy = harmonic_oscillator_energy(position, velocity, stiffness=2.0, mass=4.0)

    assert_shape(position, (5, 3))
    assert_shape(velocity, (5, 3))
    assert_shape(potential_energy, (5,))
    assert_shape(kinetic_energy, (5,))
    assert_shape(energy, (5,))


def test_linear_elastic_displacement() -> None:
    stiffness = np.eye(4)
    force = np.ones((4, 1))
    displacement = linear_elastic_displacement(stiffness, force)

    assert_shape(stiffness, (4, 4))
    assert_shape(force, (4, 1))
    assert_shape(displacement, (4, 1))


def test_gravitational_forces() -> None:
    position = np.random.randn(5, 3)
    mass = np.ones(5)
    diff = position[None, :, :] - position[:, None, :]
    distance = np.linalg.norm(diff, axis=-1, keepdims=True)
    np.fill_diagonal(distance[:, :, 0], 1.0)
    forces = mass[:, None, None] * diff * (mass[None, :, None] / distance**3)
    total_force = gravitational_forces(position, mass)

    assert_shape(position, (5, 3))
    assert_shape(mass, (5,))
    assert_shape(diff, (5, 5, 3))
    assert_shape(distance, (5, 5, 1))
    assert_shape(distance[:, :, 0], (5, 5))
    assert_shape(mass[:, None, None], (5, 1, 1))
    assert_shape(mass[None, :, None], (1, 5, 1))
    assert_shape(forces, (5, 5, 3))
    assert_shape(total_force, (5, 3))


def test_particle_in_box() -> None:
    # TODO: Remove this annotation once literal arithmetic preserves its result
    # or contextual typing lets `np.full` retain the off-diagonal length.
    n_points: Int[5] = 5
    dx = 1.0 / (n_points + 1)
    diagonal = np.full(n_points, 2.0 / dx**2)
    off_diagonal = np.full(n_points - 1, -1.0 / dx**2)
    hamiltonian = (
        np.diag(diagonal) + np.diag(off_diagonal, 1) + np.diag(off_diagonal, -1)
    )
    energies, wavefunctions = particle_in_box(n_points)

    assert_shape(diagonal, (5,))
    assert_shape(off_diagonal, (4,))
    assert_shape(np.diag(diagonal), (5, 5))
    assert_shape(np.diag(off_diagonal, 1), (5, 5))
    assert_shape(np.diag(off_diagonal, -1), (5, 5))
    assert_shape(hamiltonian, (5, 5))
    assert_shape(energies, (5,))
    assert_shape(wavefunctions, (5, 5))
