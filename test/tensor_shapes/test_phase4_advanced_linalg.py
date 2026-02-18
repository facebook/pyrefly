# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Phase 4: Advanced linear algebra operations tests
from typing import assert_type

import torch
import torch.linalg
from torch import Tensor

# ==== torch.linalg.eig ====


def test_eig_2d():
    """Eigenvalue decomposition"""
    A: Tensor[4, 4] = torch.randn(4, 4)
    eigenvalues, eigenvectors = torch.linalg.eig(A)
    # Eigenvalues: (4,), Eigenvectors: (4, 4)
    assert_type(eigenvalues, Tensor[4])
    assert_type(eigenvectors, Tensor[4, 4])


def test_eig_batched():
    """Eigenvalue decomposition with batch dims"""
    A: Tensor[2, 3, 5, 5] = torch.randn(2, 3, 5, 5)
    eigenvalues, eigenvectors = torch.linalg.eig(A)
    # Eigenvalues: (2, 3, 5), Eigenvectors: (2, 3, 5, 5)
    assert_type(eigenvalues, Tensor[2, 3, 5])
    assert_type(eigenvectors, Tensor[2, 3, 5, 5])


def test_eigh_symmetric():
    """Hermitian/symmetric eigenvalue decomposition"""
    A: Tensor[4, 4] = torch.randn(4, 4)
    eigenvalues, eigenvectors = torch.linalg.eigh(A)
    # Same shape as eig
    assert_type(eigenvalues, Tensor[4])
    assert_type(eigenvectors, Tensor[4, 4])


# ==== torch.linalg.cholesky ====


def test_cholesky_2d():
    """Cholesky decomposition"""
    A: Tensor[5, 5] = torch.randn(5, 5)
    L = torch.linalg.cholesky(A)
    # Preserves shape: (5, 5)
    assert_type(L, Tensor[5, 5])


def test_cholesky_batched():
    """Cholesky with batch dimensions"""
    A: Tensor[2, 3, 4, 4] = torch.randn(2, 3, 4, 4)
    L = torch.linalg.cholesky(A)
    # Preserves shape: (2, 3, 4, 4)
    assert_type(L, Tensor[2, 3, 4, 4])


def test_cholesky_method():
    """Cholesky as Tensor method"""
    A: Tensor[3, 3] = torch.randn(3, 3)
    L = A.cholesky()
    # Preserves shape: (3, 3)
    assert_type(L, Tensor[3, 3])


# ==== torch.linalg.solve ====


def test_solve_2d():
    """Solve linear system Ax = b"""
    A: Tensor[4, 4] = torch.randn(4, 4)
    b: Tensor[4, 2] = torch.randn(4, 2)
    x = torch.linalg.solve(A, b)
    # Output has same shape as b: (4, 2)
    assert_type(x, Tensor[4, 2])


def test_solve_1d_rhs():
    """Solve with 1D right-hand side"""
    A: Tensor[5, 5] = torch.randn(5, 5)
    b: Tensor[5] = torch.randn(5)
    x = torch.linalg.solve(A, b)
    # Output has same shape as b: (5,)
    assert_type(x, Tensor[5])


def test_solve_batched():
    """Solve with batch dimensions"""
    A: Tensor[2, 3, 4, 4] = torch.randn(2, 3, 4, 4)
    b: Tensor[2, 3, 4, 1] = torch.randn(2, 3, 4, 1)
    x = torch.linalg.solve(A, b)
    # Output has same shape as b: (2, 3, 4, 1)
    assert_type(x, Tensor[2, 3, 4, 1])


# ==== torch.triangular_solve ====


def test_triangular_solve():
    """Triangular system solver"""
    A: Tensor[5, 5] = torch.randn(5, 5)
    b: Tensor[5, 3] = torch.randn(5, 3)
    x = torch.triangular_solve(b, A)
    # Output has same shape as b: (5, 3)
    assert_type(x, Tensor[5, 3])


# ==== torch.cholesky_solve ====


def test_cholesky_solve():
    """Cholesky-based solver"""
    L: Tensor[4, 4] = torch.randn(4, 4)
    b: Tensor[4, 2] = torch.randn(4, 2)
    x = torch.cholesky_solve(b, L)
    # Output has same shape as b: (4, 2)
    assert_type(x, Tensor[4, 2])


# ==== torch.linalg.inv ====


def test_inverse_2d():
    """Matrix inverse"""
    A: Tensor[5, 5] = torch.randn(5, 5)
    A_inv = torch.linalg.inv(A)
    # Preserves shape: (5, 5)
    assert_type(A_inv, Tensor[5, 5])


def test_inverse_batched():
    """Matrix inverse with batch dimensions"""
    A: Tensor[2, 3, 4, 4] = torch.randn(2, 3, 4, 4)
    A_inv = torch.linalg.inv(A)
    # Preserves shape: (2, 3, 4, 4)
    assert_type(A_inv, Tensor[2, 3, 4, 4])


def test_inverse_method():
    """Matrix inverse as Tensor method"""
    A: Tensor[3, 3] = torch.randn(3, 3)
    A_inv = A.inverse()
    # Preserves shape: (3, 3)
    assert_type(A_inv, Tensor[3, 3])


# ==== torch.linalg.det ====


def test_det_2d():
    """Determinant"""
    A: Tensor[4, 4] = torch.randn(4, 4)
    d = torch.linalg.det(A)
    # Returns scalar: ()
    assert_type(d, Tensor[()])


def test_det_batched():
    """Determinant with batch dimensions"""
    A: Tensor[2, 3, 5, 5] = torch.randn(2, 3, 5, 5)
    d = torch.linalg.det(A)
    # Returns batch dims only: (2, 3)
    assert_type(d, Tensor[2, 3])


def test_det_method():
    """Determinant as Tensor method"""
    A: Tensor[3, 3] = torch.randn(3, 3)
    d = A.det()
    # Returns scalar: ()
    assert_type(d, Tensor[()])


# ==== torch.logdet ====


def test_logdet():
    """Log determinant"""
    A: Tensor[4, 4] = torch.randn(4, 4)
    log_d = torch.logdet(A)
    # Returns scalar: ()
    assert_type(log_d, Tensor[()])


def test_logdet_batched():
    """Log determinant with batch dimensions"""
    A: Tensor[2, 5, 5] = torch.randn(2, 5, 5)
    log_d = torch.logdet(A)
    # Returns batch dims: (2,)
    assert_type(log_d, Tensor[2])


# ==== torch.linalg.slogdet ====


def test_slogdet_2d():
    """Sign and log determinant"""
    A: Tensor[4, 4] = torch.randn(4, 4)
    sign, logabsdet = torch.linalg.slogdet(A)
    # Both return scalars: ()
    assert_type(sign, Tensor[()])
    assert_type(logabsdet, Tensor[()])


def test_slogdet_batched():
    """Sign and log determinant with batch dimensions"""
    A: Tensor[2, 3, 5, 5] = torch.randn(2, 3, 5, 5)
    sign, logabsdet = torch.linalg.slogdet(A)
    # Both return batch dims: (2, 3)
    assert_type(sign, Tensor[2, 3])
    assert_type(logabsdet, Tensor[2, 3])


def test_slogdet_method():
    """Sign and log determinant as Tensor method"""
    A: Tensor[4, 4] = torch.randn(4, 4)
    sign, logabsdet = A.slogdet()
    # Both return scalars: ()
    assert_type(sign, Tensor[()])
    assert_type(logabsdet, Tensor[()])


# ==== torch.linalg.matrix_power ====


def test_matrix_power_2d():
    """Matrix power"""
    A: Tensor[4, 4] = torch.randn(4, 4)
    A_squared = torch.linalg.matrix_power(A, 2)
    # Preserves shape: (4, 4)
    assert_type(A_squared, Tensor[4, 4])


def test_matrix_power_batched():
    """Matrix power with batch dimensions"""
    A: Tensor[2, 3, 5, 5] = torch.randn(2, 3, 5, 5)
    A_cubed = torch.linalg.matrix_power(A, 3)
    # Preserves shape: (2, 3, 5, 5)
    assert_type(A_cubed, Tensor[2, 3, 5, 5])


def test_matrix_power_method():
    """Matrix power as Tensor method"""
    A: Tensor[3, 3] = torch.randn(3, 3)
    A_inv = A.matrix_power(-1)
    # Preserves shape: (3, 3)
    assert_type(A_inv, Tensor[3, 3])


# ==== torch.linalg.matrix_exp ====


def test_matrix_exp():
    """Matrix exponential"""
    A: Tensor[4, 4] = torch.randn(4, 4)
    exp_A = torch.linalg.matrix_exp(A)
    # Preserves shape: (4, 4)
    assert_type(exp_A, Tensor[4, 4])


def test_matrix_exp_batched():
    """Matrix exponential with batch dimensions"""
    A: Tensor[2, 3, 4, 4] = torch.randn(2, 3, 4, 4)
    exp_A = torch.linalg.matrix_exp(A)
    # Preserves shape: (2, 3, 4, 4)
    assert_type(exp_A, Tensor[2, 3, 4, 4])


# ==== torch.trace ====


def test_trace_2d():
    """Matrix trace"""
    A: Tensor[5, 5] = torch.randn(5, 5)
    tr = torch.trace(A)
    # Returns scalar: ()
    assert_type(tr, Tensor[()])


def test_trace_rectangular():
    """Trace of rectangular matrix"""
    A: Tensor[4, 6] = torch.randn(4, 6)
    tr = torch.trace(A)
    # Returns scalar (trace of min(m,n) diagonal): ()
    assert_type(tr, Tensor[()])


def test_trace_batched():
    """Trace with batch dimensions"""
    A: Tensor[2, 3, 5, 5] = torch.randn(2, 3, 5, 5)
    tr = torch.trace(A)
    # Returns batch dims: (2, 3)
    assert_type(tr, Tensor[2, 3])


def test_trace_method():
    """Trace as Tensor method"""
    A: Tensor[4, 4] = torch.randn(4, 4)
    tr = A.trace()
    # Returns scalar: ()
    assert_type(tr, Tensor[()])


# ==== torch.linalg.matrix_rank ====


def test_matrix_rank_2d():
    """Matrix rank"""
    A: Tensor[5, 4] = torch.randn(5, 4)
    rank = torch.linalg.matrix_rank(A)
    # Returns scalar: ()
    assert_type(rank, Tensor[()])


def test_matrix_rank_batched():
    """Matrix rank with batch dimensions"""
    A: Tensor[2, 3, 4, 5] = torch.randn(2, 3, 4, 5)
    rank = torch.linalg.matrix_rank(A)
    # Returns batch dims: (2, 3)
    assert_type(rank, Tensor[2, 3])


# ==== torch.tensordot ====


def test_tensordot_simple():
    """Tensordot with simple contraction"""
    a: Tensor[3, 4, 5] = torch.randn(3, 4, 5)
    b: Tensor[5, 6, 7] = torch.randn(5, 6, 7)
    c = torch.tensordot(a, b, dims=1)
    # Contract last 1 dim of a with first 1 dim of b: (3, 4) + (6, 7) = (3, 4, 6, 7)
    assert_type(c, Tensor[3, 4, 6, 7])


def test_tensordot_multiple_dims():
    """Tensordot with multiple contraction dimensions"""
    a: Tensor[2, 3, 4, 5] = torch.randn(2, 3, 4, 5)
    b: Tensor[4, 5, 6, 7] = torch.randn(4, 5, 6, 7)
    c = torch.tensordot(a, b, dims=2)
    # Contract last 2 dims of a with first 2 dims of b: (2, 3) + (6, 7) = (2, 3, 6, 7)
    assert_type(c, Tensor[2, 3, 6, 7])


# ==== torch.einsum ====


def test_einsum_simple():
    """Einsum (returns Unknown shape for now)"""
    a: Tensor[3, 4] = torch.randn(3, 4)
    b: Tensor[4, 5] = torch.randn(4, 5)
    # Einsum is complex, currently returns Unknown - just check it type checks
    _ = torch.einsum("ij,jk->ik", a, b)


# ==== Tier 3: Eigenvalues Only ====


def test_eigvals_2d():
    """Eigenvalues only (no eigenvectors)"""
    A: Tensor[4, 4] = torch.randn(4, 4)
    eigenvalues = torch.linalg.eigvals(A)
    # Returns only eigenvalues: (4,)
    assert_type(eigenvalues, Tensor[4])


def test_eigvals_batched():
    """Eigenvalues with batch dimensions"""
    A: Tensor[2, 3, 5, 5] = torch.randn(2, 3, 5, 5)
    eigenvalues = torch.linalg.eigvals(A)
    # Returns eigenvalues: (2, 3, 5)
    assert_type(eigenvalues, Tensor[2, 3, 5])


def test_eigvalsh():
    """Hermitian eigenvalues only"""
    A: Tensor[4, 4] = torch.randn(4, 4)
    eigenvalues = torch.linalg.eigvalsh(A)
    # Returns only eigenvalues: (4,)
    assert_type(eigenvalues, Tensor[4])
