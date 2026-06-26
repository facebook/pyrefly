# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Type stubs for torch.linalg module (Phase 4: Advanced Linear Algebra)
from shape_extensions import uses_shape_dsl
from torch import Tensor
from torch._shapes import eig_ir, eigvals_ir, slogdet_ir, solve_ir, solve_reversed_ir

# Eigenvalue decomposition
@uses_shape_dsl(eig_ir)
def eig(self: Tensor) -> tuple[Tensor, Tensor]: ...
@uses_shape_dsl(eig_ir)
def eigh(self: Tensor, UPLO: str = "L") -> tuple[Tensor, Tensor]: ...

# Tier 3: Eigenvalues only (no eigenvectors)
@uses_shape_dsl(eigvals_ir)
def eigvals(self: Tensor) -> Tensor: ...
@uses_shape_dsl(eigvals_ir)
def eigvalsh(self: Tensor, UPLO: str = "L") -> Tensor: ...

# Cholesky decomposition
def cholesky[*Shape](input: Tensor[*Shape], upper: bool = False) -> Tensor[*Shape]: ...

# Linear system solvers
@uses_shape_dsl(solve_ir)
def solve(self: Tensor, other: Tensor) -> Tensor: ...
@uses_shape_dsl(solve_ir)
def solve_triangular(self: Tensor, other: Tensor, upper: bool = False) -> Tensor: ...
@uses_shape_dsl(solve_reversed_ir)
def cholesky_solve(self: Tensor, other: Tensor, upper: bool = False) -> Tensor: ...

# Matrix inverse
def inv[*Shape](input: Tensor[*Shape]) -> Tensor[*Shape]: ...

# Determinant
def det[*Batch, M, N](input: Tensor[*Batch, M, N]) -> Tensor[*Batch]: ...

# Sign and log determinant
@uses_shape_dsl(slogdet_ir)
def slogdet(self: Tensor) -> tuple[Tensor, Tensor]: ...

# Matrix power
def matrix_power[*Shape](input: Tensor[*Shape], n: int) -> Tensor[*Shape]: ...

# Matrix exponential
def matrix_exp[*Shape](input: Tensor[*Shape]) -> Tensor[*Shape]: ...

# Matrix rank
def matrix_rank[*Batch, M, N](
    input: Tensor[*Batch, M, N], tol: float = None, hermitian: bool = False
) -> Tensor[*Batch]: ...
