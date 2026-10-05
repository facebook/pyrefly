# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Type stubs for torch.linalg module (Phase 4: Advanced Linear Algebra)
from typing import Any, Literal, overload

from shape_extensions import Flag, gufunc_broadcast, Int as _Int, IntTuple, IntVar
from torch import dtype, return_types, Tensor
from torch._C import _LinAlgError as LinAlgError
from torch._shapes import (
    cross_shape,
    diagonal_shape,
    eig_shape,
    eigvals_shape,
    lu_solve_shape,
    matmul_shape,
    reduce_shape,
    slogdet_shape,
    transpose_shape,
    vecdot_shape,
)

# Eigenvalue decomposition
@overload
def eig[Batch: IntTuple, M: IntVar, N: IntVar](
    self: Tensor[[*Batch, M, N]],
) -> tuple[Tensor[[*Batch, M]], Tensor[[*Batch, M, N]]]: ...
@overload
def eig[Shape: IntTuple](
    self: Tensor[Shape],
) -> tuple[Tensor[eig_shape(Shape)], Tensor[Shape]]: ...
@overload
def eigh[Batch: IntTuple, M: IntVar, N: IntVar](
    self: Tensor[[*Batch, M, N]], UPLO: str = "L"
) -> tuple[Tensor[[*Batch, M]], Tensor[[*Batch, M, N]]]: ...
@overload
def eigh[Shape: IntTuple](
    self: Tensor[Shape], UPLO: str = "L"
) -> tuple[Tensor[eig_shape(Shape)], Tensor[Shape]]: ...

# Tier 3: Eigenvalues only (no eigenvectors)
@overload
def eigvals[Batch: IntTuple, M: IntVar, N: IntVar](
    self: Tensor[[*Batch, M, N]],
) -> Tensor[[*Batch, M]]: ...
@overload
def eigvals[Shape: IntTuple](self: Tensor[Shape]) -> Tensor[eigvals_shape(Shape)]: ...
@overload
def eigvalsh[Batch: IntTuple, M: IntVar, N: IntVar](
    self: Tensor[[*Batch, M, N]], UPLO: str = "L"
) -> Tensor[[*Batch, M]]: ...
@overload
def eigvalsh[Shape: IntTuple](
    self: Tensor[Shape], UPLO: str = "L"
) -> Tensor[eigvals_shape(Shape)]: ...

# Cholesky decomposition
def cholesky[Shape: IntTuple](
    input: Tensor[Shape], upper: bool = False
) -> Tensor[Shape]: ...

# Linear system solvers
def solve[Shape: IntTuple, OtherShape: IntTuple](
    self: Tensor[Shape], other: Tensor[OtherShape]
) -> Tensor[OtherShape]: ...
def solve_triangular[Shape: IntTuple, OtherShape: IntTuple](
    self: Tensor[Shape], other: Tensor[OtherShape], upper: bool = False
) -> Tensor[OtherShape]: ...
def cholesky_solve[Shape: IntTuple, OtherShape: IntTuple](
    self: Tensor[Shape], other: Tensor[OtherShape], upper: bool = False
) -> Tensor[Shape]: ...

# Matrix inverse
def inv[Shape: IntTuple](input: Tensor[Shape]) -> Tensor[Shape]: ...

# Determinant
def det[Batch: IntTuple, M: IntVar, N: IntVar](
    input: Tensor[[*Batch, M, N]],
) -> Tensor[Batch]: ...

# Sign and log determinant
@overload
def slogdet[Batch: IntTuple, M: IntVar, N: IntVar](
    self: Tensor[[*Batch, M, N]],
) -> return_types.linalg_slogdet[Batch]: ...
@overload
def slogdet[Shape: IntTuple](
    self: Tensor[Shape],
) -> return_types.linalg_slogdet[slogdet_shape(Shape)]: ...

# Matrix power
def matrix_power[Shape: IntTuple](input: Tensor[Shape], n: int) -> Tensor[Shape]: ...

# Matrix exponential
def matrix_exp[Shape: IntTuple](input: Tensor[Shape]) -> Tensor[Shape]: ...

# Matrix rank
def matrix_rank[Batch: IntTuple, M: IntVar, N: IntVar](
    input: Tensor[[*Batch, M, N]], tol: float = None, hermitian: bool = False
) -> Tensor[Batch]: ...

# TODO: Add precise types and signatures for the remaining public API.
cholesky_ex: Any
common_notes: Any
inv_ex: Any
ldl_factor: Any
ldl_factor_ex: Any
lstsq: Any
lu: Any
lu_factor: Any
lu_factor_ex: Any
multi_dot: Any
qr: Any
solve_ex: Any
svd: Any
tensorinv: Any
tensorsolve: Any

def ldl_solve[LD: IntTuple, Pivots: IntTuple, RHS: IntTuple](
    LD: Tensor[LD],
    pivots: Tensor[Pivots],
    B: Tensor[RHS],
    *,
    hermitian: bool = False,
    out: Tensor | None = None,
) -> Tensor[gufunc_broadcast("(n,n),(n),(n,k)->(n,k)", tuple[LD, Pivots, RHS])]: ...
def lu_solve[LU: IntTuple, Pivots: IntTuple, RHS: IntTuple, Left: Flag[bool] = True](
    LU: Tensor[LU],
    pivots: Tensor[Pivots],
    B: Tensor[RHS],
    *,
    left: Left = True,
    adjoint: bool = False,
    out: Tensor | None = None,
) -> Tensor[lu_solve_shape(LU, Pivots, RHS, Left)]: ...
@overload
def svdvals[Batch: IntTuple, N: IntVar](
    A: Tensor[[*Batch, N, N]], *, driver: str | None = None, out: Tensor | None = None
) -> Tensor[[*Batch, N]]: ...
@overload
def svdvals[Batch: IntTuple, M: IntVar, N: IntVar](
    A: Tensor[[*Batch, M, N]], *, driver: str | None = None, out: Tensor | None = None
) -> Tensor[[*Batch, int]]: ...
@overload
def cond[Batch: IntTuple, M: IntVar, N: IntVar](
    A: Tensor[[*Batch, M, N]],
    p: int | float | str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor[Batch]: ...
@overload
def cond[Shape: IntTuple](
    A: Tensor[Shape], p: int | float | str | None = None, *, out: Tensor | None = None
) -> Tensor[slogdet_shape(Shape)]: ...
def cross[InputShape: IntTuple, OtherShape: IntTuple, Dim: Flag[int] = -1](
    input: Tensor[InputShape],
    other: Tensor[OtherShape],
    *,
    dim: Dim = -1,
    out: Tensor | None = None,
) -> Tensor[cross_shape(InputShape, OtherShape, Dim)]: ...
def vecdot[InputShape: IntTuple, OtherShape: IntTuple, Dim: Flag[int] = -1](
    x: Tensor[InputShape],
    y: Tensor[OtherShape],
    *,
    dim: Dim = -1,
    out: Tensor | None = None,
) -> Tensor[vecdot_shape(InputShape, OtherShape, Dim)]: ...
def diagonal[
    Shape: IntTuple,
    Offset: Flag[int] = 0,
    Dim1: Flag[int] = -2,
    Dim2: Flag[int] = -1,
](
    A: Tensor[Shape], *, offset: Offset = 0, dim1: Dim1 = -2, dim2: Dim2 = -1
) -> Tensor[diagonal_shape(Shape, Offset, Dim1, Dim2)]: ...
def householder_product[Batch: IntTuple, M: IntVar, N: IntVar, K: IntVar](
    A: Tensor[[*Batch, M, N]], tau: Tensor[[*Batch, K]], *, out: Tensor | None = None
) -> Tensor[[*Batch, M, N]]: ...
def matmul[Left: IntTuple, Right: IntTuple](
    input: Tensor[Left], other: Tensor[Right], *, out: Tensor | None = None
) -> Tensor[matmul_shape(Left, Right)]: ...
@overload
def matrix_norm[Batch: IntTuple, M: IntVar, N: IntVar](
    A: Tensor[[*Batch, M, N]],
    ord: int | float | str = "fro",
    *,
    dtype: dtype | None = None,
    out: Tensor | None = None,
) -> Tensor[Batch]: ...
@overload
def matrix_norm[
    Shape: IntTuple,
    Dim: Flag[tuple[int, int]] = (-2, -1),
    Keepdim: Flag[bool] = False,
](
    A: Tensor[Shape],
    ord: int | float | str = "fro",
    dim: Dim = (-2, -1),
    keepdim: Keepdim = False,
    *,
    dtype: dtype | None = None,
    out: Tensor | None = None,
) -> Tensor[reduce_shape(Shape, Dim, Keepdim)]: ...
@overload
def pinv[Batch: IntTuple, M: IntVar, N: IntVar](
    A: Tensor[[*Batch, M, N]],
    *,
    atol: float | Tensor | None = None,
    rtol: float | Tensor | None = None,
    hermitian: bool = False,
    out: Tensor | None = None,
) -> Tensor[[*Batch, N, M]]: ...
@overload
def pinv[Shape: IntTuple](
    A: Tensor[Shape],
    *,
    atol: float | Tensor | None = None,
    rtol: float | Tensor | None = None,
    hermitian: bool = False,
    out: Tensor | None = None,
) -> Tensor[transpose_shape(Shape, -2, -1)]: ...
@overload
def vander[Batch: IntTuple, N: IntVar](
    x: Tensor[[*Batch, N]], N: None = None
) -> Tensor[[*Batch, N, N]]: ...
@overload
def vander[Batch: IntTuple, N: IntVar, Columns: IntVar](
    x: Tensor[[*Batch, N]], N: _Int[Columns]
) -> Tensor[[*Batch, N, Columns]]: ...

# Vector/matrix norm
def norm[Shape: IntTuple, Dim: Flag[int | tuple[int, ...] | None], Keepdim: Flag[bool]](
    A: Tensor[Shape],
    ord: int | float | str | None = None,
    dim: Dim = None,
    keepdim: Keepdim = False,
) -> Tensor[reduce_shape(Shape, Dim, Keepdim)]: ...
def vector_norm(
    x: Tensor,
    ord: int | float = 2,
    dim: int | tuple[int, ...] | None = None,
    keepdim: bool = False,
    *,
    dtype: Any = None,
    out: Tensor | None = None,
) -> Tensor: ...
def __getattr__(name: str) -> Any: ...
