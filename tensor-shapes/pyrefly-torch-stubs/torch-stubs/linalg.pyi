# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Type stubs for torch.linalg module (Phase 4: Advanced Linear Algebra)
from collections.abc import Sequence
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
    tensorinv_shape,
    tensorsolve_moved_shape,
    tensorsolve_shape,
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

common_notes: dict[str, str]

@overload
def lstsq[Batch: IntTuple, M: IntVar, N: IntVar, K: IntVar](
    A: Tensor[[*Batch, M, N]],
    B: Tensor[[*Batch, M, K]],
    rcond: float | None = None,
    *,
    driver: str | None = None,
) -> return_types.linalg_lstsq[[*Batch, N, K], IntTuple, IntTuple, IntTuple]: ...
@overload
def lstsq[Batch: IntTuple, M: IntVar, N: IntVar](
    A: Tensor[[*Batch, M, N]],
    B: Tensor[[*Batch, M]],
    rcond: float | None = None,
    *,
    driver: str | None = None,
) -> return_types.linalg_lstsq[[*Batch, N], IntTuple, IntTuple, IntTuple]: ...
@overload
def lstsq(
    A: Tensor, B: Tensor, rcond: float | None = None, *, driver: str | None = None
) -> return_types.linalg_lstsq[IntTuple, IntTuple, IntTuple, IntTuple]: ...
@overload
def lu[Batch: IntTuple, N: IntVar](
    A: Tensor[[*Batch, N, N]],
    *,
    pivot: bool = True,
    out: tuple[Tensor, Tensor, Tensor] | None = None,
) -> return_types.linalg_lu[[*Batch, N, N], [*Batch, N, N], [*Batch, N, N]]: ...
@overload
def lu[Batch: IntTuple, M: IntVar, N: IntVar](
    A: Tensor[[*Batch, M, N]],
    *,
    pivot: bool = True,
    out: tuple[Tensor, Tensor, Tensor] | None = None,
) -> return_types.linalg_lu[[*Batch, M, M], [*Batch, M, int], [*Batch, int, N]]: ...
@overload
def qr[Batch: IntTuple, N: IntVar](
    A: Tensor[[*Batch, N, N]],
    mode: Literal["reduced", "complete"] = "reduced",
    *,
    out: tuple[Tensor, Tensor] | None = None,
) -> return_types.linalg_qr[[*Batch, N, N], [*Batch, N, N]]: ...
@overload
def qr[Batch: IntTuple, M: IntVar, N: IntVar](
    A: Tensor[[*Batch, M, N]],
    mode: Literal["complete"],
    *,
    out: tuple[Tensor, Tensor] | None = None,
) -> return_types.linalg_qr[[*Batch, M, M], [*Batch, M, N]]: ...
@overload
def qr[Batch: IntTuple, M: IntVar, N: IntVar](
    A: Tensor[[*Batch, M, N]],
    mode: Literal["r"],
    *,
    out: tuple[Tensor, Tensor] | None = None,
) -> return_types.linalg_qr[[0], [*Batch, int, N]]: ...
@overload
def qr[Batch: IntTuple, M: IntVar, N: IntVar](
    A: Tensor[[*Batch, M, N]],
    mode: Literal["reduced"] = "reduced",
    *,
    out: tuple[Tensor, Tensor] | None = None,
) -> return_types.linalg_qr[[*Batch, M, int], [*Batch, int, N]]: ...
@overload
def qr(
    A: Tensor, mode: str, *, out: tuple[Tensor, Tensor] | None = None
) -> return_types.linalg_qr[IntTuple, IntTuple]: ...
@overload
def svd[Batch: IntTuple, N: IntVar](
    A: Tensor[[*Batch, N, N]],
    full_matrices: bool = True,
    *,
    driver: str | None = None,
    out: tuple[Tensor, Tensor, Tensor] | None = None,
) -> return_types.linalg_svd[[*Batch, N, N], [*Batch, N], [*Batch, N, N]]: ...
@overload
def svd[Batch: IntTuple, M: IntVar, N: IntVar](
    A: Tensor[[*Batch, M, N]],
    full_matrices: Literal[True] = True,
    *,
    driver: str | None = None,
    out: tuple[Tensor, Tensor, Tensor] | None = None,
) -> return_types.linalg_svd[[*Batch, M, M], [*Batch, int], [*Batch, N, N]]: ...
@overload
def svd[Batch: IntTuple, M: IntVar, N: IntVar](
    A: Tensor[[*Batch, M, N]],
    full_matrices: Literal[False],
    *,
    driver: str | None = None,
    out: tuple[Tensor, Tensor, Tensor] | None = None,
) -> return_types.linalg_svd[[*Batch, M, int], [*Batch, int], [*Batch, int, N]]: ...
@overload
def svd(
    A: Tensor,
    full_matrices: bool,
    *,
    driver: str | None = None,
    out: tuple[Tensor, Tensor, Tensor] | None = None,
) -> return_types.linalg_svd[IntTuple, IntTuple, IntTuple]: ...
@overload
def ldl_factor[Batch: IntTuple, N: IntVar](
    A: Tensor[[*Batch, N, N]],
    *,
    hermitian: bool = False,
    out: tuple[Tensor, Tensor] | None = None,
) -> return_types.linalg_ldl_factor[[*Batch, N, N], [*Batch, N]]: ...
@overload
def ldl_factor[Shape: IntTuple](
    A: Tensor[Shape],
    *,
    hermitian: bool = False,
    out: tuple[Tensor, Tensor] | None = None,
) -> return_types.linalg_ldl_factor[Shape, eig_shape(Shape)]: ...
@overload
def ldl_factor_ex[Batch: IntTuple, N: IntVar](
    A: Tensor[[*Batch, N, N]],
    *,
    hermitian: bool = False,
    check_errors: bool = False,
    out: tuple[Tensor, Tensor, Tensor] | None = None,
) -> return_types.linalg_ldl_factor_ex[[*Batch, N, N], [*Batch, N], Batch]: ...
@overload
def ldl_factor_ex[Shape: IntTuple](
    A: Tensor[Shape],
    *,
    hermitian: bool = False,
    check_errors: bool = False,
    out: tuple[Tensor, Tensor, Tensor] | None = None,
) -> return_types.linalg_ldl_factor_ex[
    Shape, eig_shape(Shape), slogdet_shape(Shape)
]: ...
@overload
def lu_factor[Batch: IntTuple, N: IntVar](
    A: Tensor[[*Batch, N, N]],
    *,
    pivot: bool = True,
    out: tuple[Tensor, Tensor] | None = None,
) -> return_types.linalg_lu_factor[[*Batch, N, N], [*Batch, N]]: ...
@overload
def lu_factor[Batch: IntTuple, M: IntVar, N: IntVar](
    A: Tensor[[*Batch, M, N]],
    *,
    pivot: bool = True,
    out: tuple[Tensor, Tensor] | None = None,
) -> return_types.linalg_lu_factor[[*Batch, M, N], [*Batch, int]]: ...
@overload
def lu_factor_ex[Batch: IntTuple, N: IntVar](
    A: Tensor[[*Batch, N, N]],
    *,
    pivot: bool = True,
    check_errors: bool = False,
    out: tuple[Tensor, Tensor, Tensor] | None = None,
) -> return_types.linalg_lu_factor_ex[[*Batch, N, N], [*Batch, N], Batch]: ...
@overload
def lu_factor_ex[Batch: IntTuple, M: IntVar, N: IntVar](
    A: Tensor[[*Batch, M, N]],
    *,
    pivot: bool = True,
    check_errors: bool = False,
    out: tuple[Tensor, Tensor, Tensor] | None = None,
) -> return_types.linalg_lu_factor_ex[[*Batch, M, N], [*Batch, int], Batch]: ...
@overload
def cholesky_ex[Batch: IntTuple, N: IntVar](
    A: Tensor[[*Batch, N, N]],
    *,
    upper: bool = False,
    check_errors: bool = False,
    out: tuple[Tensor, Tensor] | None = None,
) -> return_types.linalg_cholesky_ex[[*Batch, N, N], Batch]: ...
@overload
def cholesky_ex[Shape: IntTuple](
    A: Tensor[Shape],
    *,
    upper: bool = False,
    check_errors: bool = False,
    out: tuple[Tensor, Tensor] | None = None,
) -> return_types.linalg_cholesky_ex[Shape, slogdet_shape(Shape)]: ...
@overload
def inv_ex[Batch: IntTuple, N: IntVar](
    A: Tensor[[*Batch, N, N]],
    *,
    check_errors: bool = False,
    out: tuple[Tensor, Tensor] | None = None,
) -> return_types.linalg_inv_ex[[*Batch, N, N], Batch]: ...
@overload
def inv_ex[Shape: IntTuple](
    A: Tensor[Shape],
    *,
    check_errors: bool = False,
    out: tuple[Tensor, Tensor] | None = None,
) -> return_types.linalg_inv_ex[Shape, slogdet_shape(Shape)]: ...
@overload
def solve_ex[Batch: IntTuple, N: IntVar](
    A: Tensor[[*Batch, N, N]],
    B: Tensor[[*Batch, N]],
    *,
    left: Literal[True] = True,
    check_errors: bool = False,
    out: tuple[Tensor, Tensor] | None = None,
) -> return_types.linalg_solve_ex[[*Batch, N], Batch]: ...
@overload
def solve_ex[Batch: IntTuple, N: IntVar, K: IntVar](
    A: Tensor[[*Batch, N, N]],
    B: Tensor[[*Batch, N, K]],
    *,
    left: Literal[True] = True,
    check_errors: bool = False,
    out: tuple[Tensor, Tensor] | None = None,
) -> return_types.linalg_solve_ex[[*Batch, N, K], Batch]: ...
@overload
def solve_ex[Shape: IntTuple](
    A: Tensor[Shape],
    B: Tensor,
    *,
    left: bool = True,
    check_errors: bool = False,
    out: tuple[Tensor, Tensor] | None = None,
) -> return_types.linalg_solve_ex[IntTuple, slogdet_shape(Shape)]: ...
@overload
def multi_dot[M: IntVar, N: IntVar, K: IntVar](
    tensors: tuple[Tensor[[M, N]], Tensor[[N, K]]], *, out: Tensor | None = None
) -> Tensor[[M, K]]: ...
@overload
def multi_dot[M: IntVar, N: IntVar, K: IntVar, P: IntVar](
    tensors: tuple[Tensor[[M, N]], Tensor[[N, K]], Tensor[[K, P]]],
    *,
    out: Tensor | None = None,
) -> Tensor[[M, P]]: ...

# TODO(stroxler): Validate longer matrix chains once the IntTuples DSL retains
# exact shape inference while checking adjacent contracting dimensions.
@overload
def multi_dot(tensors: Sequence[Tensor], *, out: Tensor | None = None) -> Tensor: ...
def tensorinv[Shape: IntTuple, Ind: Flag[int] = 2](
    A: Tensor[Shape], ind: Ind = 2, *, out: Tensor | None = None
) -> Tensor[tensorinv_shape(Shape, Ind)]: ...
@overload
def tensorsolve[A: IntTuple, B: IntTuple](
    A: Tensor[A],
    B: Tensor[B],
    dims: None = None,
    *,
    out: Tensor | None = None,
) -> Tensor[tensorsolve_shape(A, B)]: ...
@overload
def tensorsolve[A: IntTuple, B: IntTuple, Dims: Flag[tuple[int, ...]]](
    A: Tensor[A],
    B: Tensor[B],
    dims: Dims,
    *,
    out: Tensor | None = None,
) -> Tensor[tensorsolve_moved_shape(A, B, Dims)]: ...
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
