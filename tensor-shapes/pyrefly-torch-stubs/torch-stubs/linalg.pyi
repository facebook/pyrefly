# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Type stubs for torch.linalg module (Phase 4: Advanced Linear Algebra)
from typing import Any, overload

from shape_extensions import Elements, IntTuple, IntVar
from torch import return_types, Tensor
from torch._shapes import eig_shape, eigvals_shape, slogdet_shape

# Eigenvalue decomposition
@overload
def eig[Batch: IntTuple, M: IntVar, N: IntVar](
    self: Tensor[[*Elements[Batch], M, N]],
) -> tuple[Tensor[[*Elements[Batch], M]], Tensor[[*Elements[Batch], M, N]]]: ...
@overload
def eig[Shape: IntTuple](
    self: Tensor[Shape],
) -> tuple[Tensor[eig_shape(Shape)], Tensor[Shape]]: ...
@overload
def eigh[Batch: IntTuple, M: IntVar, N: IntVar](
    self: Tensor[[*Elements[Batch], M, N]], UPLO: str = "L"
) -> tuple[Tensor[[*Elements[Batch], M]], Tensor[[*Elements[Batch], M, N]]]: ...
@overload
def eigh[Shape: IntTuple](
    self: Tensor[Shape], UPLO: str = "L"
) -> tuple[Tensor[eig_shape(Shape)], Tensor[Shape]]: ...

# Tier 3: Eigenvalues only (no eigenvectors)
@overload
def eigvals[Batch: IntTuple, M: IntVar, N: IntVar](
    self: Tensor[[*Elements[Batch], M, N]],
) -> Tensor[[*Elements[Batch], M]]: ...
@overload
def eigvals[Shape: IntTuple](self: Tensor[Shape]) -> Tensor[eigvals_shape(Shape)]: ...
@overload
def eigvalsh[Batch: IntTuple, M: IntVar, N: IntVar](
    self: Tensor[[*Elements[Batch], M, N]], UPLO: str = "L"
) -> Tensor[[*Elements[Batch], M]]: ...
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
    input: Tensor[[*Elements[Batch], M, N]],
) -> Tensor[Batch]: ...

# Sign and log determinant
@overload
def slogdet[Batch: IntTuple, M: IntVar, N: IntVar](
    self: Tensor[[*Elements[Batch], M, N]],
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
    input: Tensor[[*Elements[Batch], M, N]], tol: float = None, hermitian: bool = False
) -> Tensor[Batch]: ...

# Vector/matrix norm
def norm(
    A: Tensor,
    ord: int | float | str | None = None,
    dim: int | tuple[int, ...] | None = None,
    keepdim: bool = False,
) -> Tensor: ...
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
