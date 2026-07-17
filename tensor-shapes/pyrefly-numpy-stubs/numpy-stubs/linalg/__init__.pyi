# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Literal, overload

from numpy._shapes import int_min

# Preserve NumPy's canonical re-exports before local shape-aware declarations.
from numpy.linalg._linalg import (
    cholesky as cholesky,
    cond as cond,
    cross as cross,
    det as det,
    diagonal as diagonal,
    eig as eig,
    eigvals as eigvals,
    eigvalsh as eigvalsh,
    inv as inv,
    lstsq as lstsq,
    matmul as matmul,
    matrix_norm as matrix_norm,
    matrix_power as matrix_power,
    matrix_rank as matrix_rank,
    matrix_transpose as matrix_transpose,
    multi_dot as multi_dot,
    outer as outer,
    pinv as pinv,
    qr as qr,
    slogdet as slogdet,
    svdvals as svdvals,
    tensordot as tensordot,
    tensorinv as tensorinv,
    tensorsolve as tensorsolve,
    trace as trace,
    vecdot as vecdot,
    vector_norm as vector_norm,
)
from shape_extensions import Int, IntVar

from .. import ndarray

class LinAlgError(ValueError): ...

# MVP shape surface only; NumPy dtype promotion is intentionally not modeled.
@overload
def solve[N: IntVar, DType](
    a: ndarray[[N, N], DType],
    b: ndarray[[N]],
) -> ndarray[[N], DType]: ...
@overload
def solve[N: IntVar, K: IntVar, DType](
    a: ndarray[[N, N], DType],
    b: ndarray[[N, K]],
) -> ndarray[[N, K], DType]: ...
@overload
def norm[N: IntVar, M: IntVar, DType](
    x: ndarray[[N, M, 3], DType],
    ord: None,
    axis: Literal[-1],
    keepdims: Literal[True],
) -> ndarray[[N, M, 1], DType]: ...
@overload
def norm[N: IntVar, M: IntVar, DType](
    x: ndarray[[N, M, 3], DType],
    ord: None = None,
    *,
    axis: Literal[-1],
    keepdims: Literal[True],
) -> ndarray[[N, M, 1], DType]: ...
@overload
def norm(
    x: Any,
    ord: Any = None,
    axis: Any = None,
    keepdims: bool = False,
    **kwargs: Any,
) -> Any: ...
def eigh[N: IntVar, DType](
    a: ndarray[[N, N], DType],
) -> tuple[ndarray[[N], DType], ndarray[[N, N], DType]]: ...
def svd[M: IntVar, N: IntVar, DType](
    a: ndarray[[M, N], DType],
    # NumPy defaults to full SVD; this MVP accepts only the reduced form needed
    # by PCA-style demos.
    full_matrices: Literal[False],
    compute_uv: Literal[True] = True,
    hermitian: Literal[False] = False,
) -> tuple[
    ndarray[[M, int_min(Int[M], Int[N])], DType],
    ndarray[[int_min(Int[M], Int[N])], DType],
    ndarray[[int_min(Int[M], Int[N]), N], DType],
]: ...
