# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Literal, overload

from numpy._shapes import svd_reduced_2d_ir
from shape_extensions import SymVar, uses_shape_dsl

from .. import ndarray

# MVP shape surface only; NumPy dtype promotion is intentionally not modeled.
@overload
def solve[N: SymVar, DType](
    a: ndarray[[N, N], DType],
    b: ndarray[[N]],
) -> ndarray[[N], DType]: ...
@overload
def solve[N: SymVar, K: SymVar, DType](
    a: ndarray[[N, N], DType],
    b: ndarray[[N, K]],
) -> ndarray[[N, K], DType]: ...
def norm[N: SymVar, M: SymVar, DType](
    x: ndarray[[N, M, 3], DType],
    axis: Literal[-1],
    keepdims: Literal[True],
) -> ndarray[[N, M, 1], DType]: ...
def eigh[N: SymVar, DType](
    a: ndarray[[N, N], DType],
) -> tuple[ndarray[[N], DType], ndarray[[N, N], DType]]: ...
@overload
def svd[N: SymVar, DType](
    a: ndarray[[N, N], DType],
    # NumPy defaults to full SVD; this MVP accepts only the reduced form needed
    # by PCA-style demos.
    full_matrices: Literal[False],
    compute_uv: Literal[True] = True,
    hermitian: Literal[False] = False,
) -> tuple[
    ndarray[[N, N], DType],
    ndarray[[N], DType],
    ndarray[[N, N], DType],
]: ...
@uses_shape_dsl(svd_reduced_2d_ir)
@overload
def svd(
    a: ndarray,
    # NumPy defaults to full SVD; this MVP accepts only the reduced form needed
    # by PCA-style demos.
    full_matrices: Literal[False],
    compute_uv: Literal[True] = True,
    hermitian: Literal[False] = False,
) -> tuple[ndarray, ndarray, ndarray]: ...
