# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Literal, overload

from numpy._shapes import svd_reduced_2d_ir
from shape_extensions import Dim, uses_shape_dsl

from .. import ndarray

# MVP shape surface only; NumPy dtype promotion is intentionally not modeled.
@overload
def solve[N, DType](
    a: ndarray[tuple[Dim[N], Dim[N]], DType],
    b: ndarray[tuple[Dim[N]]],
) -> ndarray[tuple[Dim[N]], DType]: ...
@overload
def solve[N, K, DType](
    a: ndarray[tuple[Dim[N], Dim[N]], DType],
    b: ndarray[tuple[Dim[N], Dim[K]]],
) -> ndarray[tuple[Dim[N], Dim[K]], DType]: ...
def norm[N, M, DType](
    x: ndarray[tuple[Dim[N], Dim[M], Dim[3]], DType],
    axis: Literal[-1],
    keepdims: Literal[True],
) -> ndarray[tuple[Dim[N], Dim[M], Dim[1]], DType]: ...
def eigh[N, DType](
    a: ndarray[tuple[Dim[N], Dim[N]], DType],
) -> tuple[ndarray[tuple[Dim[N]], DType], ndarray[tuple[Dim[N], Dim[N]], DType]]: ...
@overload
def svd[N, DType](
    a: ndarray[tuple[Dim[N], Dim[N]], DType],
    # NumPy defaults to full SVD; this MVP accepts only the reduced form needed
    # by PCA-style demos.
    full_matrices: Literal[False],
    compute_uv: Literal[True] = True,
    hermitian: Literal[False] = False,
) -> tuple[
    ndarray[tuple[Dim[N], Dim[N]], DType],
    ndarray[tuple[Dim[N]], DType],
    ndarray[tuple[Dim[N], Dim[N]], DType],
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
