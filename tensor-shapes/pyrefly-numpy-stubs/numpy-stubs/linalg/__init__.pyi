# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Literal, overload

from numpy._shapes import int_min
from shape_extensions import IntVar

from .. import ndarray

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
def norm[N: IntVar, M: IntVar, DType](
    x: ndarray[[N, M, 3], DType],
    axis: Literal[-1],
    keepdims: Literal[True],
) -> ndarray[[N, M, 1], DType]: ...
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
    ndarray[[M, int_min(M, N)], DType],
    ndarray[[int_min(M, N)], DType],
    ndarray[[int_min(M, N), N], DType],
]: ...
