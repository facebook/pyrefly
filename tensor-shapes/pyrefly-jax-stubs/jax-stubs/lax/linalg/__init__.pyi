# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import enum
from typing import Any, Literal, overload

from jax._array import Array
from jax._shapes import (
    cholesky_update_shape,
    hessenberg_taus_shape,
    householder_product_shape,
    int_min,
    ormqr_shape,
    symmetric_product_shape,
    triangular_solve_shape,
    tridiagonal_d_shape,
    tridiagonal_diag_minus_one_shape,
    tridiagonal_solve_shape,
)
from shape_extensions import Elements, Flag, Int, IntTuple, IntVar

type _Shape = IntTuple

class EigImplementation(enum.Enum): ...
class EighImplementation(enum.Enum): ...
class SvdAlgorithm(enum.Enum): ...

def cholesky[Batch: IntTuple, N: IntVar](
    x: Array[[*Elements[Batch], N, N]],
    *,
    symmetrize_input: bool = True,
) -> Array[[*Elements[Batch], N, N]]: ...
def cholesky_update[RShape: _Shape, WShape: _Shape](
    r_matrix: Array[RShape],
    w_vector: Array[WShape],
) -> Array[cholesky_update_shape(RShape, WShape)]: ...
def eig[Batch: IntTuple, N: IntVar](
    x: Array[[*Elements[Batch], N, N]],
    *,
    compute_left_eigenvectors: bool = True,
    compute_right_eigenvectors: bool = True,
    enable_eigvec_derivs: bool = False,
    implementation: EigImplementation | str | None = None,
    use_magma: bool | None = None,
) -> list[Array[Any]]: ...
def eigh[Batch: IntTuple, N: IntVar](
    x: Array[[*Elements[Batch], N, N]],
    *,
    lower: bool = True,
    symmetrize_input: bool = True,
    sort_eigenvalues: bool = True,
    subset_by_index: tuple[int, int] | None = None,
    implementation: EighImplementation | str | None = None,
) -> tuple[Array[[*Elements[Batch], N, N]], Array[[*Elements[Batch], N]]]: ...
def hessenberg[AShape: _Shape](
    a: Array[AShape],
) -> tuple[Array[AShape], Array[hessenberg_taus_shape(AShape)]]: ...
def householder_product[AShape: _Shape, TShape: _Shape](
    a: Array[AShape],
    taus: Array[TShape],
) -> Array[householder_product_shape(AShape, TShape)]: ...
def lu[Batch: IntTuple, M: IntVar, N: IntVar](
    x: Array[[*Elements[Batch], M, N]],
) -> tuple[
    Array[[*Elements[Batch], M, N]],
    Array[[*Elements[Batch], int_min(Int[M], Int[N])]],
    Array[[*Elements[Batch], int_min(Int[M], Int[N])]],
]: ...
def lu_pivots_to_permutation[Batch: IntTuple, K: IntVar, N: IntVar](
    pivots: Array[[*Elements[Batch], K]],
    permutation_size: Int[N],
) -> Array[[*Elements[Batch], N]]: ...
def ormqr[AShape: _Shape, TShape: _Shape, CShape: _Shape](
    a: Array[AShape],
    taus: Array[TShape],
    c: Array[CShape],
    *,
    left: bool = True,
    transpose: bool = False,
) -> Array[ormqr_shape(AShape, TShape, CShape)]: ...
def qdwh[Batch: IntTuple, M: IntVar, N: IntVar](
    x: Array[[*Elements[Batch], M, N]],
    *,
    is_hermitian: bool = False,
    max_iterations: int | None = None,
    eps: float | None = None,
    dynamic_shape: tuple[int, int] | None = None,
) -> tuple[
    Array[[*Elements[Batch], M, N]],
    Array[[*Elements[Batch], N, N]],
    Array[[]],
    Array[[]],
]: ...
@overload
def qr[Batch: IntTuple, M: IntVar, N: IntVar](
    x: Array[[*Elements[Batch], M, N]],
    *,
    pivoting: Literal[False] = False,
    full_matrices: Literal[True] = True,
    use_magma: bool | None = None,
) -> tuple[Array[[*Elements[Batch], M, M]], Array[[*Elements[Batch], M, N]]]: ...
@overload
def qr[Batch: IntTuple, M: IntVar, N: IntVar](
    x: Array[[*Elements[Batch], M, N]],
    *,
    pivoting: Literal[False] = False,
    full_matrices: Literal[False],
    use_magma: bool | None = None,
) -> tuple[
    Array[[*Elements[Batch], M, int_min(Int[M], Int[N])]],
    Array[[*Elements[Batch], int_min(Int[M], Int[N]), N]],
]: ...
@overload
def qr[Batch: IntTuple, M: IntVar, N: IntVar](
    x: Array[[*Elements[Batch], M, N]],
    *,
    pivoting: Literal[True],
    full_matrices: Literal[True] = True,
    use_magma: bool | None = None,
) -> tuple[
    Array[[*Elements[Batch], M, M]],
    Array[[*Elements[Batch], M, N]],
    Array[[*Elements[Batch], N]],
]: ...
@overload
def qr[Batch: IntTuple, M: IntVar, N: IntVar](
    x: Array[[*Elements[Batch], M, N]],
    *,
    pivoting: Literal[True],
    full_matrices: Literal[False],
    use_magma: bool | None = None,
) -> tuple[
    Array[[*Elements[Batch], M, int_min(Int[M], Int[N])]],
    Array[[*Elements[Batch], int_min(Int[M], Int[N]), N]],
    Array[[*Elements[Batch], N]],
]: ...
@overload
def qr(
    x: Array[Any],
    *,
    pivoting: bool = False,
    full_matrices: bool = True,
    use_magma: bool | None = None,
) -> (
    tuple[Array[IntTuple], Array[IntTuple]]
    | tuple[Array[IntTuple], Array[IntTuple], Array[IntTuple]]
): ...
def schur[Batch: IntTuple, N: IntVar](
    x: Array[[*Elements[Batch], N, N]],
    *,
    compute_schur_vectors: bool = True,
    sort_eig_vals: bool = False,
    select_callable: Any = None,
) -> tuple[Array[[*Elements[Batch], N, N]], Array[[*Elements[Batch], N, N]]]: ...
@overload
def svd[Batch: IntTuple, M: IntVar, N: IntVar](
    x: Array[[*Elements[Batch], M, N]],
    *,
    full_matrices: Literal[True] = True,
    compute_uv: Literal[True] = True,
    subset_by_index: tuple[int, int] | None = None,
    algorithm: SvdAlgorithm | str | None = None,
) -> tuple[
    Array[[*Elements[Batch], M, M]],
    Array[[*Elements[Batch], int_min(Int[M], Int[N])]],
    Array[[*Elements[Batch], N, N]],
]: ...
@overload
def svd[Batch: IntTuple, M: IntVar, N: IntVar](
    x: Array[[*Elements[Batch], M, N]],
    *,
    full_matrices: Literal[False],
    compute_uv: Literal[True] = True,
    subset_by_index: tuple[int, int] | None = None,
    algorithm: SvdAlgorithm | str | None = None,
) -> tuple[
    Array[[*Elements[Batch], M, int_min(Int[M], Int[N])]],
    Array[[*Elements[Batch], int_min(Int[M], Int[N])]],
    Array[[*Elements[Batch], int_min(Int[M], Int[N]), N]],
]: ...
@overload
def svd[Batch: IntTuple, M: IntVar, N: IntVar](
    x: Array[[*Elements[Batch], M, N]],
    *,
    full_matrices: bool = True,
    compute_uv: Literal[False],
    subset_by_index: tuple[int, int] | None = None,
    algorithm: SvdAlgorithm | str | None = None,
) -> Array[[*Elements[Batch], int_min(Int[M], Int[N])]]: ...
@overload
def svd(
    x: Array[Any],
    *,
    full_matrices: bool = True,
    compute_uv: bool = True,
    subset_by_index: tuple[int, int] | None = None,
    algorithm: SvdAlgorithm | str | None = None,
) -> Array[IntTuple] | tuple[Array[IntTuple], Array[IntTuple], Array[IntTuple]]: ...
def symmetric_product[AShape: _Shape, CShape: _Shape](
    a_matrix: Array[AShape],
    c_matrix: Array[CShape],
    *,
    alpha: float = 1.0,
    beta: float = 0.0,
    symmetrize_output: bool = False,
) -> Array[symmetric_product_shape(AShape, CShape)]: ...
def triangular_solve[AShape: _Shape, BShape: _Shape, LeftSide: Flag[bool] = False](
    a: Array[AShape],
    b: Array[BShape],
    *,
    left_side: LeftSide = False,
    lower: bool = False,
    transpose_a: bool = False,
    conjugate_a: bool = False,
    unit_diagonal: bool = False,
) -> Array[triangular_solve_shape(AShape, BShape, LeftSide)]: ...
def tridiagonal[AShape: _Shape](
    a: Array[AShape],
    *,
    lower: bool = True,
) -> tuple[
    Array[AShape],
    Array[tridiagonal_d_shape(AShape)],
    Array[tridiagonal_diag_minus_one_shape(AShape)],
    Array[tridiagonal_diag_minus_one_shape(AShape)],
]: ...
def tridiagonal_solve[DLShape: _Shape, DShape: _Shape, DUShape: _Shape, BShape: _Shape](
    dl: Array[DLShape],
    d: Array[DShape],
    du: Array[DUShape],
    b: Array[BShape],
    *,
    perturb_singular: bool = False,
) -> Array[tridiagonal_solve_shape(DLShape, DShape, DUShape, BShape)]: ...

cholesky_p: Any
cholesky_update_p: Any
eig_p: Any
eigh_p: Any
hessenberg_p: Any
householder_product_p: Any
lu_p: Any
lu_pivots_to_permutation_p: Any
ormqr_p: Any
qr_p: Any
schur_p: Any
svd_p: Any
symmetric_product_p: Any
triangular_solve_p: Any
tridiagonal_p: Any
tridiagonal_solve_p: Any
