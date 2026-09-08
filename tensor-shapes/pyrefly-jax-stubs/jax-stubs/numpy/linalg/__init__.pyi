# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Literal, overload, Sequence

from jax._array import Array
from jax._shapes import (
    cross_axis_shape,
    int_min,
    matmul_shape,
    reduce_shape,
)
from shape_extensions import broadcast, Elements, Flag, Int, IntTuple, IntVar

type _Shape = IntTuple
type _Axis = int | tuple[int, ...] | None

def cholesky[Batch: IntTuple, N: IntVar](
    a: Array[[*Elements[Batch], N, N]],
    *,
    upper: bool = False,
    symmetrize_input: bool = True,
) -> Array[[*Elements[Batch], N, N]]: ...
def cond[Batch: IntTuple, M: IntVar, N: IntVar](
    x: Array[[*Elements[Batch], M, N]],
    p: Any = None,
) -> Array[Batch]: ...
@overload
def cross[
    Shape1: _Shape,
    Shape2: _Shape,
    Axis: Flag[int] = -1,
](
    x1: Array[Shape1],
    x2: Array[Shape2],
    /,
    *,
    axis: Axis = -1,
) -> Array[cross_axis_shape(Shape1, Shape2, Axis)]: ...
@overload
def cross(
    x1: Array[Any],
    x2: Array[Any],
    /,
    *,
    axis: int = -1,
) -> Array[IntTuple]: ...
def det[Batch: IntTuple, N: IntVar](
    a: Array[[*Elements[Batch], N, N]],
) -> Array[Batch]: ...
def diagonal[Shape: _Shape](
    x: Array[Shape],
    /,
    *,
    offset: int = 0,
) -> Array[IntTuple]: ...
def eig[Batch: IntTuple, N: IntVar](
    a: Array[[*Elements[Batch], N, N]],
) -> tuple[Array[[*Elements[Batch], N]], Array[[*Elements[Batch], N, N]]]: ...
def eigh[Batch: IntTuple, N: IntVar](
    a: Array[[*Elements[Batch], N, N]],
    UPLO: str | None = None,
    symmetrize_input: bool = True,
) -> tuple[Array[[*Elements[Batch], N]], Array[[*Elements[Batch], N, N]]]: ...
def eigvals[Batch: IntTuple, N: IntVar](
    a: Array[[*Elements[Batch], N, N]],
) -> Array[[*Elements[Batch], N]]: ...
def eigvalsh[Batch: IntTuple, N: IntVar](
    a: Array[[*Elements[Batch], N, N]],
    UPLO: str | None = "L",
    *,
    symmetrize_input: bool = True,
) -> Array[[*Elements[Batch], N]]: ...
def inv[Batch: IntTuple, N: IntVar](
    a: Array[[*Elements[Batch], N, N]],
) -> Array[[*Elements[Batch], N, N]]: ...
def lstsq[Batch: IntTuple, M: IntVar, N: IntVar](
    a: Array[[*Elements[Batch], M, N]],
    b: Array[[*Elements[Batch], M]],
    rcond: float | None = None,
    *,
    numpy_resid: bool = False,
) -> tuple[
    Array[[*Elements[Batch], N]],
    Array[IntTuple],
    Array[Batch],
    Array[[*Elements[Batch], int_min(Int[M], Int[N])]],
]: ...
def matmul[LeftShape: _Shape, RightShape: _Shape](
    x1: Array[LeftShape],
    x2: Array[RightShape],
    /,
    *,
    precision: Any = None,
    preferred_element_type: Any = None,
) -> Array[matmul_shape(LeftShape, RightShape)]: ...
@overload
def matrix_norm[Batch: IntTuple, M: IntVar, N: IntVar](
    x: Array[[*Elements[Batch], M, N]],
    /,
    *,
    keepdims: Literal[False] = False,
    ord: Any = "fro",
) -> Array[Batch]: ...
@overload
def matrix_norm[Batch: IntTuple, M: IntVar, N: IntVar](
    x: Array[[*Elements[Batch], M, N]],
    /,
    *,
    keepdims: Literal[True],
    ord: Any = "fro",
) -> Array[[*Elements[Batch], 1, 1]]: ...
@overload
def matrix_norm(
    x: Array[Any],
    /,
    *,
    keepdims: bool = False,
    ord: Any = "fro",
) -> Array[IntTuple]: ...
def matrix_power[Batch: IntTuple, N: IntVar](
    a: Array[[*Elements[Batch], N, N]],
    n: int,
) -> Array[[*Elements[Batch], N, N]]: ...
def matrix_rank[Batch: IntTuple, N: IntVar, K: IntVar](
    M: Array[[*Elements[Batch], N, K]],
    rtol: Any = None,
    *,
    hermitian: bool = False,
    tol: Any = None,
) -> Array[Batch]: ...
def matrix_transpose[Batch: IntTuple, M: IntVar, N: IntVar](
    x: Array[[*Elements[Batch], M, N]],
    /,
) -> Array[[*Elements[Batch], N, M]]: ...
def multi_dot(
    arrays: Sequence[Array[Any]],
    *,
    precision: Any = None,
) -> Array[IntTuple]: ...
@overload
def norm[Shape: _Shape, Axis: Flag[_Axis], KeepDims: Flag[bool]](
    x: Array[Shape],
    ord: Any = None,
    axis: Axis = None,
    keepdims: KeepDims = False,
) -> Array[reduce_shape(Shape, Axis, KeepDims)]: ...
@overload
def norm[Shape: _Shape](
    x: Array[Shape],
    ord: Any = None,
    axis: Sequence[int] = ...,
    keepdims: bool = False,
) -> Array[IntTuple]: ...
@overload
def outer[N: IntVar, M: IntVar](
    x1: Array[[N]],
    x2: Array[[M]],
    /,
) -> Array[[N, M]]: ...
@overload
def outer(
    x1: Array[Any],
    x2: Array[Any],
    /,
) -> Array[IntTuple]: ...
def pinv[Batch: IntTuple, M: IntVar, N: IntVar](
    a: Array[[*Elements[Batch], M, N]],
    rtol: Any = None,
    hermitian: bool = False,
    *,
    rcond: Any = None,
) -> Array[[*Elements[Batch], N, M]]: ...
@overload
def qr[Batch: IntTuple, M: IntVar, N: IntVar](
    a: Array[[*Elements[Batch], M, N]],
    mode: Literal["reduced"] = "reduced",
) -> tuple[
    Array[[*Elements[Batch], M, int_min(Int[M], Int[N])]],
    Array[[*Elements[Batch], int_min(Int[M], Int[N]), N]],
]: ...
@overload
def qr[Batch: IntTuple, M: IntVar, N: IntVar](
    a: Array[[*Elements[Batch], M, N]],
    mode: Literal["r"],
) -> Array[[*Elements[Batch], int_min(Int[M], Int[N]), N]]: ...
@overload
def qr[Batch: IntTuple, M: IntVar, N: IntVar](
    a: Array[[*Elements[Batch], M, N]],
    mode: Literal["complete"],
) -> tuple[Array[[*Elements[Batch], M, M]], Array[[*Elements[Batch], M, N]]]: ...
@overload
def qr(
    a: Array[Any],
    mode: str = "reduced",
) -> tuple[Array[IntTuple], Array[IntTuple]] | Array[IntTuple]: ...
def slogdet[Batch: IntTuple, N: IntVar](
    a: Array[[*Elements[Batch], N, N]],
    *,
    method: str | None = None,
) -> tuple[Array[Batch], Array[Batch]]: ...
@overload
def solve[Batch: IntTuple, N: IntVar](
    a: Array[[*Elements[Batch], N, N]],
    b: Array[[N]],
) -> Array[[*Elements[Batch], N]]: ...
@overload
def solve[Batch: IntTuple, N: IntVar, M: IntVar](
    a: Array[[*Elements[Batch], N, N]],
    b: Array[[*Elements[Batch], N, M]],
) -> Array[[*Elements[Batch], N, M]]: ...
@overload
def svd[Batch: IntTuple, M: IntVar, N: IntVar](
    a: Array[[*Elements[Batch], M, N]],
    full_matrices: Literal[False],
    compute_uv: Literal[True] = True,
    hermitian: bool = False,
    subset_by_index: Any = None,
) -> tuple[
    Array[[*Elements[Batch], M, int_min(Int[M], Int[N])]],
    Array[[*Elements[Batch], int_min(Int[M], Int[N])]],
    Array[[*Elements[Batch], int_min(Int[M], Int[N]), N]],
]: ...
@overload
def svd[Batch: IntTuple, M: IntVar, N: IntVar](
    a: Array[[*Elements[Batch], M, N]],
    full_matrices: Literal[True] = True,
    compute_uv: Literal[True] = True,
    hermitian: bool = False,
    subset_by_index: Any = None,
) -> tuple[
    Array[[*Elements[Batch], M, M]],
    Array[[*Elements[Batch], int_min(Int[M], Int[N])]],
    Array[[*Elements[Batch], N, N]],
]: ...
@overload
def svd[Batch: IntTuple, M: IntVar, N: IntVar](
    a: Array[[*Elements[Batch], M, N]],
    full_matrices: bool = ...,
    *,
    compute_uv: Literal[False],
    hermitian: bool = False,
    subset_by_index: Any = None,
) -> Array[[*Elements[Batch], int_min(Int[M], Int[N])]]: ...
@overload
def svd(
    a: Array[Any],
    full_matrices: bool = True,
    compute_uv: bool = True,
    hermitian: bool = False,
    subset_by_index: Any = None,
) -> tuple[Array[IntTuple], Array[IntTuple], Array[IntTuple]] | Array[IntTuple]: ...
def svdvals[Batch: IntTuple, M: IntVar, N: IntVar](
    x: Array[[*Elements[Batch], M, N]],
    /,
) -> Array[[*Elements[Batch], int_min(Int[M], Int[N])]]: ...
def tensordot[Shape1: _Shape, Shape2: _Shape](
    x1: Array[Shape1],
    x2: Array[Shape2],
    /,
    *,
    axes: Any = 2,
    precision: Any = None,
    preferred_element_type: Any = None,
    out_sharding: Any = None,
) -> Array[IntTuple]: ...
def tensorinv[Shape: _Shape](
    a: Array[Shape],
    ind: int = 2,
) -> Array[IntTuple]: ...
def tensorsolve[Shape1: _Shape, Shape2: _Shape](
    a: Array[Shape1],
    b: Array[Shape2],
    axes: tuple[int, ...] | None = None,
) -> Array[IntTuple]: ...
def trace[Shape: _Shape](
    x: Array[Shape],
    /,
    *,
    offset: int = 0,
    dtype: Any = None,
) -> Array[IntTuple]: ...
def vecdot[Shape1: _Shape, Shape2: _Shape](
    x1: Array[Shape1],
    x2: Array[Shape2],
    /,
    *,
    axis: int = -1,
    precision: Any = None,
    preferred_element_type: Any = None,
) -> Array[IntTuple]: ...
@overload
def vector_norm[Shape: _Shape, Axis: Flag[_Axis], KeepDims: Flag[bool]](
    x: Array[Shape],
    /,
    *,
    axis: Axis = None,
    keepdims: KeepDims = False,
    ord: Any = 2,
) -> Array[reduce_shape(Shape, Axis, KeepDims)]: ...
@overload
def vector_norm(
    x: Array[Any],
    /,
    *,
    axis: Sequence[int] = ...,
    keepdims: bool = False,
    ord: Any = 2,
) -> Array[IntTuple]: ...
