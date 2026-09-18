# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Literal, overload, Sequence

from jax._array import Array as _Array, ArrayLike as _ArrayLike
from jax._shapes import (
    cross_axis_shape,
    diagonal_shape,
    int_min,
    matmul_shape,
    reduce_shape,
    tensordot_shape,
    tensorinv_shape,
    tensorsolve_shape,
    trace_shape,
)
from jax._src.sharding_impls import (
    NamedSharding as _NamedSharding,
    PartitionSpec as _PartitionSpec,
)
from jax.typing import DTypeLike
from shape_extensions import broadcast, Elements, Flag, Int, IntTuple, IntVar

type _Shape = IntTuple
type _Axis = int | tuple[int, ...] | None

def cholesky[Batch: IntTuple, N: IntVar](
    a: _ArrayLike[[*Elements[Batch], N, N]],
    *,
    upper: bool = False,
    symmetrize_input: bool = True,
) -> _Array[[*Elements[Batch], N, N]]: ...
def cond[Batch: IntTuple, M: IntVar, N: IntVar](
    x: _ArrayLike[[*Elements[Batch], M, N]],
    p: Any = None,
) -> _Array[Batch]: ...
def cross[
    Shape1: _Shape = [],
    Shape2: _Shape = [],
    Axis: Flag[int] = -1,
](
    x1: _ArrayLike[Shape1],
    x2: _ArrayLike[Shape2],
    /,
    *,
    axis: Axis = -1,
) -> _Array[cross_axis_shape(Shape1, Shape2, Axis)]: ...
def det[Batch: IntTuple, N: IntVar](
    a: _ArrayLike[[*Elements[Batch], N, N]],
) -> _Array[Batch]: ...
def diagonal[Shape: _Shape = [], Offset: Flag[int] = 0](
    x: _ArrayLike[Shape],
    /,
    *,
    offset: Offset = 0,
) -> _Array[diagonal_shape(Shape, Offset, -2, -1)]: ...
def eig[Batch: IntTuple, N: IntVar](
    a: _ArrayLike[[*Elements[Batch], N, N]],
) -> tuple[_Array[[*Elements[Batch], N]], _Array[[*Elements[Batch], N, N]]]: ...
def eigh[Batch: IntTuple, N: IntVar](
    a: _ArrayLike[[*Elements[Batch], N, N]],
    UPLO: str | None = None,
    symmetrize_input: bool = True,
) -> tuple[_Array[[*Elements[Batch], N]], _Array[[*Elements[Batch], N, N]]]: ...
def eigvals[Batch: IntTuple, N: IntVar](
    a: _ArrayLike[[*Elements[Batch], N, N]],
) -> _Array[[*Elements[Batch], N]]: ...
def eigvalsh[Batch: IntTuple, N: IntVar](
    a: _ArrayLike[[*Elements[Batch], N, N]],
    UPLO: str | None = "L",
    *,
    symmetrize_input: bool = True,
) -> _Array[[*Elements[Batch], N]]: ...
def inv[Batch: IntTuple, N: IntVar](
    a: _ArrayLike[[*Elements[Batch], N, N]],
) -> _Array[[*Elements[Batch], N, N]]: ...
def lstsq[Batch: IntTuple, M: IntVar, N: IntVar](
    a: _ArrayLike[[*Elements[Batch], M, N]],
    b: _ArrayLike[[*Elements[Batch], M]],
    rcond: float | None = None,
    *,
    numpy_resid: bool = False,
) -> tuple[
    _Array[[*Elements[Batch], N]],
    _Array[IntTuple],
    _Array[Batch],
    _Array[[*Elements[Batch], int_min(Int[M], Int[N])]],
]: ...
def matmul[LeftShape: _Shape = [], RightShape: _Shape = []](
    x1: _ArrayLike[LeftShape],
    x2: _ArrayLike[RightShape],
    /,
    *,
    precision: Any = None,
    preferred_element_type: Any = None,
) -> _Array[matmul_shape(LeftShape, RightShape)]: ...
@overload
def matrix_norm[Batch: IntTuple, M: IntVar, N: IntVar](
    x: _ArrayLike[[*Elements[Batch], M, N]],
    /,
    *,
    keepdims: Literal[False] = False,
    ord: Any = "fro",
) -> _Array[Batch]: ...
@overload
def matrix_norm[Batch: IntTuple, M: IntVar, N: IntVar](
    x: _ArrayLike[[*Elements[Batch], M, N]],
    /,
    *,
    keepdims: Literal[True],
    ord: Any = "fro",
) -> _Array[[*Elements[Batch], 1, 1]]: ...
@overload
def matrix_norm[Batch: IntTuple, M: IntVar, N: IntVar](
    x: _ArrayLike[[*Elements[Batch], M, N]],
    /,
    *,
    keepdims: bool = False,
    ord: Any = "fro",
) -> _Array[IntTuple]: ...
def matrix_power[Batch: IntTuple, N: IntVar](
    a: _ArrayLike[[*Elements[Batch], N, N]],
    n: int,
) -> _Array[[*Elements[Batch], N, N]]: ...
def matrix_rank[Batch: IntTuple, N: IntVar, K: IntVar](
    M: _ArrayLike[[*Elements[Batch], N, K]],
    rtol: Any = None,
    *,
    hermitian: bool = False,
    tol: Any = None,
) -> _Array[Batch]: ...
def matrix_transpose[Batch: IntTuple, M: IntVar, N: IntVar](
    x: _ArrayLike[[*Elements[Batch], M, N]],
    /,
) -> _Array[[*Elements[Batch], N, M]]: ...
def multi_dot(
    arrays: Sequence[_ArrayLike[Any]],
    *,
    precision: Any = None,
) -> _Array[IntTuple]: ...
@overload
def norm[Shape: _Shape = [], Axis: Flag[_Axis] = None, KeepDims: Flag[bool] = False](
    x: _ArrayLike[Shape],
    ord: Any = None,
    axis: Axis = None,
    keepdims: KeepDims = False,
) -> _Array[reduce_shape(Shape, Axis, KeepDims)]: ...
@overload
def norm[Shape: _Shape = []](
    x: _ArrayLike[Shape],
    ord: Any = None,
    axis: Sequence[int] = ...,
    keepdims: bool = False,
) -> _Array[IntTuple]: ...
def outer[N: IntVar, M: IntVar](
    x1: _ArrayLike[[N]],
    x2: _ArrayLike[[M]],
    /,
) -> _Array[[N, M]]: ...
def pinv[Batch: IntTuple, M: IntVar, N: IntVar](
    a: _ArrayLike[[*Elements[Batch], M, N]],
    rtol: Any = None,
    hermitian: bool = False,
    *,
    rcond: Any = None,
) -> _Array[[*Elements[Batch], N, M]]: ...
@overload
def qr[Batch: IntTuple, M: IntVar, N: IntVar](
    a: _ArrayLike[[*Elements[Batch], M, N]],
    mode: Literal["reduced"] = "reduced",
) -> tuple[
    _Array[[*Elements[Batch], M, int_min(Int[M], Int[N])]],
    _Array[[*Elements[Batch], int_min(Int[M], Int[N]), N]],
]: ...
@overload
def qr[Batch: IntTuple, M: IntVar, N: IntVar](
    a: _ArrayLike[[*Elements[Batch], M, N]],
    mode: Literal["r"],
) -> _Array[[*Elements[Batch], int_min(Int[M], Int[N]), N]]: ...
@overload
def qr[Batch: IntTuple, M: IntVar, N: IntVar](
    a: _ArrayLike[[*Elements[Batch], M, N]],
    mode: Literal["complete"],
) -> tuple[_Array[[*Elements[Batch], M, M]], _Array[[*Elements[Batch], M, N]]]: ...
@overload
def qr[Batch: IntTuple, M: IntVar, N: IntVar](
    a: _ArrayLike[[*Elements[Batch], M, N]],
    mode: str = "reduced",
) -> tuple[_Array[IntTuple], _Array[IntTuple]] | _Array[IntTuple]: ...
def slogdet[Batch: IntTuple, N: IntVar](
    a: _ArrayLike[[*Elements[Batch], N, N]],
    *,
    method: str | None = None,
) -> tuple[_Array[Batch], _Array[Batch]]: ...
@overload
def solve[Batch: IntTuple, N: IntVar](
    a: _ArrayLike[[*Elements[Batch], N, N]],
    b: _ArrayLike[[N]],
) -> _Array[[*Elements[Batch], N]]: ...
@overload
def solve[Batch: IntTuple, N: IntVar, M: IntVar](
    a: _ArrayLike[[*Elements[Batch], N, N]],
    b: _ArrayLike[[*Elements[Batch], N, M]],
) -> _Array[[*Elements[Batch], N, M]]: ...
@overload
def svd[Batch: IntTuple, M: IntVar, N: IntVar](
    a: _ArrayLike[[*Elements[Batch], M, N]],
    full_matrices: Literal[False],
    compute_uv: Literal[True] = True,
    hermitian: bool = False,
    subset_by_index: Any = None,
) -> tuple[
    _Array[[*Elements[Batch], M, int_min(Int[M], Int[N])]],
    _Array[[*Elements[Batch], int_min(Int[M], Int[N])]],
    _Array[[*Elements[Batch], int_min(Int[M], Int[N]), N]],
]: ...
@overload
def svd[Batch: IntTuple, M: IntVar, N: IntVar](
    a: _ArrayLike[[*Elements[Batch], M, N]],
    full_matrices: Literal[True] = True,
    compute_uv: Literal[True] = True,
    hermitian: bool = False,
    subset_by_index: Any = None,
) -> tuple[
    _Array[[*Elements[Batch], M, M]],
    _Array[[*Elements[Batch], int_min(Int[M], Int[N])]],
    _Array[[*Elements[Batch], N, N]],
]: ...
@overload
def svd[Batch: IntTuple, M: IntVar, N: IntVar](
    a: _ArrayLike[[*Elements[Batch], M, N]],
    full_matrices: bool = ...,
    *,
    compute_uv: Literal[False],
    hermitian: bool = False,
    subset_by_index: Any = None,
) -> _Array[[*Elements[Batch], int_min(Int[M], Int[N])]]: ...
@overload
def svd[Batch: IntTuple, M: IntVar, N: IntVar](
    a: _ArrayLike[[*Elements[Batch], M, N]],
    full_matrices: bool = True,
    compute_uv: bool = True,
    hermitian: bool = False,
    subset_by_index: Any = None,
) -> tuple[_Array[IntTuple], _Array[IntTuple], _Array[IntTuple]] | _Array[IntTuple]: ...
def svdvals[Batch: IntTuple, M: IntVar, N: IntVar](
    x: _ArrayLike[[*Elements[Batch], M, N]],
    /,
) -> _Array[[*Elements[Batch], int_min(Int[M], Int[N])]]: ...
def tensordot[
    Shape1: _Shape = [],
    Shape2: _Shape = [],
    Axes: Flag[_Axis] = 2,
](
    x1: _ArrayLike[Shape1],
    x2: _ArrayLike[Shape2],
    /,
    *,
    axes: Axes = 2,
    precision: Any = None,
    preferred_element_type: Any = None,
    out_sharding: _NamedSharding | _PartitionSpec | None = None,
) -> _Array[tensordot_shape(Shape1, Shape2, Axes)]: ...
def tensorinv[Shape: _Shape = [], Ind: Flag[int] = 2](
    a: _ArrayLike[Shape],
    ind: Ind = 2,
) -> _Array[tensorinv_shape(Shape, Ind)]: ...
def tensorsolve[
    Shape1: _Shape = [],
    Shape2: _Shape = [],
    Axes: Flag[_Axis] = None,
](
    a: _ArrayLike[Shape1],
    b: _ArrayLike[Shape2],
    axes: Axes = None,
) -> _Array[tensorsolve_shape(Shape1, Shape2, Axes)]: ...
def trace[Shape: _Shape = [], Offset: Flag[int] = 0](
    x: _ArrayLike[Shape],
    /,
    *,
    offset: Offset = 0,
    dtype: DTypeLike | None = None,
) -> _Array[trace_shape(Shape, Offset, -2, -1)]: ...
def vecdot[Shape1: _Shape = [], Shape2: _Shape = [], Axis: Flag[_Axis] = -1](
    x1: _ArrayLike[Shape1],
    x2: _ArrayLike[Shape2],
    /,
    *,
    axis: Axis = -1,
    precision: Any = None,
    preferred_element_type: Any = None,
) -> _Array[reduce_shape(broadcast(Shape1, Shape2), Axis, False)]: ...
@overload
def vector_norm[
    Shape: _Shape = [],
    Axis: Flag[_Axis] = None,
    KeepDims: Flag[bool] = False,
](
    x: _ArrayLike[Shape],
    /,
    *,
    axis: Axis = None,
    keepdims: KeepDims = False,
    ord: Any = 2,
) -> _Array[reduce_shape(Shape, Axis, KeepDims)]: ...
@overload
def vector_norm[Shape: _Shape = []](
    x: _ArrayLike[Shape],
    /,
    *,
    axis: Sequence[int] = ...,
    keepdims: bool = False,
    ord: Any = 2,
) -> _Array[IntTuple]: ...
