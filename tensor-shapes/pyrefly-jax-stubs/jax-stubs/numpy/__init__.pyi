# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import (
    Any,
    Callable,
    ContextManager,
    Literal,
    NamedTuple,
    overload,
    Sequence,
    Unpack,
)

import numpy as np
from jax._array import Array as _Array, Array as ndarray, ArrayLike as _ArrayLike
from jax._shapes import (
    append_shape,
    arange_size,
    arange_stop,
    atleast_1d_shape,
    atleast_2d_shape,
    atleast_3d_shape,
    broadcast_to_shape,
    column_stack_shape,
    compress_shape,
    concatenate_shape,
    convolve_shape,
    cross_axes_shape,
    cross_axis_shape,
    diag_indices_from_shape,
    diagonal_shape,
    dot_shape,
    dstack_shape,
    einsum_shape,
    expand_dims_shape,
    fill_diagonal_shape,
    flip_shape,
    histogram2d_counts_shape,
    histogram_counts_shape,
    histogram_edges_shape,
    hstack_shape,
    inner_shape,
    int_min,
    ix_shapes,
    kron_shape,
    linspace_shape,
    matmul_shape,
    matvec_shape,
    moveaxis_shape,
    packbits_shape,
    permute_shape,
    poly_shape,
    polyadd_shape,
    polyder_shape,
    polydiv_quotient_shape,
    polyfit_cov_shape,
    polyfit_shape,
    polyint_shape,
    ravel_shape,
    reduce_shape,
    repeat_shape,
    reshape_shape,
    reverse_shape,
    roll_shape,
    rollaxis_shape,
    rot90_shape,
    sort_shape,
    squeeze_shape,
    stack_shape,
    swapaxes_shape,
    take_along_axis_shape,
    take_shape,
    tensordot_shape,
    tile_shape,
    top_k_shape,
    trace_shape,
    unpackbits_shape,
    unstack_shape,
    vecmat_shape,
    vstack_shape,
)
from jax._src.lib import Device as _Device
from jax._src.sharding_impls import (
    NamedSharding as _NamedSharding,
    PartitionSpec as _PartitionSpec,
)
from jax.sharding import Sharding as _Sharding
from jax.typing import DTypeLike
from numpy import (
    array_repr as array_repr,
    array_str as array_str,
    character as character,
    complexfloating as complexfloating,
    dtype as dtype,
    e as e,
    euler_gamma as euler_gamma,
    flexible as flexible,
    floating as floating,
    generic as generic,
    inexact as inexact,
    inf as inf,
    integer as integer,
    iterable as iterable,
    nan as nan,
    newaxis as newaxis,
    number as number,
    object_ as object_,
    pi as pi,
    save as save,
    savez as savez,
    signedinteger as signedinteger,
    unsignedinteger as unsignedinteger,
)
from shape_extensions import (
    broadcast,
    Elements,
    Flag,
    Int,
    IntTuple,
    IntTuples,
    IntVar,
    MapIntTuples,
    RegularNestedList,
)

from . import fft as fft, linalg as linalg

type _Shape = IntTuple
type _Axis = int | tuple[int, ...] | None
# The trailing `None` is not a legal argument to `reshape`. It is present because
# an `int | tuple[int, ...]` parameter cannot be iterated inside a DSL function
# after narrowing with `is_int_value` alone. See `reshape_shape`, which rejects it.
type _NewShape = int | tuple[int, ...] | None
type _Scalar = bool | int | float | complex | np.number
type _ShapedArrayLike[Shape: _Shape] = _Array[Shape] | np.ndarray[Shape]

@overload
def array[Shape: _Shape = []](
    object: _Scalar | _ShapedArrayLike[Shape] | RegularNestedList[Shape, _Scalar],
    dtype: DTypeLike | None = ...,
    copy: bool | None = ...,
    order: str | None = ...,
    ndmin: Literal[0] = 0,
    *,
    device: _Device | _Sharding | None = ...,
    out_sharding: _NamedSharding | _PartitionSpec | None = ...,
) -> _Array[Shape]: ...
@overload
def array(
    object: Any,
    dtype: DTypeLike | None = ...,
    copy: bool | None = ...,
    order: str | None = ...,
    ndmin: int = 0,
    *,
    device: _Device | _Sharding | None = ...,
    out_sharding: _NamedSharding | _PartitionSpec | None = ...,
) -> _Array[IntTuple]: ...
@overload
def asarray[Shape: _Shape = []](
    a: _Scalar | _ShapedArrayLike[Shape] | RegularNestedList[Shape, _Scalar],
    dtype: DTypeLike | None = ...,
    order: str | None = ...,
    *,
    copy: bool | None = ...,
    device: _Device | _Sharding | None = ...,
    out_sharding: _NamedSharding | _PartitionSpec | None = ...,
) -> _Array[Shape]: ...
@overload
def asarray(
    a: Any,
    dtype: DTypeLike | None = ...,
    order: str | None = ...,
    *,
    copy: bool | None = ...,
    device: _Device | _Sharding | None = ...,
    out_sharding: _NamedSharding | _PartitionSpec | None = ...,
) -> _Array[IntTuple]: ...
def copy[Shape: _Shape = []](
    a: _ArrayLike[Shape], order: str | None = None
) -> _Array[Shape]: ...

# Literal tuples and values typed as `IntTuple` retain their shape. Other integer
# sequences fall through to a gradual overload rather than being rejected.
# TODO(stroxler): Replace these finite tuple-shape constructor overloads with a
# single `Shape: tuple[int, ...]` overload once whole-shape parameters flow
# through downstream array operations without degrading to unknown. The NumPy
# stubs carry the same limitation.
@overload
def zeros[N: IntVar](
    shape: Int[N],
    dtype: DTypeLike | None = ...,
    *,
    device: _Device | _Sharding | None = ...,
) -> _Array[[N]]: ...
@overload
def zeros[Shape: _Shape](
    shape: Shape,
    dtype: DTypeLike | None = ...,
    *,
    device: _Device | _Sharding | None = ...,
) -> _Array[Shape]: ...
@overload
def zeros(
    shape: Sequence[int] | int,
    dtype: DTypeLike | None = ...,
    *,
    device: _Device | _Sharding | None = ...,
) -> _Array[IntTuple]: ...
@overload
def ones[N: IntVar](
    shape: Int[N],
    dtype: DTypeLike | None = ...,
    *,
    device: _Device | _Sharding | None = ...,
) -> _Array[[N]]: ...
@overload
def ones[Shape: _Shape](
    shape: Shape,
    dtype: DTypeLike | None = ...,
    *,
    device: _Device | _Sharding | None = ...,
) -> _Array[Shape]: ...
@overload
def ones(
    shape: Sequence[int] | int,
    dtype: DTypeLike | None = ...,
    *,
    device: _Device | _Sharding | None = ...,
) -> _Array[IntTuple]: ...
@overload
def empty[N: IntVar](
    shape: Int[N],
    dtype: DTypeLike | None = ...,
    *,
    device: _Device | _Sharding | None = ...,
    out_sharding: _NamedSharding | _PartitionSpec | None = ...,
) -> _Array[[N]]: ...
@overload
def empty[Shape: _Shape](
    shape: Shape,
    dtype: DTypeLike | None = ...,
    *,
    device: _Device | _Sharding | None = ...,
    out_sharding: _NamedSharding | _PartitionSpec | None = ...,
) -> _Array[Shape]: ...
@overload
def empty(
    shape: Sequence[int] | int,
    dtype: DTypeLike | None = ...,
    *,
    device: _Device | _Sharding | None = ...,
    out_sharding: _NamedSharding | _PartitionSpec | None = ...,
) -> _Array[IntTuple]: ...
@overload
def full[N: IntVar](
    shape: Int[N],
    fill_value: Any,
    dtype: DTypeLike | None = ...,
    *,
    device: _Device | _Sharding | None = ...,
) -> _Array[[N]]: ...
@overload
def full[Shape: _Shape](
    shape: Shape,
    fill_value: Any,
    dtype: DTypeLike | None = ...,
    *,
    device: _Device | _Sharding | None = ...,
) -> _Array[Shape]: ...
@overload
def full(
    shape: Sequence[int] | int,
    fill_value: Any,
    dtype: DTypeLike | None = ...,
    *,
    device: _Device | _Sharding | None = ...,
) -> _Array[IntTuple]: ...

# `_like` constructors
@overload
def empty_like[Shape: _Shape = []](
    prototype: _ArrayLike[Shape],
    dtype: DTypeLike | None = ...,
    shape: None = None,
    *,
    device: _Device | _Sharding | None = ...,
) -> _Array[Shape]: ...
@overload
def empty_like[N: IntVar](
    prototype: _ArrayLike[Any],
    dtype: DTypeLike | None = ...,
    shape: Int[N] = ...,
    *,
    device: _Device | _Sharding | None = ...,
) -> _Array[[N]]: ...
@overload
def empty_like[Shape: _Shape](
    prototype: _ArrayLike[Any],
    dtype: DTypeLike | None = ...,
    shape: Shape = ...,
    *,
    device: _Device | _Sharding | None = ...,
) -> _Array[Shape]: ...
@overload
def empty_like(
    prototype: _ArrayLike[Any],
    dtype: DTypeLike | None = ...,
    shape: Sequence[int] | int | None = None,
    *,
    device: _Device | _Sharding | None = ...,
) -> _Array[IntTuple]: ...
@overload
def zeros_like[Shape: _Shape = []](
    a: _ArrayLike[Shape],
    dtype: DTypeLike | None = ...,
    shape: None = None,
    *,
    device: _Device | _Sharding | None = ...,
    out_sharding: _NamedSharding | _PartitionSpec | None = ...,
) -> _Array[Shape]: ...
@overload
def zeros_like[N: IntVar](
    a: _ArrayLike[Any],
    dtype: DTypeLike | None = ...,
    shape: Int[N] = ...,
    *,
    device: _Device | _Sharding | None = ...,
    out_sharding: _NamedSharding | _PartitionSpec | None = ...,
) -> _Array[[N]]: ...
@overload
def zeros_like[Shape: _Shape](
    a: _ArrayLike[Any],
    dtype: DTypeLike | None = ...,
    shape: Shape = ...,
    *,
    device: _Device | _Sharding | None = ...,
    out_sharding: _NamedSharding | _PartitionSpec | None = ...,
) -> _Array[Shape]: ...
@overload
def zeros_like(
    a: _ArrayLike[Any],
    dtype: DTypeLike | None = ...,
    shape: Sequence[int] | int | None = None,
    *,
    device: _Device | _Sharding | None = ...,
    out_sharding: _NamedSharding | _PartitionSpec | None = ...,
) -> _Array[IntTuple]: ...
@overload
def ones_like[Shape: _Shape = []](
    a: _ArrayLike[Shape],
    dtype: DTypeLike | None = ...,
    shape: None = None,
    *,
    device: _Device | _Sharding | None = ...,
    out_sharding: _NamedSharding | _PartitionSpec | None = ...,
) -> _Array[Shape]: ...
@overload
def ones_like[N: IntVar](
    a: _ArrayLike[Any],
    dtype: DTypeLike | None = ...,
    shape: Int[N] = ...,
    *,
    device: _Device | _Sharding | None = ...,
    out_sharding: _NamedSharding | _PartitionSpec | None = ...,
) -> _Array[[N]]: ...
@overload
def ones_like[Shape: _Shape](
    a: _ArrayLike[Any],
    dtype: DTypeLike | None = ...,
    shape: Shape = ...,
    *,
    device: _Device | _Sharding | None = ...,
    out_sharding: _NamedSharding | _PartitionSpec | None = ...,
) -> _Array[Shape]: ...
@overload
def ones_like(
    a: _ArrayLike[Any],
    dtype: DTypeLike | None = ...,
    shape: Sequence[int] | int | None = None,
    *,
    device: _Device | _Sharding | None = ...,
    out_sharding: _NamedSharding | _PartitionSpec | None = ...,
) -> _Array[IntTuple]: ...
@overload
def full_like[Shape: _Shape = []](
    a: _ArrayLike[Shape],
    fill_value: Any,
    dtype: DTypeLike | None = ...,
    shape: None = None,
    *,
    device: _Device | _Sharding | None = ...,
    out_sharding: _NamedSharding | _PartitionSpec | None = ...,
) -> _Array[Shape]: ...
@overload
def full_like[N: IntVar](
    a: _ArrayLike[Any],
    fill_value: Any,
    dtype: DTypeLike | None = ...,
    shape: Int[N] = ...,
    *,
    device: _Device | _Sharding | None = ...,
    out_sharding: _NamedSharding | _PartitionSpec | None = ...,
) -> _Array[[N]]: ...
@overload
def full_like[Shape: _Shape](
    a: _ArrayLike[Any],
    fill_value: Any,
    dtype: DTypeLike | None = ...,
    shape: Shape = ...,
    *,
    device: _Device | _Sharding | None = ...,
    out_sharding: _NamedSharding | _PartitionSpec | None = ...,
) -> _Array[Shape]: ...
@overload
def full_like(
    a: _ArrayLike[Any],
    fill_value: Any,
    dtype: DTypeLike | None = ...,
    shape: Sequence[int] | int | None = None,
    *,
    device: _Device | _Sharding | None = ...,
    out_sharding: _NamedSharding | _PartitionSpec | None = ...,
) -> _Array[IntTuple]: ...

# `arange`, `linspace`, `logspace`, `geomspace`
@overload
def arange[N: IntVar](
    start: Int[N],
    *,
    dtype: DTypeLike | None = ...,
    device: _Device | _Sharding | None = ...,
) -> _Array[[arange_stop(Int[N])]]: ...
@overload
def arange(
    start: float,
    *,
    dtype: DTypeLike | None = ...,
    device: _Device | _Sharding | None = ...,
) -> _Array[[int]]: ...
@overload
def arange[Start: Flag[int], Stop: Flag[int]](
    start: Start,
    stop: Stop,
    step: Literal[1] = ...,
    dtype: DTypeLike | None = ...,
) -> _Array[[arange_size(Start, Stop, 1)]]: ...
@overload
def arange[Start: Flag[int], Stop: Flag[int], Step: Flag[int]](
    start: Start,
    stop: Stop,
    step: Step,
    dtype: DTypeLike | None = ...,
) -> _Array[[arange_size(Start, Stop, Step)]]: ...
@overload
def arange(
    start: int | float,
    stop: int | float,
    step: int | float = ...,
    dtype: DTypeLike | None = ...,
) -> _Array[[int]]: ...
@overload
def linspace[
    StartShape: _Shape = [],
    StopShape: _Shape = [],
    N: IntVar = 50,
    Axis: Flag[int] = 0,
](
    start: _ArrayLike[StartShape],
    stop: _ArrayLike[StopShape],
    num: Int[N] = 50,
    endpoint: bool = True,
    retstep: Literal[False] = False,
    dtype: DTypeLike | None = None,
    axis: Axis = 0,
    *,
    device: _Device | _Sharding | None = None,
) -> _Array[linspace_shape(broadcast(StartShape, StopShape), Int[N], Axis)]: ...
@overload
def linspace[
    StartShape: _Shape = [],
    StopShape: _Shape = [],
    N: IntVar = 50,
    Axis: Flag[int] = 0,
](
    start: _ArrayLike[StartShape],
    stop: _ArrayLike[StopShape],
    num: Int[N] = 50,
    endpoint: bool = True,
    retstep: Literal[True] = ...,
    dtype: DTypeLike | None = None,
    axis: Axis = 0,
    *,
    device: _Device | _Sharding | None = None,
) -> tuple[
    _Array[linspace_shape(broadcast(StartShape, StopShape), Int[N], Axis)],
    _Array[broadcast(StartShape, StopShape)],
]: ...
@overload
def linspace[
    StartShape: _Shape = [],
    StopShape: _Shape = [],
](
    start: _ArrayLike[StartShape],
    stop: _ArrayLike[StopShape],
    num: int = 50,
    endpoint: bool = True,
    retstep: bool = False,
    dtype: DTypeLike | None = None,
    axis: int = 0,
    *,
    device: _Device | _Sharding | None = None,
) -> (
    _Array[IntTuple] | tuple[_Array[IntTuple], _Array[broadcast(StartShape, StopShape)]]
): ...
@overload
def logspace[
    StartShape: _Shape = [],
    StopShape: _Shape = [],
    N: IntVar = 50,
    Axis: Flag[int] = 0,
](
    start: _ArrayLike[StartShape],
    stop: _ArrayLike[StopShape],
    num: Int[N] = 50,
    endpoint: bool = True,
    base: _ArrayLike[Any] = 10.0,
    dtype: DTypeLike | None = None,
    axis: Axis = 0,
) -> _Array[linspace_shape(broadcast(StartShape, StopShape), Int[N], Axis)]: ...
@overload
def logspace(
    start: _ArrayLike[Any],
    stop: _ArrayLike[Any],
    num: int = 50,
    endpoint: bool = True,
    base: _ArrayLike[Any] = 10.0,
    dtype: DTypeLike | None = None,
    axis: int = 0,
) -> _Array[IntTuple]: ...
@overload
def geomspace[
    StartShape: _Shape = [],
    StopShape: _Shape = [],
    N: IntVar = 50,
    Axis: Flag[int] = 0,
](
    start: _ArrayLike[StartShape],
    stop: _ArrayLike[StopShape],
    num: Int[N] = 50,
    endpoint: bool = True,
    dtype: DTypeLike | None = None,
    axis: Axis = 0,
) -> _Array[linspace_shape(broadcast(StartShape, StopShape), Int[N], Axis)]: ...
@overload
def geomspace(
    start: _ArrayLike[Any],
    stop: _ArrayLike[Any],
    num: int = 50,
    endpoint: bool = True,
    dtype: DTypeLike | None = None,
    axis: int = 0,
) -> _Array[IntTuple]: ...

# `eye`, `identity`, `diag`, `diagflat`, `tri`, `tril`, `triu`, `vander`
@overload
def eye[N: IntVar](
    N: Int[N],
    M: None = ...,
    k: int = ...,
    dtype: DTypeLike | None = ...,
    *,
    device: _Device | _Sharding | None = ...,
) -> _Array[[N, N]]: ...
@overload
def eye[N: IntVar, M: IntVar](
    N: Int[N],
    M: Int[M],
    k: int = ...,
    dtype: DTypeLike | None = ...,
    *,
    device: _Device | _Sharding | None = ...,
) -> _Array[[N, M]]: ...
def identity[N: IntVar](
    n: Int[N],
    dtype: DTypeLike | None = ...,
    *,
    device: _Device | _Sharding | None = ...,
) -> _Array[[N, N]]: ...
@overload
def diag[N: IntVar](v: _ShapedArrayLike[[N]], k: int = 0) -> _Array[[N, N]]: ...
@overload
def diag[N: IntVar, M: IntVar](
    v: _ShapedArrayLike[[N, M]], k: int = 0
) -> _Array[[int_min(Int[N], Int[M])]]: ...
@overload
def diag(v: _ShapedArrayLike[Any], k: int = 0) -> _Array[IntTuple]: ...
@overload
def diagflat[N: IntVar](v: _ShapedArrayLike[[N]], k: int = 0) -> _Array[[N, N]]: ...
@overload
def diagflat(v: _ArrayLike[Any], k: int = 0) -> _Array[IntTuple]: ...
@overload
def tri[N: IntVar](
    N: Int[N], M: None = None, k: int = 0, dtype: DTypeLike | None = None
) -> _Array[[N, N]]: ...
@overload
def tri[N: IntVar, M: IntVar](
    N: Int[N], M: Int[M], k: int = 0, dtype: DTypeLike | None = None
) -> _Array[[N, M]]: ...
@overload
def tri(
    N: int, M: int | None = None, k: int = 0, dtype: DTypeLike | None = None
) -> _Array[IntTuple]: ...
def tril[Shape: _Shape = []](m: _ArrayLike[Shape], k: int = 0) -> _Array[Shape]: ...
def triu[Shape: _Shape = []](m: _ArrayLike[Shape], k: int = 0) -> _Array[Shape]: ...
@overload
def vander[M: IntVar](
    x: _ShapedArrayLike[[M]], N: None = None, increasing: bool = False
) -> _Array[[M, M]]: ...
@overload
def vander[M: IntVar, N: IntVar](
    x: _ShapedArrayLike[[M]], N: Int[N], increasing: bool = False
) -> _Array[[M, N]]: ...
@overload
def vander(
    x: _ShapedArrayLike[Any], N: int | None = None, increasing: bool = False
) -> _Array[IntTuple]: ...

# `indices`, `meshgrid`
@overload
def indices[N: IntVar](
    dimensions: IntTuple[N],
    dtype: DTypeLike | None = None,
    sparse: Literal[False] = False,
) -> _Array[[1, N]]: ...
@overload
def indices[N: IntVar, M: IntVar](
    dimensions: IntTuple[N, M],
    dtype: DTypeLike | None = None,
    sparse: Literal[False] = False,
) -> _Array[[2, N, M]]: ...
@overload
def indices[N: IntVar, M: IntVar, K: IntVar](
    dimensions: IntTuple[N, M, K],
    dtype: DTypeLike | None = None,
    sparse: Literal[False] = False,
) -> _Array[[3, N, M, K]]: ...
@overload
def indices(
    dimensions: Sequence[int], dtype: DTypeLike | None = None, sparse: bool = False
) -> _Array[IntTuple] | tuple[_Array[IntTuple], ...]: ...
@overload
def meshgrid[N: IntVar, M: IntVar](
    x1: _ShapedArrayLike[[N]],
    x2: _ShapedArrayLike[[M]],
    /,
    *,
    copy: bool = True,
    sparse: Literal[False] = False,
    indexing: Literal["xy"] = "xy",
) -> tuple[_Array[[M, N]], _Array[[M, N]]]: ...
@overload
def meshgrid[N: IntVar, M: IntVar](
    x1: _ShapedArrayLike[[N]],
    x2: _ShapedArrayLike[[M]],
    /,
    *,
    copy: bool = True,
    sparse: Literal[False] = False,
    indexing: Literal["ij"],
) -> tuple[_Array[[N, M]], _Array[[N, M]]]: ...
@overload
def meshgrid[N: IntVar, M: IntVar, K: IntVar](
    x1: _ShapedArrayLike[[N]],
    x2: _ShapedArrayLike[[M]],
    x3: _ShapedArrayLike[[K]],
    /,
    *,
    copy: bool = True,
    sparse: Literal[False] = False,
    indexing: Literal["xy"] = "xy",
) -> tuple[_Array[[M, N, K]], _Array[[M, N, K]], _Array[[M, N, K]]]: ...
@overload
def meshgrid[N: IntVar, M: IntVar, K: IntVar](
    x1: _ShapedArrayLike[[N]],
    x2: _ShapedArrayLike[[M]],
    x3: _ShapedArrayLike[[K]],
    /,
    *,
    copy: bool = True,
    sparse: Literal[False] = False,
    indexing: Literal["ij"],
) -> tuple[_Array[[N, M, K]], _Array[[N, M, K]], _Array[[N, M, K]]]: ...
@overload
def meshgrid(
    *xi: _ArrayLike[Any], copy: bool = True, sparse: bool = False, indexing: str = "xy"
) -> tuple[_Array[IntTuple], ...]: ...

# `from_*` constructors
def from_dlpack(
    x: Any, /, *, device: _Device | _Sharding | None = None, copy: bool | None = None
) -> _Array[IntTuple]: ...
def frombuffer(
    buffer: Any, dtype: DTypeLike = float, count: int = -1, offset: int = 0
) -> _Array[IntTuple]: ...
def fromfile(*args: Any, **kwargs: Any) -> _Array[IntTuple]: ...
@overload
def fromfunction[N: IntVar](
    function: Callable[..., Any],
    shape: IntTuple[N],
    *,
    dtype: DTypeLike = float,
    **kwargs: Any,
) -> _Array[[N]]: ...
@overload
def fromfunction[N: IntVar, M: IntVar](
    function: Callable[..., Any],
    shape: IntTuple[N, M],
    *,
    dtype: DTypeLike = float,
    **kwargs: Any,
) -> _Array[[N, M]]: ...
@overload
def fromfunction[N: IntVar, M: IntVar, K: IntVar](
    function: Callable[..., Any],
    shape: IntTuple[N, M, K],
    *,
    dtype: DTypeLike = float,
    **kwargs: Any,
) -> _Array[[N, M, K]]: ...
@overload
def fromfunction(
    function: Callable[..., Any],
    shape: Sequence[int],
    *,
    dtype: DTypeLike = float,
    **kwargs: Any,
) -> _Array[IntTuple]: ...
def fromiter(*args: Any, **kwargs: Any) -> _Array[IntTuple]: ...
def fromstring(
    string: str, dtype: DTypeLike = float, count: int = -1, *, sep: str
) -> _Array[IntTuple]: ...

# Window functions
@overload
def bartlett[N: IntVar](M: Int[N]) -> _Array[[N]]: ...
@overload
def bartlett(M: int) -> _Array[IntTuple]: ...
@overload
def blackman[N: IntVar](M: Int[N]) -> _Array[[N]]: ...
@overload
def blackman(M: int) -> _Array[IntTuple]: ...
@overload
def hamming[N: IntVar](M: Int[N]) -> _Array[[N]]: ...
@overload
def hamming(M: int) -> _Array[IntTuple]: ...
@overload
def hanning[N: IntVar](M: Int[N]) -> _Array[[N]]: ...
@overload
def hanning(M: int) -> _Array[IntTuple]: ...
@overload
def kaiser[N: IntVar](M: Int[N], beta: Any) -> _Array[[N]]: ...
@overload
def kaiser(M: int, beta: Any) -> _Array[IntTuple]: ...

# Shape-preserving elementwise unary functions.
def abs[Shape: _Shape = []](x: _ArrayLike[Shape], /) -> _Array[Shape]: ...
def absolute[Shape: _Shape = []](x: _ArrayLike[Shape], /) -> _Array[Shape]: ...
def acos[Shape: _Shape = []](x: _ArrayLike[Shape], /) -> _Array[Shape]: ...
def acosh[Shape: _Shape = []](x: _ArrayLike[Shape], /) -> _Array[Shape]: ...
def angle[Shape: _Shape = []](
    z: _ArrayLike[Shape], deg: bool = False
) -> _Array[Shape]: ...
def arccos[Shape: _Shape = []](x: _ArrayLike[Shape], /) -> _Array[Shape]: ...
def arccosh[Shape: _Shape = []](x: _ArrayLike[Shape], /) -> _Array[Shape]: ...
def arcsin[Shape: _Shape = []](x: _ArrayLike[Shape], /) -> _Array[Shape]: ...
def arcsinh[Shape: _Shape = []](x: _ArrayLike[Shape], /) -> _Array[Shape]: ...
def arctan[Shape: _Shape = []](x: _ArrayLike[Shape], /) -> _Array[Shape]: ...
def arctanh[Shape: _Shape = []](x: _ArrayLike[Shape], /) -> _Array[Shape]: ...
def around[Shape: _Shape = []](
    a: _ArrayLike[Shape], decimals: int = 0
) -> _Array[Shape]: ...
def asin[Shape: _Shape = []](x: _ArrayLike[Shape], /) -> _Array[Shape]: ...
def asinh[Shape: _Shape = []](x: _ArrayLike[Shape], /) -> _Array[Shape]: ...
def atan[Shape: _Shape = []](x: _ArrayLike[Shape], /) -> _Array[Shape]: ...
def atanh[Shape: _Shape = []](x: _ArrayLike[Shape], /) -> _Array[Shape]: ...
def bitwise_count[Shape: _Shape = []](x: _ArrayLike[Shape], /) -> _Array[Shape]: ...
def bitwise_invert[Shape: _Shape = []](x: _ArrayLike[Shape], /) -> _Array[Shape]: ...
def bitwise_not[Shape: _Shape = []](x: _ArrayLike[Shape], /) -> _Array[Shape]: ...
def cbrt[Shape: _Shape = []](x: _ArrayLike[Shape], /) -> _Array[Shape]: ...
def ceil[Shape: _Shape = []](x: _ArrayLike[Shape], /) -> _Array[Shape]: ...
def conj[Shape: _Shape = []](x: _ArrayLike[Shape], /) -> _Array[Shape]: ...
def conjugate[Shape: _Shape = []](x: _ArrayLike[Shape], /) -> _Array[Shape]: ...
def cos[Shape: _Shape = []](x: _ArrayLike[Shape], /) -> _Array[Shape]: ...
def cosh[Shape: _Shape = []](x: _ArrayLike[Shape], /) -> _Array[Shape]: ...
def deg2rad[Shape: _Shape = []](x: _ArrayLike[Shape], /) -> _Array[Shape]: ...
def degrees[Shape: _Shape = []](x: _ArrayLike[Shape], /) -> _Array[Shape]: ...
def exp[Shape: _Shape = []](x: _ArrayLike[Shape], /) -> _Array[Shape]: ...
def exp2[Shape: _Shape = []](x: _ArrayLike[Shape], /) -> _Array[Shape]: ...
def expm1[Shape: _Shape = []](x: _ArrayLike[Shape], /) -> _Array[Shape]: ...
def fabs[Shape: _Shape = []](x: _ArrayLike[Shape], /) -> _Array[Shape]: ...
def floor[Shape: _Shape = []](x: _ArrayLike[Shape], /) -> _Array[Shape]: ...
def frexp[Shape: _Shape = []](
    x: _ArrayLike[Shape], /
) -> tuple[_Array[Shape], _Array[Shape]]: ...
def i0[Shape: _Shape = []](x: _ArrayLike[Shape], /) -> _Array[Shape]: ...
def imag[Shape: _Shape = []](val: _ArrayLike[Shape], /) -> _Array[Shape]: ...
def invert[Shape: _Shape = []](x: _ArrayLike[Shape], /) -> _Array[Shape]: ...
def log[Shape: _Shape = []](x: _ArrayLike[Shape], /) -> _Array[Shape]: ...
def log10[Shape: _Shape = []](x: _ArrayLike[Shape], /) -> _Array[Shape]: ...
def log1p[Shape: _Shape = []](x: _ArrayLike[Shape], /) -> _Array[Shape]: ...
def log2[Shape: _Shape = []](x: _ArrayLike[Shape], /) -> _Array[Shape]: ...
def modf[Shape: _Shape = []](
    x: _ArrayLike[Shape], /
) -> tuple[_Array[Shape], _Array[Shape]]: ...
def negative[Shape: _Shape = []](x: _ArrayLike[Shape], /) -> _Array[Shape]: ...
def positive[Shape: _Shape = []](x: _ArrayLike[Shape], /) -> _Array[Shape]: ...
def rad2deg[Shape: _Shape = []](x: _ArrayLike[Shape], /) -> _Array[Shape]: ...
def radians[Shape: _Shape = []](x: _ArrayLike[Shape], /) -> _Array[Shape]: ...
def real[Shape: _Shape = []](val: _ArrayLike[Shape], /) -> _Array[Shape]: ...
def reciprocal[Shape: _Shape = []](x: _ArrayLike[Shape], /) -> _Array[Shape]: ...
def rint[Shape: _Shape = []](x: _ArrayLike[Shape], /) -> _Array[Shape]: ...
def round[Shape: _Shape = []](
    a: _ArrayLike[Shape], decimals: int = 0
) -> _Array[Shape]: ...
def sign[Shape: _Shape = []](x: _ArrayLike[Shape], /) -> _Array[Shape]: ...
def signbit[Shape: _Shape = []](x: _ArrayLike[Shape], /) -> _Array[Shape]: ...
def sin[Shape: _Shape = []](x: _ArrayLike[Shape], /) -> _Array[Shape]: ...
def sinc[Shape: _Shape = []](x: _ArrayLike[Shape], /) -> _Array[Shape]: ...
def sinh[Shape: _Shape = []](x: _ArrayLike[Shape], /) -> _Array[Shape]: ...
def spacing[Shape: _Shape = []](x: _ArrayLike[Shape], /) -> _Array[Shape]: ...
def sqrt[Shape: _Shape = []](x: _ArrayLike[Shape], /) -> _Array[Shape]: ...
def square[Shape: _Shape = []](x: _ArrayLike[Shape], /) -> _Array[Shape]: ...
def tan[Shape: _Shape = []](x: _ArrayLike[Shape], /) -> _Array[Shape]: ...
def tanh[Shape: _Shape = []](x: _ArrayLike[Shape], /) -> _Array[Shape]: ...
def trunc[Shape: _Shape = []](x: _ArrayLike[Shape], /) -> _Array[Shape]: ...
def unwrap[Shape: _Shape = []](
    p: _ArrayLike[Shape],
    discont: Any = None,
    axis: int = -1,
    period: Any = ...,
) -> _Array[Shape]: ...

# Broadcasting elementwise binary functions. Each takes a scalar in either
# position as well as an array: rejecting `jnp.add(a, 1)` would flag valid code.
def add[Shape1: _Shape = [], Shape2: _Shape = []](
    x1: _ArrayLike[Shape1], x2: _ArrayLike[Shape2], /
) -> _Array[broadcast(Shape1, Shape2)]: ...
def arctan2[Shape1: _Shape = [], Shape2: _Shape = []](
    x1: _ArrayLike[Shape1], x2: _ArrayLike[Shape2], /
) -> _Array[broadcast(Shape1, Shape2)]: ...
def atan2[Shape1: _Shape = [], Shape2: _Shape = []](
    x1: _ArrayLike[Shape1], x2: _ArrayLike[Shape2], /
) -> _Array[broadcast(Shape1, Shape2)]: ...
def bitwise_and[Shape1: _Shape = [], Shape2: _Shape = []](
    x1: _ArrayLike[Shape1], x2: _ArrayLike[Shape2], /
) -> _Array[broadcast(Shape1, Shape2)]: ...
def bitwise_left_shift[Shape1: _Shape = [], Shape2: _Shape = []](
    x1: _ArrayLike[Shape1], x2: _ArrayLike[Shape2], /
) -> _Array[broadcast(Shape1, Shape2)]: ...
def bitwise_or[Shape1: _Shape = [], Shape2: _Shape = []](
    x1: _ArrayLike[Shape1], x2: _ArrayLike[Shape2], /
) -> _Array[broadcast(Shape1, Shape2)]: ...
def bitwise_right_shift[Shape1: _Shape = [], Shape2: _Shape = []](
    x1: _ArrayLike[Shape1], x2: _ArrayLike[Shape2], /
) -> _Array[broadcast(Shape1, Shape2)]: ...
def bitwise_xor[Shape1: _Shape = [], Shape2: _Shape = []](
    x1: _ArrayLike[Shape1], x2: _ArrayLike[Shape2], /
) -> _Array[broadcast(Shape1, Shape2)]: ...
def copysign[Shape1: _Shape = [], Shape2: _Shape = []](
    x1: _ArrayLike[Shape1], x2: _ArrayLike[Shape2], /
) -> _Array[broadcast(Shape1, Shape2)]: ...
def divide[Shape1: _Shape = [], Shape2: _Shape = []](
    x1: _ArrayLike[Shape1], x2: _ArrayLike[Shape2], /
) -> _Array[broadcast(Shape1, Shape2)]: ...
def divmod[Shape1: _Shape = [], Shape2: _Shape = []](
    x1: _ArrayLike[Shape1], x2: _ArrayLike[Shape2], /
) -> tuple[_Array[broadcast(Shape1, Shape2)], _Array[broadcast(Shape1, Shape2)]]: ...
def float_power[Shape1: _Shape = [], Shape2: _Shape = []](
    x1: _ArrayLike[Shape1], x2: _ArrayLike[Shape2], /
) -> _Array[broadcast(Shape1, Shape2)]: ...
def floor_divide[Shape1: _Shape = [], Shape2: _Shape = []](
    x1: _ArrayLike[Shape1], x2: _ArrayLike[Shape2], /
) -> _Array[broadcast(Shape1, Shape2)]: ...
def fmod[Shape1: _Shape = [], Shape2: _Shape = []](
    x1: _ArrayLike[Shape1], x2: _ArrayLike[Shape2], /
) -> _Array[broadcast(Shape1, Shape2)]: ...
def gcd[Shape1: _Shape = [], Shape2: _Shape = []](
    x1: _ArrayLike[Shape1], x2: _ArrayLike[Shape2], /
) -> _Array[broadcast(Shape1, Shape2)]: ...
def heaviside[Shape1: _Shape = [], Shape2: _Shape = []](
    x1: _ArrayLike[Shape1], x2: _ArrayLike[Shape2], /
) -> _Array[broadcast(Shape1, Shape2)]: ...
def hypot[Shape1: _Shape = [], Shape2: _Shape = []](
    x1: _ArrayLike[Shape1], x2: _ArrayLike[Shape2], /
) -> _Array[broadcast(Shape1, Shape2)]: ...
def lcm[Shape1: _Shape = [], Shape2: _Shape = []](
    x1: _ArrayLike[Shape1], x2: _ArrayLike[Shape2], /
) -> _Array[broadcast(Shape1, Shape2)]: ...
def ldexp[Shape1: _Shape = [], Shape2: _Shape = []](
    x1: _ArrayLike[Shape1], x2: _ArrayLike[Shape2], /
) -> _Array[broadcast(Shape1, Shape2)]: ...
def left_shift[Shape1: _Shape = [], Shape2: _Shape = []](
    x1: _ArrayLike[Shape1], x2: _ArrayLike[Shape2], /
) -> _Array[broadcast(Shape1, Shape2)]: ...
def logaddexp[Shape1: _Shape = [], Shape2: _Shape = []](
    x1: _ArrayLike[Shape1], x2: _ArrayLike[Shape2], /
) -> _Array[broadcast(Shape1, Shape2)]: ...
def logaddexp2[Shape1: _Shape = [], Shape2: _Shape = []](
    x1: _ArrayLike[Shape1], x2: _ArrayLike[Shape2], /
) -> _Array[broadcast(Shape1, Shape2)]: ...
@overload
def cross[
    Axis: Flag[int],
    Shape1: _Shape = [],
    Shape2: _Shape = [],
](
    a: _ArrayLike[Shape1],
    b: _ArrayLike[Shape2],
    /,
    axisa: int = -1,
    axisb: int = -1,
    axisc: int = -1,
    *,
    axis: Axis,
) -> _Array[cross_axis_shape(Shape1, Shape2, Axis)]: ...
@overload
def cross[
    Shape1: _Shape = [],
    Shape2: _Shape = [],
    AxisA: Flag[int] = -1,
    AxisB: Flag[int] = -1,
    AxisC: Flag[int] = -1,
](
    a: _ArrayLike[Shape1],
    b: _ArrayLike[Shape2],
    /,
    axisa: AxisA = -1,
    axisb: AxisB = -1,
    axisc: AxisC = -1,
    axis: None = None,
) -> _Array[cross_axes_shape(Shape1, Shape2, AxisA, AxisB, AxisC)]: ...
@overload
def cross(
    a: _ArrayLike[Any],
    b: _ArrayLike[Any],
    /,
    axisa: int = -1,
    axisb: int = -1,
    axisc: int = -1,
    axis: int | None = None,
) -> _Array[IntTuple]: ...
@overload
def diagonal[
    Shape: _Shape = [],
    Offset: Flag[int] = 0,
    Axis1: Flag[int] = 0,
    Axis2: Flag[int] = 1,
](
    a: _ArrayLike[Shape],
    offset: Offset = 0,
    axis1: Axis1 = 0,
    axis2: Axis2 = 1,
) -> _Array[diagonal_shape(Shape, Offset, Axis1, Axis2)]: ...
@overload
def diagonal(
    a: _ArrayLike[Any],
    offset: int = 0,
    axis1: int = 0,
    axis2: int = 1,
) -> _Array[IntTuple]: ...
def dot[LeftShape: _Shape = [], RightShape: _Shape = []](
    a: _ArrayLike[LeftShape],
    b: _ArrayLike[RightShape],
    *,
    precision: Any = None,
    preferred_element_type: Any = None,
    out_sharding: _NamedSharding | _PartitionSpec | None = None,
) -> _Array[dot_shape(LeftShape, RightShape)]: ...
@overload
def einsum[Spec: Flag[str], Shapes: IntTuples](
    subscripts: Spec,
    /,
    *operands: Unpack[MapIntTuples[lambda S: _Array[S], Shapes]],
    out: None = None,
    optimize: str | bool | Sequence[tuple[int, ...]] = "auto",
    precision: Any = None,
    preferred_element_type: Any = None,
    _dot_general: Any = ...,
    out_sharding: _NamedSharding | _PartitionSpec | None = None,
) -> _Array[einsum_shape(Spec, Shapes)]: ...
@overload
def einsum(
    subscripts: str,
    /,
    *operands: _ArrayLike[Any] | Sequence[Any],
    out: None = None,
    optimize: str | bool | Sequence[tuple[int, ...]] = "auto",
    precision: Any = None,
    preferred_element_type: Any = None,
    _dot_general: Any = ...,
    out_sharding: _NamedSharding | _PartitionSpec | None = None,
) -> _Array[IntTuple]: ...
def einsum_path(
    subscripts: str,
    /,
    *operands: _ArrayLike[Any] | Sequence[Any],
    optimize: bool | str | Sequence[tuple[int, ...]] = "auto",
) -> tuple[list[tuple[int, ...]], Any]: ...
def inner[LeftShape: _Shape = [], RightShape: _Shape = []](
    a: _ArrayLike[LeftShape],
    b: _ArrayLike[RightShape],
    *,
    precision: Any = None,
    preferred_element_type: Any = None,
) -> _Array[inner_shape(LeftShape, RightShape)]: ...
def kron[AShape: _Shape = [], BShape: _Shape = []](
    a: _ArrayLike[AShape],
    b: _ArrayLike[BShape],
) -> _Array[kron_shape(AShape, BShape)]: ...
def matmul[LeftShape: _Shape, RightShape: _Shape](
    a: _ShapedArrayLike[LeftShape], b: _ShapedArrayLike[RightShape]
) -> _Array[matmul_shape(LeftShape, RightShape)]: ...
def matvec[LeftShape: _Shape, RightShape: _Shape](
    x1: _ShapedArrayLike[LeftShape],
    x2: _ShapedArrayLike[RightShape],
    /,
) -> _Array[matvec_shape(LeftShape, RightShape)]: ...
@overload
def outer[M: IntVar, N: IntVar](
    a: _ShapedArrayLike[[M]],
    b: _ShapedArrayLike[[N]],
    out: None = None,
) -> _Array[[M, N]]: ...
@overload
def outer(
    a: _ArrayLike[Any] | Sequence[Any],
    b: _ArrayLike[Any] | Sequence[Any],
    out: None = None,
) -> _Array[IntTuple]: ...
def tensordot[
    Left: _Shape = [],
    Right: _Shape = [],
    Axes: Flag[_Axis] = 2,
](
    a: _ArrayLike[Left],
    b: _ArrayLike[Right],
    axes: Axes = 2,
    *,
    precision: Any = None,
    preferred_element_type: Any = None,
    out_sharding: _NamedSharding | _PartitionSpec | None = None,
) -> _Array[tensordot_shape(Left, Right, Axes)]: ...
@overload
def trace[
    Shape: _Shape = [],
    Offset: Flag[int] = 0,
    Axis1: Flag[int] = 0,
    Axis2: Flag[int] = 1,
](
    a: _ArrayLike[Shape],
    offset: Offset = 0,
    axis1: Axis1 = 0,
    axis2: Axis2 = 1,
    dtype: DTypeLike | None = None,
    out: None = None,
) -> _Array[trace_shape(Shape, Offset, Axis1, Axis2)]: ...
@overload
def trace(
    a: _ArrayLike[Any],
    offset: int = 0,
    axis1: int = 0,
    axis2: int = 1,
    dtype: DTypeLike | None = None,
    out: None = None,
) -> _Array[IntTuple]: ...
def vdot[Shape1: _Shape = [], Shape2: _Shape = []](
    a: _ArrayLike[Shape1],
    b: _ArrayLike[Shape2],
    *,
    precision: Any = None,
    preferred_element_type: Any = None,
) -> _Array[[]]: ...
def vecdot[Shape1: _Shape, Shape2: _Shape, Axis: Flag[_Axis] = -1](
    x1: _ShapedArrayLike[Shape1],
    x2: _ShapedArrayLike[Shape2],
    /,
    *,
    axis: Axis = -1,
    precision: Any = None,
    preferred_element_type: Any = None,
) -> _Array[reduce_shape(broadcast(Shape1, Shape2), Axis, False)]: ...
def vecmat[LeftShape: _Shape, RightShape: _Shape](
    x1: _ShapedArrayLike[LeftShape],
    x2: _ShapedArrayLike[RightShape],
    /,
) -> _Array[vecmat_shape(LeftShape, RightShape)]: ...
def maximum[Shape1: _Shape = [], Shape2: _Shape = []](
    x1: _ArrayLike[Shape1], x2: _ArrayLike[Shape2], /
) -> _Array[broadcast(Shape1, Shape2)]: ...
def minimum[Shape1: _Shape = [], Shape2: _Shape = []](
    x1: _ArrayLike[Shape1], x2: _ArrayLike[Shape2], /
) -> _Array[broadcast(Shape1, Shape2)]: ...
def mod[Shape1: _Shape = [], Shape2: _Shape = []](
    x1: _ArrayLike[Shape1], x2: _ArrayLike[Shape2], /
) -> _Array[broadcast(Shape1, Shape2)]: ...
def multiply[Shape1: _Shape = [], Shape2: _Shape = []](
    x1: _ArrayLike[Shape1], x2: _ArrayLike[Shape2], /
) -> _Array[broadcast(Shape1, Shape2)]: ...
def nextafter[Shape1: _Shape = [], Shape2: _Shape = []](
    x1: _ArrayLike[Shape1], x2: _ArrayLike[Shape2], /
) -> _Array[broadcast(Shape1, Shape2)]: ...
def pow[Shape1: _Shape = [], Shape2: _Shape = []](
    x1: _ArrayLike[Shape1], x2: _ArrayLike[Shape2], /
) -> _Array[broadcast(Shape1, Shape2)]: ...
def power[Shape1: _Shape = [], Shape2: _Shape = []](
    x1: _ArrayLike[Shape1], x2: _ArrayLike[Shape2], /
) -> _Array[broadcast(Shape1, Shape2)]: ...
def remainder[Shape1: _Shape = [], Shape2: _Shape = []](
    x1: _ArrayLike[Shape1], x2: _ArrayLike[Shape2], /
) -> _Array[broadcast(Shape1, Shape2)]: ...
def right_shift[Shape1: _Shape = [], Shape2: _Shape = []](
    x1: _ArrayLike[Shape1], x2: _ArrayLike[Shape2], /
) -> _Array[broadcast(Shape1, Shape2)]: ...
def subtract[Shape1: _Shape = [], Shape2: _Shape = []](
    x1: _ArrayLike[Shape1], x2: _ArrayLike[Shape2], /
) -> _Array[broadcast(Shape1, Shape2)]: ...
def true_divide[Shape1: _Shape = [], Shape2: _Shape = []](
    x1: _ArrayLike[Shape1], x2: _ArrayLike[Shape2], /
) -> _Array[broadcast(Shape1, Shape2)]: ...
@overload
def transpose[Shape: _Shape = []](
    a: _ArrayLike[Shape], axes: None = None
) -> _Array[reverse_shape(Shape)]: ...
@overload
def transpose[Axes: Flag[_Axis], Shape: _Shape = []](
    a: _ArrayLike[Shape], axes: Axes = None
) -> _Array[permute_shape(Shape, Axes)]: ...
@overload
def transpose[Shape: _Shape = []](
    a: _ArrayLike[Shape], axes: Sequence[int]
) -> _Array[IntTuple]: ...
@overload
def permute_dims[Axes: Flag[_Axis], Shape: _Shape = []](
    a: _ArrayLike[Shape], /, axes: Axes
) -> _Array[permute_shape(Shape, Axes)]: ...
@overload
def permute_dims[Shape: _Shape = []](
    a: _ArrayLike[Shape], /, axes: Sequence[int]
) -> _Array[IntTuple]: ...
def matrix_transpose[Batch: IntTuple, M: IntVar, N: IntVar](
    x: _ShapedArrayLike[[*Elements[Batch], M, N]],
    /,
) -> _Array[[*Elements[Batch], N, M]]: ...

# A single int or tuple, matching JAX: the free function is not variadic, so
# `jnp.reshape(a, 2, 3)` is an error there. `Array.reshape` is the variadic one.
@overload
def reshape[NewShape: Flag[_NewShape], Shape: _Shape = []](
    a: _ArrayLike[Shape],
    shape: NewShape,
    order: str = ...,
    *,
    copy: bool | None = ...,
    out_sharding: _NamedSharding | _PartitionSpec | None = ...,
) -> _Array[reshape_shape(Shape, NewShape)]: ...
@overload
def reshape[NewShape: _Shape](
    a: _ArrayLike[Any],
    shape: NewShape,
    order: str = ...,
    *,
    copy: bool | None = ...,
    out_sharding: _NamedSharding | _PartitionSpec | None = ...,
) -> _Array[NewShape]: ...
@overload
def reshape(
    a: _ArrayLike[Any],
    shape: Sequence[int],
    order: str = ...,
    *,
    copy: bool | None = ...,
    out_sharding: _NamedSharding | _PartitionSpec | None = ...,
) -> _Array[IntTuple]: ...
def ravel[Shape: _Shape = []](
    a: _ArrayLike[Shape],
    order: str = "C",
) -> _Array[ravel_shape(Shape)]: ...
@overload
def squeeze[Shape: _Shape = [], Axis: Flag[_Axis] = None](
    a: _ArrayLike[Shape],
    axis: Axis = None,
) -> _Array[squeeze_shape(Shape, Axis)]: ...
@overload
def squeeze(
    a: _ArrayLike[Any],
    axis: Sequence[int] | None = None,
) -> _Array[IntTuple]: ...
@overload
def expand_dims[Axis: Flag[int], Shape: _Shape = []](
    a: _ArrayLike[Shape],
    axis: Axis,
) -> _Array[expand_dims_shape(Shape, Axis)]: ...
@overload
def expand_dims(
    a: _ArrayLike[Any],
    axis: int | Sequence[int],
) -> _Array[IntTuple]: ...
@overload
def broadcast_to[TargetShape: Flag[_NewShape], Shape: _Shape = []](
    array: _ArrayLike[Shape],
    shape: TargetShape,
) -> _Array[broadcast_to_shape(Shape, TargetShape)]: ...
@overload
def broadcast_to(
    array: _ArrayLike[Any],
    shape: Sequence[int] | int,
) -> _Array[IntTuple]: ...
@overload
def broadcast_arrays[Shape: _Shape = []](
    a: _ArrayLike[Shape],
    /,
) -> tuple[_Array[Shape]]: ...
@overload
def broadcast_arrays[Shape1: _Shape = [], Shape2: _Shape = []](
    a1: _ArrayLike[Shape1],
    a2: _ArrayLike[Shape2],
    /,
) -> tuple[_Array[broadcast(Shape1, Shape2)], _Array[broadcast(Shape1, Shape2)]]: ...
@overload
def broadcast_arrays(*args: _ArrayLike[Any]) -> tuple[_Array[IntTuple], ...]: ...
def broadcast_shapes(*shapes: Sequence[int]) -> tuple[int, ...]: ...
@overload
def concatenate[Shapes: IntTuples, Axis: Flag[int] = 0](
    arrays: MapIntTuples[lambda S: _Array[S], Shapes],
    axis: Axis = 0,
    dtype: DTypeLike | None = None,
) -> _Array[concatenate_shape(Shapes, Axis)]: ...
@overload
def concatenate(
    arrays: Sequence[_ArrayLike[Any]] | _ArrayLike[Any],
    axis: int | None = 0,
    dtype: DTypeLike | None = None,
) -> _Array[IntTuple]: ...

concat = concatenate

@overload
def append[Shape1: _Shape = [], Shape2: _Shape = [], Axis: Flag[int | None] = None](
    arr: _ArrayLike[Shape1],
    values: _ArrayLike[Shape2],
    axis: Axis = None,
) -> _Array[append_shape(Shape1, Shape2, Axis)]: ...
@overload
def append(
    arr: _ArrayLike[Any],
    values: _ArrayLike[Any],
    axis: int | None = None,
) -> _Array[IntTuple]: ...
@overload
def stack[Shapes: IntTuples, Axis: Flag[int] = 0](
    arrays: MapIntTuples[lambda S: _Array[S], Shapes],
    axis: Axis = 0,
    dtype: DTypeLike | None = None,
    *,
    out: Any = None,
) -> _Array[stack_shape(Shapes, Axis)]: ...
@overload
def stack(
    arrays: Sequence[_ArrayLike[Any]] | _ArrayLike[Any],
    axis: int = 0,
    dtype: DTypeLike | None = None,
    *,
    out: Any = None,
) -> _Array[IntTuple]: ...
@overload
def vstack[Shapes: IntTuples](
    tup: MapIntTuples[lambda S: _Array[S], Shapes],
    *,
    dtype: DTypeLike | None = None,
) -> _Array[vstack_shape(Shapes)]: ...
@overload
def vstack(
    tup: Sequence[_ArrayLike[Any]] | _ArrayLike[Any],
    *,
    dtype: DTypeLike | None = None,
) -> _Array[IntTuple]: ...
@overload
def hstack[Shapes: IntTuples](
    tup: MapIntTuples[lambda S: _Array[S], Shapes],
    *,
    dtype: DTypeLike | None = None,
) -> _Array[hstack_shape(Shapes)]: ...
@overload
def hstack(
    tup: Sequence[_ArrayLike[Any]] | _ArrayLike[Any],
    *,
    dtype: DTypeLike | None = None,
) -> _Array[IntTuple]: ...
@overload
def column_stack[Shapes: IntTuples](
    tup: MapIntTuples[lambda S: _Array[S], Shapes],
) -> _Array[column_stack_shape(Shapes)]: ...
@overload
def column_stack(
    tup: Sequence[_ArrayLike[Any]] | _ArrayLike[Any],
) -> _Array[IntTuple]: ...
@overload
def dstack[Shapes: IntTuples](
    tup: MapIntTuples[lambda S: _Array[S], Shapes],
    *,
    dtype: DTypeLike | None = None,
) -> _Array[dstack_shape(Shapes)]: ...
@overload
def dstack(
    tup: Sequence[_ArrayLike[Any]] | _ArrayLike[Any],
    *,
    dtype: DTypeLike | None = None,
) -> _Array[IntTuple]: ...
def block(arrays: Any) -> _Array[IntTuple]: ...
def array_split[Batch: IntTuple, M: IntVar](
    ary: _ShapedArrayLike[[*Elements[Batch], M]],
    indices_or_sections: _ArrayLike[Any] | Sequence[int],
    axis: int = 0,
) -> list[_Array[IntTuple]]: ...
def split[Batch: IntTuple, M: IntVar](
    ary: _ShapedArrayLike[[*Elements[Batch], M]],
    indices_or_sections: _ArrayLike[Any] | Sequence[int],
    axis: int = 0,
) -> list[_Array[IntTuple]]: ...
def dsplit[Batch: IntTuple, M: IntVar, N: IntVar, P: IntVar](
    ary: _ShapedArrayLike[[*Elements[Batch], M, N, P]],
    indices_or_sections: _ArrayLike[Any] | Sequence[int],
) -> list[_Array[IntTuple]]: ...
def hsplit[Batch: IntTuple, M: IntVar](
    ary: _ShapedArrayLike[[*Elements[Batch], M]],
    indices_or_sections: _ArrayLike[Any] | Sequence[int],
) -> list[_Array[IntTuple]]: ...
def vsplit[Batch: IntTuple, M: IntVar](
    ary: _ShapedArrayLike[[*Elements[Batch], M]],
    indices_or_sections: _ArrayLike[Any] | Sequence[int],
) -> list[_Array[IntTuple]]: ...
def unstack[Batch: IntTuple, M: IntVar, Axis: Flag[int]](
    x: _ShapedArrayLike[[*Elements[Batch], M]],
    /,
    *,
    axis: Axis = 0,
) -> tuple[_Array[unstack_shape(Batch, Int[M], Axis)], ...]: ...
def pad(
    array: _ArrayLike[Any],
    pad_width: Any,
    mode: str | Callable[..., Any] = "constant",
    **kwargs: Any,
) -> _Array[IntTuple]: ...
@overload
def repeat[
    Repeats: Int,
    Axis: Flag[int | None],
    Shape: _Shape = [],
](
    a: _ArrayLike[Shape],
    repeats: Repeats,
    axis: Axis = None,
    *,
    total_repeat_length: None = None,
) -> _Array[repeat_shape(Shape, Repeats, Axis)]: ...
@overload
def repeat(
    a: _ArrayLike[Any],
    repeats: _ArrayLike[Any] | Sequence[int],
    axis: int | None = None,
    *,
    total_repeat_length: int | None = None,
) -> _Array[IntTuple]: ...
@overload
def resize(a: _ArrayLike[Any], new_shape: tuple[()]) -> _Array[[]]: ...
@overload
def resize[N: IntVar](a: _ArrayLike[Any], new_shape: Int[N]) -> _Array[[N]]: ...
@overload
def resize[Shape: _Shape](a: _ArrayLike[Any], new_shape: Shape) -> _Array[Shape]: ...
@overload
def resize(a: _ArrayLike[Any], new_shape: Sequence[int] | int) -> _Array[IntTuple]: ...
@overload
def tile[Repeats: Flag[int | tuple[int, ...]], Shape: _Shape = []](
    A: _ArrayLike[Shape],
    reps: Repeats,
) -> _Array[tile_shape(Shape, Repeats)]: ...
@overload
def tile(A: _ArrayLike[Any], reps: Sequence[int]) -> _Array[IntTuple]: ...
@overload
def swapaxes[Axis1: Flag[int], Axis2: Flag[int], Shape: _Shape = []](
    a: _ArrayLike[Shape],
    axis1: Axis1,
    axis2: Axis2,
) -> _Array[swapaxes_shape(Shape, Axis1, Axis2)]: ...
@overload
def swapaxes(
    a: _ArrayLike[Any],
    axis1: int,
    axis2: int,
) -> _Array[IntTuple]: ...
@overload
def moveaxis[Source: Flag[int], Destination: Flag[int], Shape: _Shape = []](
    a: _ArrayLike[Shape],
    source: Source,
    destination: Destination,
) -> _Array[moveaxis_shape(Shape, Source, Destination)]: ...
@overload
def moveaxis(
    a: _ArrayLike[Any],
    source: int | Sequence[int],
    destination: int | Sequence[int],
) -> _Array[IntTuple]: ...
@overload
def rollaxis[Axis: Flag[int], Shape: _Shape = [], Start: Flag[int] = 0](
    a: _ArrayLike[Shape],
    axis: Axis,
    start: Start = 0,
) -> _Array[rollaxis_shape(Shape, Axis, Start)]: ...
@overload
def rollaxis(
    a: _ArrayLike[Any],
    axis: int,
    start: int = 0,
) -> _Array[IntTuple]: ...
@overload
def flip[Shape: _Shape = [], Axis: Flag[_Axis] = None](
    m: _ArrayLike[Shape],
    axis: Axis = None,
) -> _Array[flip_shape(Shape, Axis)]: ...
@overload
def flip[Shape: _Shape = []](
    m: _ArrayLike[Shape],
    axis: Sequence[int] | None = None,
) -> _Array[Shape]: ...
def fliplr[Batch: IntTuple, M: IntVar, N: IntVar](
    m: _ShapedArrayLike[[*Elements[Batch], M, N]],
) -> _Array[[*Elements[Batch], M, N]]: ...
def flipud[Batch: IntTuple, M: IntVar](
    m: _ShapedArrayLike[[*Elements[Batch], M]],
) -> _Array[[*Elements[Batch], M]]: ...
@overload
def roll[Shape: _Shape = [], Axis: Flag[_Axis] = None](
    a: _ArrayLike[Shape],
    shift: Any,
    axis: Axis = None,
) -> _Array[roll_shape(Shape, Axis)]: ...
@overload
def roll[Shape: _Shape = []](
    a: _ArrayLike[Shape],
    shift: Any,
    axis: Sequence[int] | None = None,
) -> _Array[Shape]: ...
@overload
def rot90[Shape: _Shape = [], K: Flag[int] = 1, Axes: Flag[tuple[int, int]] = (0, 1)](
    m: _ArrayLike[Shape],
    k: K = 1,
    axes: Axes = (0, 1),
) -> _Array[rot90_shape(Shape, K, Axes)]: ...
@overload
def rot90(
    m: _ArrayLike[Any],
    k: int = 1,
    axes: tuple[int, int] = (0, 1),
) -> _Array[IntTuple]: ...
@overload
def atleast_1d[Shape: _Shape = []](
    ary: _ArrayLike[Shape], /
) -> _Array[atleast_1d_shape(Shape)]: ...
@overload
def atleast_1d(*arys: _ArrayLike[Any]) -> list[_Array[IntTuple]]: ...
@overload
def atleast_2d[Shape: _Shape = []](
    ary: _ArrayLike[Shape], /
) -> _Array[atleast_2d_shape(Shape)]: ...
@overload
def atleast_2d(*arys: _ArrayLike[Any]) -> list[_Array[IntTuple]]: ...
@overload
def atleast_3d[Shape: _Shape = []](
    ary: _ArrayLike[Shape], /
) -> _Array[atleast_3d_shape(Shape)]: ...
@overload
def atleast_3d(*arys: _ArrayLike[Any]) -> list[_Array[IntTuple]]: ...

# JAX accepts any integer sequence for an axis, but only a tuple is a Flag
# domain, so any other sequence yields a gradual shape. Rejecting it would flag
# valid code. The exact overload is declared first so that a tuple resolves to
# it rather than being absorbed by the fallback.
@overload
def sum[Axis: Flag[_Axis], KeepDims: Flag[bool], Shape: _Shape = []](
    a: _ArrayLike[Shape],
    axis: Axis = None,
    *,
    keepdims: KeepDims = False,
    dtype: DTypeLike | None = ...,
    out: Any = ...,
    initial: Any = ...,
    where: Any = ...,
    promote_integers: bool = ...,
) -> _Array[reduce_shape(Shape, Axis, KeepDims)]: ...
@overload
def sum[Shape: _Shape = []](
    a: _ArrayLike[Shape],
    axis: Sequence[int],
    *,
    keepdims: bool = False,
    dtype: DTypeLike | None = ...,
    out: Any = ...,
    initial: Any = ...,
    where: Any = ...,
    promote_integers: bool = ...,
) -> _Array[IntTuple]: ...
@overload
def prod[Axis: Flag[_Axis], KeepDims: Flag[bool], Shape: _Shape = []](
    a: _ArrayLike[Shape],
    axis: Axis = None,
    *,
    keepdims: KeepDims = False,
    dtype: DTypeLike | None = ...,
    out: Any = ...,
    initial: Any = ...,
    where: Any = ...,
    promote_integers: bool = ...,
) -> _Array[reduce_shape(Shape, Axis, KeepDims)]: ...
@overload
def prod[Shape: _Shape = []](
    a: _ArrayLike[Shape],
    axis: Sequence[int],
    *,
    keepdims: bool = False,
    dtype: DTypeLike | None = ...,
    out: Any = ...,
    initial: Any = ...,
    where: Any = ...,
    promote_integers: bool = ...,
) -> _Array[IntTuple]: ...
@overload
def mean[Axis: Flag[_Axis], KeepDims: Flag[bool], Shape: _Shape = []](
    a: _ArrayLike[Shape],
    axis: Axis = None,
    *,
    keepdims: KeepDims = False,
    dtype: DTypeLike | None = ...,
    out: Any = ...,
    initial: Any = ...,
    where: Any = ...,
    promote_integers: bool = ...,
) -> _Array[reduce_shape(Shape, Axis, KeepDims)]: ...
@overload
def mean[Shape: _Shape = []](
    a: _ArrayLike[Shape],
    axis: Sequence[int],
    *,
    keepdims: bool = False,
    dtype: DTypeLike | None = ...,
    out: Any = ...,
    initial: Any = ...,
    where: Any = ...,
    promote_integers: bool = ...,
) -> _Array[IntTuple]: ...
@overload
def max[Axis: Flag[_Axis], KeepDims: Flag[bool], Shape: _Shape = []](
    a: _ArrayLike[Shape],
    axis: Axis = None,
    *,
    keepdims: KeepDims = False,
    dtype: DTypeLike | None = ...,
    out: Any = ...,
    initial: Any = ...,
    where: Any = ...,
    promote_integers: bool = ...,
) -> _Array[reduce_shape(Shape, Axis, KeepDims)]: ...
@overload
def max[Shape: _Shape = []](
    a: _ArrayLike[Shape],
    axis: Sequence[int],
    *,
    keepdims: bool = False,
    dtype: DTypeLike | None = ...,
    out: Any = ...,
    initial: Any = ...,
    where: Any = ...,
    promote_integers: bool = ...,
) -> _Array[IntTuple]: ...
@overload
def min[Axis: Flag[_Axis], KeepDims: Flag[bool], Shape: _Shape = []](
    a: _ArrayLike[Shape],
    axis: Axis = None,
    *,
    keepdims: KeepDims = False,
    dtype: DTypeLike | None = ...,
    out: Any = ...,
    initial: Any = ...,
    where: Any = ...,
    promote_integers: bool = ...,
) -> _Array[reduce_shape(Shape, Axis, KeepDims)]: ...
@overload
def min[Shape: _Shape = []](
    a: _ArrayLike[Shape],
    axis: Sequence[int],
    *,
    keepdims: bool = False,
    dtype: DTypeLike | None = ...,
    out: Any = ...,
    initial: Any = ...,
    where: Any = ...,
    promote_integers: bool = ...,
) -> _Array[IntTuple]: ...

# Boolean reductions
@overload
def all[Axis: Flag[_Axis], KeepDims: Flag[bool], Shape: _Shape = []](
    a: _ArrayLike[Shape],
    axis: Axis = None,
    out: Any = None,
    keepdims: KeepDims = False,
    *,
    where: Any = None,
) -> _Array[reduce_shape(Shape, Axis, KeepDims)]: ...
@overload
def all[Shape: _Shape = []](
    a: _ArrayLike[Shape],
    axis: Sequence[int],
    out: Any = None,
    keepdims: bool = False,
    *,
    where: Any = None,
) -> _Array[IntTuple]: ...
@overload
def any[Axis: Flag[_Axis], KeepDims: Flag[bool], Shape: _Shape = []](
    a: _ArrayLike[Shape],
    axis: Axis = None,
    out: Any = None,
    keepdims: KeepDims = False,
    *,
    where: Any = None,
) -> _Array[reduce_shape(Shape, Axis, KeepDims)]: ...
@overload
def any[Shape: _Shape = []](
    a: _ArrayLike[Shape],
    axis: Sequence[int],
    out: Any = None,
    keepdims: bool = False,
    *,
    where: Any = None,
) -> _Array[IntTuple]: ...

# Count nonzero
@overload
def count_nonzero[Axis: Flag[_Axis], KeepDims: Flag[bool], Shape: _Shape = []](
    a: _ArrayLike[Shape],
    axis: Axis = None,
    keepdims: KeepDims = False,
) -> _Array[reduce_shape(Shape, Axis, KeepDims)]: ...
@overload
def count_nonzero[Shape: _Shape = []](
    a: _ArrayLike[Shape],
    axis: Sequence[int],
    keepdims: bool = False,
) -> _Array[IntTuple]: ...

# amax / amin aliases
@overload
def amax[Axis: Flag[_Axis], KeepDims: Flag[bool], Shape: _Shape = []](
    a: _ArrayLike[Shape],
    axis: Axis = None,
    out: Any = None,
    keepdims: KeepDims = False,
    initial: Any = None,
    where: Any = None,
) -> _Array[reduce_shape(Shape, Axis, KeepDims)]: ...
@overload
def amax[Shape: _Shape = []](
    a: _ArrayLike[Shape],
    axis: Sequence[int],
    out: Any = None,
    keepdims: bool = False,
    initial: Any = None,
    where: Any = None,
) -> _Array[IntTuple]: ...
@overload
def amin[Axis: Flag[_Axis], KeepDims: Flag[bool], Shape: _Shape = []](
    a: _ArrayLike[Shape],
    axis: Axis = None,
    out: Any = None,
    keepdims: KeepDims = False,
    initial: Any = None,
    where: Any = None,
) -> _Array[reduce_shape(Shape, Axis, KeepDims)]: ...
@overload
def amin[Shape: _Shape = []](
    a: _ArrayLike[Shape],
    axis: Sequence[int],
    out: Any = None,
    keepdims: bool = False,
    initial: Any = None,
    where: Any = None,
) -> _Array[IntTuple]: ...

# Standard deviation & variance
@overload
def std[Axis: Flag[_Axis], KeepDims: Flag[bool], Shape: _Shape = []](
    a: _ArrayLike[Shape],
    axis: Axis = None,
    dtype: DTypeLike | None = None,
    out: Any = None,
    ddof: int = 0,
    keepdims: KeepDims = False,
    *,
    where: Any = None,
    mean: Any = None,
    correction: Any = None,
) -> _Array[reduce_shape(Shape, Axis, KeepDims)]: ...
@overload
def std[Shape: _Shape = []](
    a: _ArrayLike[Shape],
    axis: Sequence[int],
    dtype: DTypeLike | None = None,
    out: Any = None,
    ddof: int = 0,
    keepdims: bool = False,
    *,
    where: Any = None,
    mean: Any = None,
    correction: Any = None,
) -> _Array[IntTuple]: ...
@overload
def var[Axis: Flag[_Axis], KeepDims: Flag[bool], Shape: _Shape = []](
    a: _ArrayLike[Shape],
    axis: Axis = None,
    dtype: DTypeLike | None = None,
    out: Any = None,
    ddof: int = 0,
    keepdims: KeepDims = False,
    *,
    where: Any = None,
    mean: Any = None,
    correction: Any = None,
) -> _Array[reduce_shape(Shape, Axis, KeepDims)]: ...
@overload
def var[Shape: _Shape = []](
    a: _ArrayLike[Shape],
    axis: Sequence[int],
    dtype: DTypeLike | None = None,
    out: Any = None,
    ddof: int = 0,
    keepdims: bool = False,
    *,
    where: Any = None,
    mean: Any = None,
    correction: Any = None,
) -> _Array[IntTuple]: ...

# Peak-to-peak (ptp)
@overload
def ptp[Axis: Flag[_Axis], KeepDims: Flag[bool], Shape: _Shape = []](
    a: _ArrayLike[Shape],
    axis: Axis = None,
    out: Any = None,
    keepdims: KeepDims = False,
) -> _Array[reduce_shape(Shape, Axis, KeepDims)]: ...
@overload
def ptp[Shape: _Shape = []](
    a: _ArrayLike[Shape],
    axis: Sequence[int],
    out: Any = None,
    keepdims: bool = False,
) -> _Array[IntTuple]: ...

# Median
@overload
def median[Axis: Flag[_Axis], KeepDims: Flag[bool], Shape: _Shape = []](
    a: _ArrayLike[Shape],
    axis: Axis = None,
    out: Any = None,
    overwrite_input: bool = False,
    keepdims: KeepDims = False,
) -> _Array[reduce_shape(Shape, Axis, KeepDims)]: ...
@overload
def median[Shape: _Shape = []](
    a: _ArrayLike[Shape],
    axis: Sequence[int],
    out: Any = None,
    overwrite_input: bool = False,
    keepdims: bool = False,
) -> _Array[IntTuple]: ...

# NaN-safe reductions
@overload
def nanmax[Axis: Flag[_Axis], KeepDims: Flag[bool], Shape: _Shape = []](
    a: _ArrayLike[Shape],
    axis: Axis = None,
    out: Any = None,
    keepdims: KeepDims = False,
    initial: Any = None,
    where: Any = None,
) -> _Array[reduce_shape(Shape, Axis, KeepDims)]: ...
@overload
def nanmax[Shape: _Shape = []](
    a: _ArrayLike[Shape],
    axis: Sequence[int],
    out: Any = None,
    keepdims: bool = False,
    initial: Any = None,
    where: Any = None,
) -> _Array[IntTuple]: ...
@overload
def nanmin[Axis: Flag[_Axis], KeepDims: Flag[bool], Shape: _Shape = []](
    a: _ArrayLike[Shape],
    axis: Axis = None,
    out: Any = None,
    keepdims: KeepDims = False,
    initial: Any = None,
    where: Any = None,
) -> _Array[reduce_shape(Shape, Axis, KeepDims)]: ...
@overload
def nanmin[Shape: _Shape = []](
    a: _ArrayLike[Shape],
    axis: Sequence[int],
    out: Any = None,
    keepdims: bool = False,
    initial: Any = None,
    where: Any = None,
) -> _Array[IntTuple]: ...
@overload
def nansum[Axis: Flag[_Axis], KeepDims: Flag[bool], Shape: _Shape = []](
    a: _ArrayLike[Shape],
    axis: Axis = None,
    dtype: DTypeLike | None = None,
    out: Any = None,
    keepdims: KeepDims = False,
    initial: Any = None,
    where: Any = None,
) -> _Array[reduce_shape(Shape, Axis, KeepDims)]: ...
@overload
def nansum[Shape: _Shape = []](
    a: _ArrayLike[Shape],
    axis: Sequence[int],
    dtype: DTypeLike | None = None,
    out: Any = None,
    keepdims: bool = False,
    initial: Any = None,
    where: Any = None,
) -> _Array[IntTuple]: ...
@overload
def nanprod[Axis: Flag[_Axis], KeepDims: Flag[bool], Shape: _Shape = []](
    a: _ArrayLike[Shape],
    axis: Axis = None,
    dtype: DTypeLike | None = None,
    out: Any = None,
    keepdims: KeepDims = False,
    initial: Any = None,
    where: Any = None,
) -> _Array[reduce_shape(Shape, Axis, KeepDims)]: ...
@overload
def nanprod[Shape: _Shape = []](
    a: _ArrayLike[Shape],
    axis: Sequence[int],
    dtype: DTypeLike | None = None,
    out: Any = None,
    keepdims: bool = False,
    initial: Any = None,
    where: Any = None,
) -> _Array[IntTuple]: ...
@overload
def nanmean[Axis: Flag[_Axis], KeepDims: Flag[bool], Shape: _Shape = []](
    a: _ArrayLike[Shape],
    axis: Axis = None,
    dtype: DTypeLike | None = None,
    out: Any = None,
    keepdims: KeepDims = False,
    where: Any = None,
) -> _Array[reduce_shape(Shape, Axis, KeepDims)]: ...
@overload
def nanmean[Shape: _Shape = []](
    a: _ArrayLike[Shape],
    axis: Sequence[int],
    dtype: DTypeLike | None = None,
    out: Any = None,
    keepdims: bool = False,
    where: Any = None,
) -> _Array[IntTuple]: ...
@overload
def nanstd[Axis: Flag[_Axis], KeepDims: Flag[bool], Shape: _Shape = []](
    a: _ArrayLike[Shape],
    axis: Axis = None,
    dtype: DTypeLike | None = None,
    out: Any = None,
    ddof: int = 0,
    keepdims: KeepDims = False,
    where: Any = None,
    mean: Any = None,
) -> _Array[reduce_shape(Shape, Axis, KeepDims)]: ...
@overload
def nanstd[Shape: _Shape = []](
    a: _ArrayLike[Shape],
    axis: Sequence[int],
    dtype: DTypeLike | None = None,
    out: Any = None,
    ddof: int = 0,
    keepdims: bool = False,
    where: Any = None,
    mean: Any = None,
) -> _Array[IntTuple]: ...
@overload
def nanvar[Axis: Flag[_Axis], KeepDims: Flag[bool], Shape: _Shape = []](
    a: _ArrayLike[Shape],
    axis: Axis = None,
    dtype: DTypeLike | None = None,
    out: Any = None,
    ddof: int = 0,
    keepdims: KeepDims = False,
    where: Any = None,
    mean: Any = None,
) -> _Array[reduce_shape(Shape, Axis, KeepDims)]: ...
@overload
def nanvar[Shape: _Shape = []](
    a: _ArrayLike[Shape],
    axis: Sequence[int],
    dtype: DTypeLike | None = None,
    out: Any = None,
    ddof: int = 0,
    keepdims: bool = False,
    where: Any = None,
    mean: Any = None,
) -> _Array[IntTuple]: ...
@overload
def nanmedian[Axis: Flag[_Axis], KeepDims: Flag[bool], Shape: _Shape = []](
    a: _ArrayLike[Shape],
    axis: Axis = None,
    out: Any = None,
    overwrite_input: bool = False,
    keepdims: KeepDims = False,
) -> _Array[reduce_shape(Shape, Axis, KeepDims)]: ...
@overload
def nanmedian[Shape: _Shape = []](
    a: _ArrayLike[Shape],
    axis: Sequence[int],
    out: Any = None,
    overwrite_input: bool = False,
    keepdims: bool = False,
) -> _Array[IntTuple]: ...

# Average
@overload
def average[Axis: Flag[_Axis], KeepDims: Flag[bool], Shape: _Shape = []](
    a: _ArrayLike[Shape],
    axis: Axis = None,
    weights: Any = None,
    returned: Literal[False] = False,
    keepdims: KeepDims = False,
) -> _Array[reduce_shape(Shape, Axis, KeepDims)]: ...
@overload
def average[Axis: Flag[_Axis], KeepDims: Flag[bool], Shape: _Shape = []](
    a: _ArrayLike[Shape],
    axis: Axis = None,
    weights: Any = None,
    returned: Literal[True] = ...,
    keepdims: KeepDims = False,
) -> tuple[
    _Array[reduce_shape(Shape, Axis, KeepDims)],
    _Array[reduce_shape(Shape, Axis, KeepDims)],
]: ...
@overload
def average[Shape: _Shape = []](
    a: _ArrayLike[Shape],
    axis: Sequence[int],
    weights: Any = None,
    returned: bool = False,
    keepdims: bool = False,
) -> _Array[IntTuple] | tuple[_Array[IntTuple], _Array[IntTuple]]: ...

# Arg reductions
@overload
def argmax[Axis: Flag[_Axis], KeepDims: Flag[bool], Shape: _Shape = []](
    a: _ArrayLike[Shape],
    axis: Axis = None,
    out: Any = None,
    keepdims: KeepDims = False,
) -> _Array[reduce_shape(Shape, Axis, KeepDims)]: ...
@overload
def argmax(
    a: _ArrayLike[Any],
    axis: int | None = None,
    out: Any = None,
    keepdims: bool | None = None,
) -> _Array[IntTuple]: ...
@overload
def argmin[Axis: Flag[_Axis], KeepDims: Flag[bool], Shape: _Shape = []](
    a: _ArrayLike[Shape],
    axis: Axis = None,
    out: Any = None,
    keepdims: KeepDims = False,
) -> _Array[reduce_shape(Shape, Axis, KeepDims)]: ...
@overload
def argmin(
    a: _ArrayLike[Any],
    axis: int | None = None,
    out: Any = None,
    keepdims: bool | None = None,
) -> _Array[IntTuple]: ...
@overload
def nanargmax[Axis: Flag[_Axis], KeepDims: Flag[bool], Shape: _Shape = []](
    a: _ArrayLike[Shape],
    axis: Axis = None,
    out: Any = None,
    keepdims: KeepDims = False,
) -> _Array[reduce_shape(Shape, Axis, KeepDims)]: ...
@overload
def nanargmax(
    a: _ArrayLike[Any],
    axis: int | None = None,
    out: Any = None,
    keepdims: bool | None = None,
) -> _Array[IntTuple]: ...
@overload
def nanargmin[Axis: Flag[_Axis], KeepDims: Flag[bool], Shape: _Shape = []](
    a: _ArrayLike[Shape],
    axis: Axis = None,
    out: Any = None,
    keepdims: KeepDims = False,
) -> _Array[reduce_shape(Shape, Axis, KeepDims)]: ...
@overload
def nanargmin(
    a: _ArrayLike[Any],
    axis: int | None = None,
    out: Any = None,
    keepdims: bool | None = None,
) -> _Array[IntTuple]: ...

# Cumulative operations
@overload
def cumsum[Shape: _Shape = []](
    a: _ArrayLike[Shape],
    axis: int,
    dtype: DTypeLike | None = None,
    out: Any = None,
) -> _Array[Shape]: ...
@overload
def cumsum(
    a: _ArrayLike[Any],
    axis: None = None,
    dtype: DTypeLike | None = None,
    out: Any = None,
) -> _Array[IntTuple]: ...
@overload
def cumprod[Shape: _Shape = []](
    a: _ArrayLike[Shape],
    axis: int,
    dtype: DTypeLike | None = None,
    out: Any = None,
) -> _Array[Shape]: ...
@overload
def cumprod(
    a: _ArrayLike[Any],
    axis: None = None,
    dtype: DTypeLike | None = None,
    out: Any = None,
) -> _Array[IntTuple]: ...
@overload
def cumulative_sum[Shape: _Shape = []](
    x: _ArrayLike[Shape],
    /,
    *,
    axis: int,
    dtype: DTypeLike | None = None,
    include_initial: Literal[False] = False,
) -> _Array[Shape]: ...
@overload
def cumulative_sum(
    x: _ArrayLike[Any],
    /,
    *,
    axis: int | None = None,
    dtype: DTypeLike | None = None,
    include_initial: bool = False,
) -> _Array[IntTuple]: ...
@overload
def cumulative_prod[Shape: _Shape = []](
    x: _ArrayLike[Shape],
    /,
    *,
    axis: int,
    dtype: DTypeLike | None = None,
    include_initial: Literal[False] = False,
) -> _Array[Shape]: ...
@overload
def cumulative_prod(
    x: _ArrayLike[Any],
    /,
    *,
    axis: int | None = None,
    dtype: DTypeLike | None = None,
    include_initial: bool = False,
) -> _Array[IntTuple]: ...
@overload
def nancumsum[Shape: _Shape = []](
    a: _ArrayLike[Shape],
    axis: int,
    dtype: DTypeLike | None = None,
    out: Any = None,
) -> _Array[Shape]: ...
@overload
def nancumsum(
    a: _ArrayLike[Any],
    axis: None = None,
    dtype: DTypeLike | None = None,
    out: Any = None,
) -> _Array[IntTuple]: ...
@overload
def nancumprod[Shape: _Shape = []](
    a: _ArrayLike[Shape],
    axis: int,
    dtype: DTypeLike | None = None,
    out: Any = None,
) -> _Array[Shape]: ...
@overload
def nancumprod(
    a: _ArrayLike[Any],
    axis: None = None,
    dtype: DTypeLike | None = None,
    out: Any = None,
) -> _Array[IntTuple]: ...

# Quantile & Percentile
@overload
def quantile[Axis: Flag[_Axis], KeepDims: Flag[bool], Shape: _Shape = []](
    a: _ArrayLike[Shape],
    q: int | float,
    axis: Axis = None,
    out: Any = None,
    overwrite_input: bool = False,
    method: str = "linear",
    keepdims: KeepDims = False,
    *,
    weights: Any = None,
) -> _Array[reduce_shape(Shape, Axis, KeepDims)]: ...
@overload
def quantile(
    a: _ArrayLike[Any],
    q: Any,
    axis: Any = None,
    out: Any = None,
    overwrite_input: bool = False,
    method: str = "linear",
    keepdims: bool = False,
    *,
    weights: Any = None,
) -> _Array[IntTuple]: ...
@overload
def percentile[Axis: Flag[_Axis], KeepDims: Flag[bool], Shape: _Shape = []](
    a: _ArrayLike[Shape],
    q: int | float,
    axis: Axis = None,
    out: Any = None,
    overwrite_input: bool = False,
    method: str = "linear",
    keepdims: KeepDims = False,
    *,
    weights: Any = None,
    out_sharding: _NamedSharding | _PartitionSpec | None = None,
) -> _Array[reduce_shape(Shape, Axis, KeepDims)]: ...
@overload
def percentile(
    a: _ArrayLike[Any],
    q: Any,
    axis: Any = None,
    out: Any = None,
    overwrite_input: bool = False,
    method: str = "linear",
    keepdims: bool = False,
    *,
    weights: Any = None,
    out_sharding: _NamedSharding | _PartitionSpec | None = None,
) -> _Array[IntTuple]: ...
@overload
def nanquantile[Axis: Flag[_Axis], KeepDims: Flag[bool], Shape: _Shape = []](
    a: _ArrayLike[Shape],
    q: int | float,
    axis: Axis = None,
    out: Any = None,
    overwrite_input: bool = False,
    method: str = "linear",
    keepdims: KeepDims = False,
    *,
    weights: Any = None,
) -> _Array[reduce_shape(Shape, Axis, KeepDims)]: ...
@overload
def nanquantile(
    a: _ArrayLike[Any],
    q: Any,
    axis: Any = None,
    out: Any = None,
    overwrite_input: bool = False,
    method: str = "linear",
    keepdims: bool = False,
    *,
    weights: Any = None,
) -> _Array[IntTuple]: ...
@overload
def nanpercentile[Axis: Flag[_Axis], KeepDims: Flag[bool], Shape: _Shape = []](
    a: _ArrayLike[Shape],
    q: int | float,
    axis: Axis = None,
    out: Any = None,
    overwrite_input: bool = False,
    method: str = "linear",
    keepdims: KeepDims = False,
    *,
    weights: Any = None,
) -> _Array[reduce_shape(Shape, Axis, KeepDims)]: ...
@overload
def nanpercentile(
    a: _ArrayLike[Any],
    q: Any,
    axis: Any = None,
    out: Any = None,
    overwrite_input: bool = False,
    method: str = "linear",
    keepdims: bool = False,
    *,
    weights: Any = None,
) -> _Array[IntTuple]: ...

# Differences & Calculus
def diff(
    a: _ArrayLike[Any],
    n: int = 1,
    axis: int = -1,
    prepend: Any = None,
    append: Any = None,
) -> _Array[IntTuple]: ...
def ediff1d(
    ary: _ArrayLike[Any],
    to_end: Any = None,
    to_begin: Any = None,
) -> _Array[IntTuple]: ...
@overload
def gradient(
    f: _ArrayLike[Any],
    *varargs: Any,
    axis: int,
    edge_order: int | None = None,
) -> _Array[IntTuple]: ...
@overload
def gradient(
    f: _ArrayLike[Any],
    *varargs: Any,
    axis: Sequence[int] | None = None,
    edge_order: int | None = None,
) -> list[_Array[IntTuple]]: ...
@overload
def trapezoid[Axis: Flag[_Axis], Shape: _Shape = []](
    y: _ArrayLike[Shape],
    x: Any = None,
    dx: Any = 1.0,
    axis: Axis = -1,
) -> _Array[reduce_shape(Shape, Axis, False)]: ...
@overload
def trapezoid(
    y: _ArrayLike[Any],
    x: Any = None,
    dx: Any = 1.0,
    axis: int = -1,
) -> _Array[IntTuple]: ...
def corrcoef(
    x: _ArrayLike[Any],
    y: Any = None,
    rowvar: bool = True,
    dtype: DTypeLike | None = None,
) -> _Array[IntTuple]: ...
def cov(
    m: _ArrayLike[Any],
    y: Any = None,
    rowvar: bool = True,
    bias: bool = False,
    ddof: int | None = None,
    fweights: Any = None,
    aweights: Any = None,
    dtype: DTypeLike | None = None,
) -> _Array[IntTuple]: ...

# Logic and Comparison
def allclose(
    a: _ArrayLike[Any],
    b: _ArrayLike[Any],
    rtol: Any = 1e-05,
    atol: Any = 1e-08,
    equal_nan: bool = False,
) -> _Array[[]]: ...
def array_equal(
    a1: _ArrayLike[Any], a2: _ArrayLike[Any], equal_nan: bool = False
) -> _Array[[]]: ...
def array_equiv(a1: _ArrayLike[Any], a2: _ArrayLike[Any]) -> _Array[[]]: ...
def equal[Shape1: _Shape = [], Shape2: _Shape = []](
    x1: _ArrayLike[Shape1], x2: _ArrayLike[Shape2], /
) -> _Array[broadcast(Shape1, Shape2)]: ...
def greater[Shape1: _Shape = [], Shape2: _Shape = []](
    x1: _ArrayLike[Shape1], x2: _ArrayLike[Shape2], /
) -> _Array[broadcast(Shape1, Shape2)]: ...
def greater_equal[Shape1: _Shape = [], Shape2: _Shape = []](
    x1: _ArrayLike[Shape1], x2: _ArrayLike[Shape2], /
) -> _Array[broadcast(Shape1, Shape2)]: ...
def isclose[Shape1: _Shape = [], Shape2: _Shape = []](
    a: _ArrayLike[Shape1],
    b: _ArrayLike[Shape2],
    rtol: Any = 1e-05,
    atol: Any = 1e-08,
    equal_nan: bool = False,
) -> _Array[broadcast(Shape1, Shape2)]: ...
def iscomplex[Shape: _Shape = []](x: _ArrayLike[Shape], /) -> _Array[Shape]: ...
def iscomplexobj(x: Any) -> bool: ...
def isfinite[Shape: _Shape = []](x: _ArrayLike[Shape], /) -> _Array[Shape]: ...
def isinf[Shape: _Shape = []](x: _ArrayLike[Shape], /) -> _Array[Shape]: ...
def isnan[Shape: _Shape = []](x: _ArrayLike[Shape], /) -> _Array[Shape]: ...
def isneginf[Shape: _Shape = []](
    x: _ArrayLike[Shape], /, out: Any = None
) -> _Array[Shape]: ...
def isposinf[Shape: _Shape = []](
    x: _ArrayLike[Shape], /, out: Any = None
) -> _Array[Shape]: ...
def isreal[Shape: _Shape = []](x: _ArrayLike[Shape], /) -> _Array[Shape]: ...
def isrealobj(x: Any) -> bool: ...
def isscalar(element: Any) -> bool: ...
def iterable(y: Any) -> bool: ...
def less[Shape1: _Shape = [], Shape2: _Shape = []](
    x1: _ArrayLike[Shape1], x2: _ArrayLike[Shape2], /
) -> _Array[broadcast(Shape1, Shape2)]: ...
def less_equal[Shape1: _Shape = [], Shape2: _Shape = []](
    x1: _ArrayLike[Shape1], x2: _ArrayLike[Shape2], /
) -> _Array[broadcast(Shape1, Shape2)]: ...
def logical_and[Shape1: _Shape = [], Shape2: _Shape = []](
    x1: _ArrayLike[Shape1], x2: _ArrayLike[Shape2], /
) -> _Array[broadcast(Shape1, Shape2)]: ...
def logical_not[Shape: _Shape = []](x: _ArrayLike[Shape], /) -> _Array[Shape]: ...
def logical_or[Shape1: _Shape = [], Shape2: _Shape = []](
    x1: _ArrayLike[Shape1], x2: _ArrayLike[Shape2], /
) -> _Array[broadcast(Shape1, Shape2)]: ...
def logical_xor[Shape1: _Shape = [], Shape2: _Shape = []](
    x1: _ArrayLike[Shape1], x2: _ArrayLike[Shape2], /
) -> _Array[broadcast(Shape1, Shape2)]: ...
def not_equal[Shape1: _Shape = [], Shape2: _Shape = []](
    x1: _ArrayLike[Shape1], x2: _ArrayLike[Shape2], /
) -> _Array[broadcast(Shape1, Shape2)]: ...

# Sorting and Partitioning
@overload
def sort[Shape: _Shape = [], Axis: Flag[int | None] = -1](
    a: _ArrayLike[Shape],
    axis: Axis = -1,
    *,
    kind: None = None,
    order: None = None,
    stable: bool = True,
    descending: bool = False,
) -> _Array[sort_shape(Shape, Axis)]: ...
@overload
def sort(
    a: _ArrayLike[Any],
    axis: int | None = -1,
    *,
    kind: None = None,
    order: None = None,
    stable: bool = True,
    descending: bool = False,
) -> _Array[IntTuple]: ...
@overload
def argsort[Shape: _Shape = [], Axis: Flag[int | None] = -1](
    a: _ArrayLike[Shape],
    axis: Axis = -1,
    *,
    kind: None = None,
    order: None = None,
    stable: bool = True,
    descending: bool = False,
    dtype: DTypeLike | None = None,
) -> _Array[sort_shape(Shape, Axis)]: ...
@overload
def argsort(
    a: _ArrayLike[Any],
    axis: int | None = -1,
    *,
    kind: None = None,
    order: None = None,
    stable: bool = True,
    descending: bool = False,
    dtype: DTypeLike | None = None,
) -> _Array[IntTuple]: ...
def sort_complex[Shape: _Shape = []](a: _ArrayLike[Shape]) -> _Array[Shape]: ...
@overload
def partition[Shape: _Shape = [], Axis: Flag[int] = -1](
    a: _ArrayLike[Shape],
    kth: int | Sequence[int],
    axis: Axis = -1,
) -> _Array[sort_shape(Shape, Axis)]: ...
@overload
def partition(
    a: _ArrayLike[Any],
    kth: int | Sequence[int],
    axis: int = -1,
) -> _Array[IntTuple]: ...
@overload
def argpartition[Shape: _Shape = [], Axis: Flag[int] = -1](
    a: _ArrayLike[Shape],
    kth: int | Sequence[int],
    axis: Axis = -1,
) -> _Array[sort_shape(Shape, Axis)]: ...
@overload
def argpartition(
    a: _ArrayLike[Any],
    kth: int | Sequence[int],
    axis: int = -1,
) -> _Array[IntTuple]: ...
@overload
def lexsort[Shape: _Shape = []](
    keys: Sequence[_ArrayLike[Shape]],
    axis: int = -1,
) -> _Array[Shape]: ...
@overload
def lexsort(
    keys: _ArrayLike[Any],
    axis: int = -1,
) -> _Array[IntTuple]: ...
@overload
def top_k[K: Flag[int], Shape: _Shape = [], Axis: Flag[int] = -1](
    a: _ArrayLike[Shape],
    k: K,
    /,
    *,
    axis: Axis = -1,
    mode: str = "largest",
    sorted: bool = True,
) -> tuple[
    _Array[top_k_shape(Shape, K, Axis)],
    _Array[top_k_shape(Shape, K, Axis)],
]: ...
@overload
def top_k(
    a: _ArrayLike[Any],
    k: int,
    /,
    *,
    axis: int = -1,
    mode: str = "largest",
    sorted: bool = True,
) -> tuple[_Array[IntTuple], _Array[IntTuple]]: ...

# Searching
def searchsorted[N: IntVar, Shape: _Shape = []](
    a: _ShapedArrayLike[[N]],
    v: _ArrayLike[Shape],
    side: str = "left",
    sorter: _ArrayLike[Any] | None = None,
    *,
    method: str = "scan",
) -> _Array[Shape]: ...
@overload
def nonzero[Size: IntVar](
    a: _ArrayLike[Any],
    *,
    size: Int[Size],
    fill_value: Any = None,
) -> tuple[_Array[[Size]], ...]: ...
@overload
def nonzero(
    a: _ArrayLike[Any],
    *,
    size: int | None = None,
    fill_value: Any = None,
) -> tuple[_Array[IntTuple], ...]: ...
@overload
def flatnonzero[Size: IntVar](
    a: _ArrayLike[Any],
    *,
    size: Int[Size],
    fill_value: Any = None,
) -> _Array[[Size]]: ...
@overload
def flatnonzero(
    a: _ArrayLike[Any],
    *,
    size: int | None = None,
    fill_value: Any = None,
) -> _Array[IntTuple]: ...
@overload
def argwhere[N: IntVar, Size: IntVar](
    a: _ShapedArrayLike[[N]],
    *,
    size: Int[Size],
    fill_value: Any = None,
) -> _Array[[Size, 1]]: ...
@overload
def argwhere[M: IntVar, N: IntVar, Size: IntVar](
    a: _ShapedArrayLike[[M, N]],
    *,
    size: Int[Size],
    fill_value: Any = None,
) -> _Array[[Size, 2]]: ...
@overload
def argwhere[L: IntVar, M: IntVar, N: IntVar, Size: IntVar](
    a: _ShapedArrayLike[[L, M, N]],
    *,
    size: Int[Size],
    fill_value: Any = None,
) -> _Array[[Size, 3]]: ...
@overload
def argwhere[K: IntVar, L: IntVar, M: IntVar, N: IntVar, Size: IntVar](
    a: _ShapedArrayLike[[K, L, M, N]],
    *,
    size: Int[Size],
    fill_value: Any = None,
) -> _Array[[Size, 4]]: ...
@overload
def argwhere[Size: IntVar](
    a: _ArrayLike[Any],
    *,
    size: Int[Size],
    fill_value: Any = None,
) -> _Array[[Size, int]]: ...
@overload
def argwhere(
    a: _ArrayLike[Any],
    *,
    size: int | None = None,
    fill_value: Any = None,
) -> _Array[IntTuple]: ...
def nan_to_num[Shape: _Shape = []](
    x: _ArrayLike[Shape],
    copy: bool = True,
    nan: _Scalar = 0.0,
    posinf: _Scalar | None = None,
    neginf: _Scalar | None = None,
) -> _Array[Shape]: ...
def digitize[Shape: _Shape = []](
    x: _ArrayLike[Shape],
    bins: _ArrayLike[Any],
    right: bool = False,
    *,
    method: str | None = None,
) -> _Array[Shape]: ...
@overload
def where[Size: IntVar](
    condition: _ArrayLike[Any],
    *,
    size: Int[Size],
    fill_value: Any = None,
) -> tuple[_Array[[Size]], ...]: ...
@overload
def where(
    condition: _ArrayLike[Any],
    *,
    size: int | None = None,
    fill_value: Any = None,
) -> tuple[_Array[IntTuple], ...]: ...
@overload
def where[
    CondShape: _Shape = [],
    XShape: _Shape = [],
    YShape: _Shape = [],
](
    condition: _ArrayLike[CondShape],
    x: _ArrayLike[XShape],
    y: _ArrayLike[YShape],
    /,
) -> _Array[broadcast(broadcast(CondShape, XShape), YShape)]: ...

# Selection & Clipping
@overload
def bincount[Length: IntVar](
    x: _ArrayLike[Any],
    weights: _ArrayLike[Any] | None = None,
    minlength: int = 0,
    *,
    length: Int[Length],
    out_sharding: _NamedSharding | _PartitionSpec | None = None,
) -> _Array[[Length]]: ...
@overload
def bincount(
    x: _ArrayLike[Any],
    weights: _ArrayLike[Any] | None = None,
    minlength: int = 0,
    *,
    length: int | None = None,
    out_sharding: _NamedSharding | _PartitionSpec | None = None,
) -> _Array[IntTuple]: ...
@overload
def choose[Shape: _Shape = []](
    a: _ArrayLike[Shape],
    choices: Sequence[_ArrayLike[Shape]],
    out: Any = None,
    mode: str = "raise",
) -> _Array[Shape]: ...
@overload
def choose(
    a: _ArrayLike[Any],
    choices: Sequence[_ArrayLike[Any]] | _ArrayLike[Any],
    out: Any = None,
    mode: str = "raise",
) -> _Array[IntTuple]: ...
def clip[
    Shape: _Shape = [],
    MinShape: _Shape = [],
    MaxShape: _Shape = [],
](
    a: _ArrayLike[Shape],
    min: _ArrayLike[MinShape] | None = None,
    max: _ArrayLike[MaxShape] | None = None,
) -> _Array[broadcast(broadcast(Shape, MinShape), MaxShape)]: ...
def fmax[Shape1: _Shape = [], Shape2: _Shape = []](
    x1: _ArrayLike[Shape1], x2: _ArrayLike[Shape2], /
) -> _Array[broadcast(Shape1, Shape2)]: ...
def fmin[Shape1: _Shape = [], Shape2: _Shape = []](
    x1: _ArrayLike[Shape1], x2: _ArrayLike[Shape2], /
) -> _Array[broadcast(Shape1, Shape2)]: ...
def piecewise[Shape: _Shape = []](
    x: _ArrayLike[Shape],
    condlist: _ArrayLike[Any] | Sequence[_ArrayLike[Any]],
    funclist: Sequence[Any],
    *args: Any,
    **kw: Any,
) -> _Array[Shape]: ...
@overload
def select[Shape: _Shape = []](
    condlist: Sequence[_ArrayLike[Any]],
    choicelist: Sequence[_ArrayLike[Shape]],
    default: _ArrayLike[Shape] = 0,
) -> _Array[Shape]: ...
@overload
def select(
    condlist: Sequence[_ArrayLike[Any]],
    choicelist: Sequence[_ArrayLike[Any]],
    default: _ArrayLike[Any] = 0,
) -> _Array[IntTuple]: ...

# Set-like operations
@overload
def intersect1d[Size: IntVar](
    ar1: _ArrayLike[Any],
    ar2: _ArrayLike[Any],
    assume_unique: bool = False,
    return_indices: Literal[False] = False,
    *,
    size: Int[Size],
    fill_value: Any = None,
) -> _Array[[Size]]: ...
@overload
def intersect1d[Size: IntVar](
    ar1: _ArrayLike[Any],
    ar2: _ArrayLike[Any],
    assume_unique: bool = False,
    return_indices: Literal[True] = ...,
    *,
    size: Int[Size],
    fill_value: Any = None,
) -> tuple[_Array[[Size]], _Array[[Size]], _Array[[Size]]]: ...
@overload
def intersect1d(
    ar1: _ArrayLike[Any],
    ar2: _ArrayLike[Any],
    assume_unique: bool = False,
    return_indices: Literal[True] = ...,
    *,
    size: int | None = None,
    fill_value: Any = None,
) -> tuple[_Array[IntTuple], _Array[IntTuple], _Array[IntTuple]]: ...
@overload
def intersect1d(
    ar1: _ArrayLike[Any],
    ar2: _ArrayLike[Any],
    assume_unique: bool = False,
    return_indices: bool = False,
    *,
    size: int | None = None,
    fill_value: Any = None,
) -> _Array[IntTuple] | tuple[_Array[IntTuple], _Array[IntTuple], _Array[IntTuple]]: ...
def isin[Shape: _Shape = []](
    element: _ArrayLike[Shape],
    test_elements: _ArrayLike[Any],
    assume_unique: bool = False,
    invert: bool = False,
    *,
    method: str = "auto",
) -> _Array[Shape]: ...
@overload
def setdiff1d[Size: IntVar](
    ar1: _ArrayLike[Any],
    ar2: _ArrayLike[Any],
    assume_unique: bool = False,
    *,
    size: Int[Size],
    fill_value: Any = None,
) -> _Array[[Size]]: ...
@overload
def setdiff1d(
    ar1: _ArrayLike[Any],
    ar2: _ArrayLike[Any],
    assume_unique: bool = False,
    *,
    size: int | None = None,
    fill_value: Any = None,
) -> _Array[IntTuple]: ...
@overload
def setxor1d[Size: IntVar](
    ar1: _ArrayLike[Any],
    ar2: _ArrayLike[Any],
    assume_unique: bool = False,
    *,
    size: Int[Size],
    fill_value: Any = None,
) -> _Array[[Size]]: ...
@overload
def setxor1d(
    ar1: _ArrayLike[Any],
    ar2: _ArrayLike[Any],
    assume_unique: bool = False,
    *,
    size: int | None = None,
    fill_value: Any = None,
) -> _Array[IntTuple]: ...
@overload
def union1d[Size: IntVar](
    ar1: _ArrayLike[Any],
    ar2: _ArrayLike[Any],
    *,
    size: Int[Size],
    fill_value: Any = None,
) -> _Array[[Size]]: ...
@overload
def union1d(
    ar1: _ArrayLike[Any],
    ar2: _ArrayLike[Any],
    *,
    size: int | None = None,
    fill_value: Any = None,
) -> _Array[IntTuple]: ...
@overload
def unique[Size: IntVar](
    ar: _ArrayLike[Any],
    return_index: Literal[False] = False,
    return_inverse: Literal[False] = False,
    return_counts: Literal[False] = False,
    axis: int | None = None,
    *,
    equal_nan: bool = True,
    size: Int[Size],
    fill_value: Any = None,
    sorted: bool = True,
) -> _Array[[Size]]: ...
@overload
def unique(
    ar: _ArrayLike[Any],
    return_index: bool = False,
    return_inverse: bool = False,
    return_counts: bool = False,
    axis: int | None = None,
    *,
    equal_nan: bool = True,
    size: int | None = None,
    fill_value: Any = None,
    sorted: bool = True,
) -> Any: ...

class _UniqueAllResult(NamedTuple):
    values: _Array[IntTuple]
    indices: _Array[IntTuple]
    inverse_indices: _Array[IntTuple]
    counts: _Array[IntTuple]

class _UniqueCountsResult(NamedTuple):
    values: _Array[IntTuple]
    counts: _Array[IntTuple]

class _UniqueInverseResult(NamedTuple):
    values: _Array[IntTuple]
    inverse_indices: _Array[IntTuple]

def unique_all(
    x: _ArrayLike[Any],
    /,
    *,
    size: int | None = None,
    fill_value: Any = None,
) -> _UniqueAllResult: ...
def unique_counts(
    x: _ArrayLike[Any],
    /,
    *,
    size: int | None = None,
    fill_value: Any = None,
) -> _UniqueCountsResult: ...
def unique_inverse(
    x: _ArrayLike[Any],
    /,
    *,
    size: int | None = None,
    fill_value: Any = None,
) -> _UniqueInverseResult: ...
@overload
def unique_values[Size: IntVar](
    x: _ArrayLike[Any],
    /,
    *,
    size: Int[Size],
    fill_value: Any = None,
) -> _Array[[Size]]: ...
@overload
def unique_values(
    x: _ArrayLike[Any],
    /,
    *,
    size: int | None = None,
    fill_value: Any = None,
) -> _Array[IntTuple]: ...

# Indexing, Slicing & Masking
@overload
def compress[Size: Flag[int], Shape: _Shape = [], Axis: Flag[int | None] = None](
    condition: _ArrayLike[Any],
    a: _ArrayLike[Shape],
    axis: Axis = None,
    *,
    size: Size,
    fill_value: Any = 0,
    out: None = None,
) -> _Array[compress_shape(Shape, Size, Axis)]: ...
@overload
def compress[Shape: _Shape = [], Axis: Flag[int | None] = None](
    condition: _ArrayLike[Any],
    a: _ArrayLike[Shape],
    axis: Axis = None,
    *,
    size: int | None = None,
    fill_value: Any = 0,
    out: None = None,
) -> _Array[IntTuple]: ...
@overload
def compress(
    condition: _ArrayLike[Any],
    a: _ArrayLike[Any],
    axis: int | None = None,
    *,
    size: int | None = None,
    fill_value: Any = 0,
    out: None = None,
) -> _Array[IntTuple]: ...
def delete(
    arr: _ArrayLike[Any],
    obj: _ArrayLike[Any] | slice | Sequence[int],
    axis: int | None = None,
    *,
    assume_unique_indices: bool = False,
) -> _Array[IntTuple]: ...
@overload
def extract[Size: IntVar](
    condition: _ArrayLike[Any],
    arr: _ArrayLike[Any],
    *,
    size: Int[Size],
    fill_value: Any = 0,
) -> _Array[[Size]]: ...
@overload
def extract(
    condition: _ArrayLike[Any],
    arr: _ArrayLike[Any],
    *,
    size: int | None = None,
    fill_value: Any = 0,
) -> _Array[IntTuple]: ...
def fill_diagonal[Shape: _Shape = []](
    a: _ArrayLike[Shape],
    val: Any,
    wrap: bool = False,
    *,
    inplace: bool = True,
) -> _Array[fill_diagonal_shape(Shape)]: ...
def insert(
    arr: _ArrayLike[Any],
    obj: _ArrayLike[Any] | slice | Sequence[int],
    values: _ArrayLike[Any],
    axis: int | None = None,
) -> _Array[IntTuple]: ...
def place[Shape: _Shape = []](
    arr: _ArrayLike[Shape],
    mask: _ArrayLike[Any],
    vals: _ArrayLike[Any],
    *,
    inplace: bool = True,
) -> _Array[Shape]: ...
def put[Shape: _Shape = []](
    a: _ArrayLike[Shape],
    ind: _ArrayLike[Any],
    v: _ArrayLike[Any],
    mode: str | None = None,
    *,
    inplace: bool = True,
) -> _Array[Shape]: ...
def put_along_axis[Shape: _Shape = []](
    arr: _ArrayLike[Shape],
    indices: _ArrayLike[Any],
    values: _ArrayLike[Any],
    axis: int | None,
    inplace: bool = True,
    *,
    mode: str | None = None,
) -> _Array[Shape]: ...
def take[Shape: _Shape = [], IdxShape: _Shape = [], Axis: Flag[int | None] = None](
    a: _ArrayLike[Shape],
    indices: _ArrayLike[IdxShape],
    axis: Axis = None,
    out: None = None,
    mode: str | None = None,
    unique_indices: bool = False,
    indices_are_sorted: bool = False,
    fill_value: Any = None,
) -> _Array[take_shape(Shape, IdxShape, Axis)]: ...
def take_along_axis[
    ArrShape: _Shape = [],
    IdxShape: _Shape = [],
    Axis: Flag[int | None] = -1,
](
    arr: _ArrayLike[ArrShape],
    indices: _ArrayLike[IdxShape],
    axis: Axis = -1,
    mode: str | None = None,
    fill_value: Any = None,
    *,
    wrap_negative_indices: bool = True,
) -> _Array[take_along_axis_shape(ArrShape, IdxShape, Axis)]: ...
def trim_zeros(
    filt: _ArrayLike[Any] | Sequence[Any],
    trim: str = "fb",
    axis: int | Sequence[int] | None = None,
) -> _Array[IntTuple]: ...
@overload
def diag_indices[N: IntVar](
    n: Int[N],
    ndim: int = 2,
) -> tuple[_Array[[N]], ...]: ...
@overload
def diag_indices(
    n: int,
    ndim: int = 2,
) -> tuple[_Array[IntTuple], ...]: ...
def diag_indices_from[Shape: _Shape = []](
    arr: _ArrayLike[Shape],
) -> tuple[_Array[diag_indices_from_shape(Shape)], ...]: ...
@overload
def mask_indices[Size: IntVar](
    n: int,
    mask_func: Callable[..., Any],
    k: int = 0,
    *,
    size: Int[Size],
) -> tuple[_Array[[Size]], _Array[[Size]]]: ...
@overload
def mask_indices(
    n: int,
    mask_func: Callable[..., Any],
    k: int = 0,
    *,
    size: int | None = None,
) -> tuple[_Array[IntTuple], _Array[IntTuple]]: ...
@overload
def ravel_multi_index[Shape: _Shape](
    multi_index: Sequence[_ShapedArrayLike[Shape]],
    dims: Sequence[int],
    mode: str = "raise",
    order: str = "C",
    *,
    dtype: Any = None,
) -> _Array[Shape]: ...
@overload
def ravel_multi_index(
    multi_index: Sequence[Any],
    dims: Sequence[int],
    mode: str = "raise",
    order: str = "C",
    *,
    dtype: Any = None,
) -> _Array[IntTuple]: ...
def tril_indices(
    n: int,
    k: int = 0,
    m: int | None = None,
) -> tuple[_Array[IntTuple], _Array[IntTuple]]: ...
def tril_indices_from(
    arr: _ArrayLike[Any],
    k: int = 0,
) -> tuple[_Array[IntTuple], _Array[IntTuple]]: ...
def triu_indices(
    n: int,
    k: int = 0,
    m: int | None = None,
) -> tuple[_Array[IntTuple], _Array[IntTuple]]: ...
def triu_indices_from(
    arr: _ArrayLike[Any],
    k: int = 0,
) -> tuple[_Array[IntTuple], _Array[IntTuple]]: ...
def unravel_index[Shape: _Shape = []](
    indices: _ArrayLike[Shape],
    shape: Sequence[int] | int,
) -> tuple[_Array[Shape], ...]: ...
@overload
def ix_[Shapes: IntTuples](
    *args: Unpack[MapIntTuples[lambda S: _Array[S], Shapes]],
) -> MapIntTuples[lambda S: _Array[S], ix_shapes(Shapes)]: ...
@overload
def ix_(*args: _ArrayLike[Any]) -> tuple[_Array[IntTuple], ...]: ...

class finfo:
    bits: int
    dtype: Any
    eps: float
    epsneg: float
    iexp: int
    machep: int
    max: float
    maxexp: int
    min: float
    minexp: int
    negep: int
    nexp: int
    nmant: int
    precision: int
    resolution: float
    smallest_normal: float
    smallest_subnormal: float
    tiny: float
    def __init__(self, dtype: DTypeLike) -> None: ...

class iinfo:
    bits: int
    dtype: Any
    kind: str
    max: int
    min: int
    def __init__(self, dtype: DTypeLike) -> None: ...

class ufunc:
    @property
    def nin(self) -> int: ...
    @property
    def nout(self) -> int: ...
    @property
    def nargs(self) -> int: ...
    @property
    def ntypes(self) -> int: ...
    @property
    def types(self) -> list[str]: ...
    @property
    def identity(self) -> Any: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...
    def reduce(
        self,
        a: Any,
        axis: int = 0,
        dtype: Any = None,
        out: Any = None,
        keepdims: bool = False,
    ) -> Any: ...
    def accumulate(
        self,
        a: Any,
        axis: int = 0,
        dtype: Any = None,
        out: Any = None,
    ) -> Any: ...
    def reduceat(
        self,
        a: Any,
        indices: Sequence[int],
        axis: int = 0,
        dtype: Any = None,
        out: Any = None,
    ) -> Any: ...
    def outer(self, A: Any, B: Any, /, **kwargs: Any) -> Any: ...

class ComplexWarning(UserWarning): ...

def astype[Shape: _Shape = []](
    x: _ArrayLike[Shape],
    dtype: DTypeLike | None,
    /,
    *,
    copy: bool = False,
    device: _Device | _Sharding | None = None,
) -> _Array[Shape]: ...
def can_cast(from_: Any, to: DTypeLike, casting: str = "safe") -> bool: ...
def isdtype(
    dtype: DTypeLike, kind: str | DTypeLike | tuple[str | DTypeLike, ...]
) -> bool: ...
def issubdtype(arg1: DTypeLike, arg2: DTypeLike) -> bool: ...
def promote_types(a: DTypeLike, b: DTypeLike) -> Any: ...
def result_type(*args: Any) -> Any: ...

class _Mgrid:
    @overload
    def __getitem__(self, key: slice) -> _Array[[Any]]: ...
    @overload
    def __getitem__(self, key: tuple[slice, slice]) -> _Array[[2, Any, Any]]: ...
    @overload
    def __getitem__(
        self, key: tuple[slice, slice, slice]
    ) -> _Array[[3, Any, Any, Any]]: ...
    @overload
    def __getitem__(self, key: tuple[slice, ...]) -> _Array[IntTuple]: ...
    @overload
    def __getitem__(self, key: Any) -> _Array[IntTuple]: ...

class _Ogrid:
    @overload
    def __getitem__(self, key: slice) -> _Array[[Any]]: ...
    @overload
    def __getitem__(self, key: tuple[slice, slice]) -> list[_Array[IntTuple]]: ...
    @overload
    def __getitem__(
        self, key: tuple[slice, slice, slice]
    ) -> list[_Array[IntTuple]]: ...
    @overload
    def __getitem__(self, key: tuple[slice, ...]) -> list[_Array[IntTuple]]: ...
    @overload
    def __getitem__(self, key: Any) -> _Array[IntTuple] | list[_Array[IntTuple]]: ...

mgrid: _Mgrid
ogrid: _Ogrid

class _CClass:
    def __getitem__(self, key: Any) -> _Array[IntTuple]: ...

class _RClass:
    def __getitem__(self, key: Any) -> _Array[IntTuple]: ...

c_: _CClass
r_: _RClass

class _IndexExpression:
    @overload
    def __getitem__[TupleT: tuple[Any, ...]](self, item: TupleT) -> TupleT: ...
    @overload
    def __getitem__[T](self, item: T) -> tuple[T]: ...

class _SClass:
    def __getitem__[T](self, item: T) -> T: ...

index_exp: _IndexExpression
s_: _SClass

def ndim(a: _ArrayLike[Any] | Sequence[Any]) -> int: ...
@overload
def shape[Shape: _Shape = []](
    a: _ArrayLike[Shape] | RegularNestedList[Shape, _Scalar],
) -> Shape: ...
@overload
def shape(a: Sequence[Any]) -> tuple[int, ...]: ...
def size(
    a: _ArrayLike[Any] | Sequence[Any], axis: int | Sequence[int] | None = None
) -> int: ...
def get_printoptions() -> dict[str, Any]: ...
def set_printoptions(
    precision: int | None = None,
    threshold: int | None = None,
    edgeitems: int | None = None,
    linewidth: int | None = None,
    suppress: bool | None = None,
    nanstr: str | None = None,
    infstr: str | None = None,
    formatter: dict[str, Callable[..., str]] | None = None,
    sign: str | None = None,
    floatmode: str | None = None,
    **kwarg: Any,
) -> None: ...
def printoptions(*args: Any, **kwargs: Any) -> ContextManager[dict[str, Any]]: ...
def apply_along_axis(
    func1d: Callable[..., Any],
    axis: int,
    arr: _ArrayLike[Any],
    *args: Any,
    **kwargs: Any,
) -> _Array[IntTuple]: ...
def apply_over_axes(
    func: Callable[[Any, int], Any], a: _ArrayLike[Any], axes: Sequence[int]
) -> _Array[IntTuple]: ...
def frompyfunc(
    func: Callable[..., Any], /, nin: int, nout: int, *, identity: Any = None
) -> ufunc: ...
def vectorize(pyfunc: Any, *, excluded: Any = ..., signature: Any = None) -> Any: ...
def load(file: Any, *args: Any, **kwargs: Any) -> Any: ...

# Bit packing

@overload
def packbits[Shape: _Shape = [], Axis: Flag[int | None] = None](
    a: _ArrayLike[Shape],
    axis: Axis = None,
    bitorder: str = "big",
) -> _Array[packbits_shape(Shape, Axis)]: ...
@overload
def packbits(
    a: _ArrayLike[Any],
    axis: int | None = None,
    bitorder: str = "big",
) -> _Array[IntTuple]: ...
@overload
def unpackbits[
    Shape: _Shape = [],
    Axis: Flag[int | None] = None,
    Count: Flag[int | None] = None,
](
    a: _ArrayLike[Shape],
    axis: Axis = None,
    count: Count = None,
    bitorder: str = "big",
) -> _Array[unpackbits_shape(Shape, Axis, Count)]: ...
@overload
def unpackbits(
    a: _ArrayLike[Any],
    axis: int | None = None,
    count: int | None = None,
    bitorder: str = "big",
) -> _Array[IntTuple]: ...

# Interpolation

def interp[N: IntVar, Shape: _Shape = []](
    x: _ArrayLike[Shape],
    xp: _ShapedArrayLike[[N]],
    fp: _ShapedArrayLike[[N]],
    left: Any = None,
    right: Any = None,
    period: Any = None,
) -> _Array[Shape]: ...

# Convolutions & Signal Processing

@overload
def convolve[ShapeA: _Shape = [], ShapeV: _Shape = [], Mode: Flag[str] = "full"](
    a: _ArrayLike[ShapeA],
    v: _ArrayLike[ShapeV],
    mode: Mode = "full",
    *,
    precision: Any = None,
    preferred_element_type: DTypeLike | None = None,
) -> _Array[convolve_shape(ShapeA, ShapeV, Mode)]: ...
@overload
def convolve(
    a: _ArrayLike[Any],
    v: _ArrayLike[Any],
    mode: str = "full",
    *,
    precision: Any = None,
    preferred_element_type: DTypeLike | None = None,
) -> _Array[IntTuple]: ...
@overload
def correlate[ShapeA: _Shape = [], ShapeV: _Shape = [], Mode: Flag[str] = "valid"](
    a: _ArrayLike[ShapeA],
    v: _ArrayLike[ShapeV],
    mode: Mode = "valid",
    *,
    precision: Any = None,
    preferred_element_type: DTypeLike | None = None,
) -> _Array[convolve_shape(ShapeA, ShapeV, Mode)]: ...
@overload
def correlate(
    a: _ArrayLike[Any],
    v: _ArrayLike[Any],
    mode: str = "valid",
    *,
    precision: Any = None,
    preferred_element_type: DTypeLike | None = None,
) -> _Array[IntTuple]: ...

# Histograms

@overload
def histogram[Bins: Flag[int] = 10](
    a: _ArrayLike[Any],
    bins: Bins = 10,
    range: Sequence[Any] | None = None,
    weights: _ArrayLike[Any] | None = None,
    density: bool | None = None,
) -> tuple[
    _Array[histogram_counts_shape(Bins)], _Array[histogram_edges_shape(Bins)]
]: ...
@overload
def histogram(
    a: _ArrayLike[Any],
    bins: Any = 10,
    range: Sequence[Any] | None = None,
    weights: _ArrayLike[Any] | None = None,
    density: bool | None = None,
) -> tuple[_Array[IntTuple], _Array[IntTuple]]: ...
@overload
def histogram2d[Bins: Flag[int] = 10](
    x: _ArrayLike[Any],
    y: _ArrayLike[Any],
    bins: Bins = 10,
    range: Sequence[Any] | None = None,
    weights: _ArrayLike[Any] | None = None,
    density: bool | None = None,
) -> tuple[
    _Array[histogram2d_counts_shape(Bins)],
    _Array[histogram_edges_shape(Bins)],
    _Array[histogram_edges_shape(Bins)],
]: ...
@overload
def histogram2d(
    x: _ArrayLike[Any],
    y: _ArrayLike[Any],
    bins: Any = 10,
    range: Sequence[Any] | None = None,
    weights: _ArrayLike[Any] | None = None,
    density: bool | None = None,
) -> tuple[_Array[IntTuple], _Array[IntTuple], _Array[IntTuple]]: ...
@overload
def histogram_bin_edges[Bins: Flag[int] = 10](
    a: _ArrayLike[Any],
    bins: Bins = 10,
    range: Any = None,
    weights: _ArrayLike[Any] | None = None,
) -> _Array[histogram_edges_shape(Bins)]: ...
@overload
def histogram_bin_edges(
    a: _ArrayLike[Any],
    bins: Any = 10,
    range: Any = None,
    weights: _ArrayLike[Any] | None = None,
) -> _Array[IntTuple]: ...
@overload
def histogramdd[Bins: IntTuple](
    sample: _ArrayLike[Any],
    bins: Bins,
    range: Sequence[Any] | None = None,
    weights: _ArrayLike[Any] | None = None,
    density: bool | None = None,
) -> tuple[_Array[Bins], list[_Array[IntTuple]]]: ...
@overload
def histogramdd(
    sample: _ArrayLike[Any],
    bins: Any = 10,
    range: Sequence[Any] | None = None,
    weights: _ArrayLike[Any] | None = None,
    density: bool | None = None,
) -> tuple[_Array[IntTuple], list[_Array[IntTuple]]]: ...

# Polynomials

def poly[Shape: _Shape](
    seq_of_zeros: _ShapedArrayLike[Shape],
) -> _Array[poly_shape(Shape)]: ...
def polyadd[Shape1: _Shape = [], Shape2: _Shape = []](
    a1: _ArrayLike[Shape1],
    a2: _ArrayLike[Shape2],
) -> _Array[polyadd_shape(Shape1, Shape2)]: ...
@overload
def polyder[Shape: _Shape = [], M: Flag[int] = 1](
    p: _ArrayLike[Shape],
    m: M = 1,
) -> _Array[polyder_shape(Shape, M)]: ...
@overload
def polyder(p: _ArrayLike[Any], m: int = 1) -> _Array[IntTuple]: ...
@overload
def polydiv[Shape1: _Shape = [], Shape2: _Shape = []](
    u: _ArrayLike[Shape1],
    v: _ArrayLike[Shape2],
    *,
    trim_leading_zeros: Literal[False] = False,
) -> tuple[_Array[polydiv_quotient_shape(Shape1, Shape2)], _Array[Shape1]]: ...
@overload
def polydiv(
    u: _ArrayLike[Any],
    v: _ArrayLike[Any],
    *,
    trim_leading_zeros: bool = False,
) -> tuple[_Array[IntTuple], _Array[IntTuple]]: ...
@overload
def polyfit[Deg: Flag[int]](
    x: _ArrayLike[Any],
    y: _ArrayLike[Any],
    deg: Deg,
    rcond: float | None = None,
    full: Literal[False] = False,
    w: _ArrayLike[Any] | None = None,
    cov: Literal[False] = False,
) -> _Array[polyfit_shape(Deg)]: ...
@overload
def polyfit[Deg: Flag[int]](
    x: _ArrayLike[Any],
    y: _ArrayLike[Any],
    deg: Deg,
    rcond: float | None = None,
    full: Literal[False] = False,
    w: _ArrayLike[Any] | None = None,
    cov: Literal[True, "unscaled"] = ...,
) -> tuple[_Array[polyfit_shape(Deg)], _Array[polyfit_cov_shape(Deg)]]: ...
@overload
def polyfit(
    x: _ArrayLike[Any],
    y: _ArrayLike[Any],
    deg: int,
    rcond: float | None = None,
    full: Literal[True] = ...,
    w: _ArrayLike[Any] | None = None,
    cov: bool = False,
) -> tuple[_Array[IntTuple], ...]: ...
@overload
def polyfit(
    x: _ArrayLike[Any],
    y: _ArrayLike[Any],
    deg: int,
    rcond: float | None = None,
    full: bool = False,
    w: _ArrayLike[Any] | None = None,
    cov: bool | str = False,
) -> Any: ...
@overload
def polyint[Shape: _Shape = [], M: Flag[int] = 1](
    p: _ArrayLike[Shape],
    m: M = 1,
    k: _ArrayLike[Any] | None = None,
) -> _Array[polyint_shape(Shape, M)]: ...
@overload
def polyint(
    p: _ArrayLike[Any],
    m: int = 1,
    k: _ArrayLike[Any] | None = None,
) -> _Array[IntTuple]: ...
@overload
def polymul[Shape1: _Shape = [], Shape2: _Shape = []](
    a1: _ArrayLike[Shape1],
    a2: _ArrayLike[Shape2],
    *,
    trim_leading_zeros: Literal[False] = False,
) -> _Array[convolve_shape(Shape1, Shape2, "full")]: ...
@overload
def polymul(
    a1: _ArrayLike[Any],
    a2: _ArrayLike[Any],
    *,
    trim_leading_zeros: bool = False,
) -> _Array[IntTuple]: ...
def polysub[Shape1: _Shape = [], Shape2: _Shape = []](
    a1: _ArrayLike[Shape1],
    a2: _ArrayLike[Shape2],
) -> _Array[polyadd_shape(Shape1, Shape2)]: ...
def polyval[Shape: _Shape = []](
    p: _ArrayLike[Any],
    x: _ArrayLike[Shape],
    *,
    unroll: int = 16,
) -> _Array[Shape]: ...
def roots(p: _ArrayLike[Any], *, strip_zeros: bool = True) -> _Array[IntTuple]: ...

# Scalar constructors

bool: Any
bool_: Any
int_: Any
int8: Any
int16: Any
int32: Any
int64: Any
uint: Any
uint8: Any
uint16: Any
uint32: Any
uint64: Any
int1: Any
int2: Any
int4: Any
uint1: Any
uint2: Any
uint4: Any
float_: Any
float16: Any
float32: Any
float64: Any
bfloat16: Any
single: Any
double: Any
csingle: Any
cdouble: Any
complex_: Any
complex64: Any
complex128: Any
float4_e2m1fn: Any
float6_e2m3fn: Any
float6_e3m2fn: Any
float8_e3m4: Any
float8_e4m3: Any
float8_e4m3b11fnuz: Any
float8_e4m3fn: Any
float8_e4m3fnuz: Any
float8_e5m2: Any
float8_e5m2fnuz: Any
float8_e8m0fnu: Any
