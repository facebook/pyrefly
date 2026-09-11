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

from jax._array import Array as Array, Array as ndarray
from jax._shapes import (
    append_shape,
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
    take_scalar_idx_shape,
    take_shape,
    tensordot_shape,
    top_k_shape,
    trace_shape,
    unpackbits_shape,
    vecmat_shape,
    vstack_shape,
)
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
)

from . import fft as fft, linalg as linalg

type _Shape = IntTuple
type _Axis = int | tuple[int, ...] | None
# The trailing `None` is not a legal argument to `reshape`. It is present because
# an `int | tuple[int, ...]` parameter cannot be iterated inside a DSL function
# after narrowing with `is_int_value` alone. See `reshape_shape`, which rejects it.
type _NewShape = int | tuple[int, ...] | None
type _Scalar = bool | int | float | complex

@overload
def array(
    object: _Scalar,
    dtype: DTypeLike | None = ...,
    copy: bool | None = ...,
    order: str | None = ...,
    ndmin: Literal[0] = 0,
    *,
    device: Any = ...,
    out_sharding: Any = ...,
) -> Array[[]]: ...
@overload
def array[Shape: _Shape](
    object: Array[Shape],
    dtype: DTypeLike | None = ...,
    copy: bool | None = ...,
    order: str | None = ...,
    ndmin: Literal[0] = 0,
    *,
    device: Any = ...,
    out_sharding: Any = ...,
) -> Array[Shape]: ...
@overload
def array(
    object: Any,
    dtype: DTypeLike | None = ...,
    copy: bool | None = ...,
    order: str | None = ...,
    ndmin: int = 0,
    *,
    device: Any = ...,
    out_sharding: Any = ...,
) -> Array[IntTuple]: ...
@overload
def asarray(
    a: _Scalar,
    dtype: DTypeLike | None = ...,
    order: str | None = ...,
    *,
    copy: bool | None = ...,
    device: Any = ...,
    out_sharding: Any = ...,
) -> Array[[]]: ...
@overload
def asarray[Shape: _Shape](
    a: Array[Shape],
    dtype: DTypeLike | None = ...,
    order: str | None = ...,
    *,
    copy: bool | None = ...,
    device: Any = ...,
    out_sharding: Any = ...,
) -> Array[Shape]: ...
@overload
def asarray(
    a: Any,
    dtype: DTypeLike | None = ...,
    order: str | None = ...,
    *,
    copy: bool | None = ...,
    device: Any = ...,
    out_sharding: Any = ...,
) -> Array[IntTuple]: ...
@overload
def copy[Shape: _Shape](a: Array[Shape], order: str | None = None) -> Array[Shape]: ...
@overload
def copy(a: Any, order: str | None = None) -> Array[IntTuple]: ...

# Literal tuples and values typed as `IntTuple` retain their shape. Other integer
# sequences fall through to a gradual overload rather than being rejected.
# TODO(stroxler): Replace these finite tuple-shape constructor overloads with a
# single `Shape: tuple[int, ...]` overload once whole-shape parameters flow
# through downstream array operations without degrading to unknown. The NumPy
# stubs carry the same limitation.
@overload
def zeros(
    shape: tuple[()], dtype: DTypeLike | None = ..., *, device: Any = ...
) -> Array[[]]: ...
@overload
def zeros[N: IntVar](
    shape: Int[N], dtype: DTypeLike | None = ..., *, device: Any = ...
) -> Array[[N]]: ...
@overload
def zeros[Shape: _Shape](
    shape: Shape, dtype: DTypeLike | None = ..., *, device: Any = ...
) -> Array[Shape]: ...
@overload
def zeros(
    shape: Sequence[int] | int, dtype: DTypeLike | None = ..., *, device: Any = ...
) -> Array[IntTuple]: ...
@overload
def ones(
    shape: tuple[()], dtype: DTypeLike | None = ..., *, device: Any = ...
) -> Array[[]]: ...
@overload
def ones[N: IntVar](
    shape: Int[N], dtype: DTypeLike | None = ..., *, device: Any = ...
) -> Array[[N]]: ...
@overload
def ones[Shape: _Shape](
    shape: Shape, dtype: DTypeLike | None = ..., *, device: Any = ...
) -> Array[Shape]: ...
@overload
def ones(
    shape: Sequence[int] | int, dtype: DTypeLike | None = ..., *, device: Any = ...
) -> Array[IntTuple]: ...
@overload
def empty[N: IntVar](
    shape: Int[N],
    dtype: DTypeLike | None = ...,
    *,
    device: Any = ...,
    out_sharding: Any = ...,
) -> Array[[N]]: ...
@overload
def empty[Shape: _Shape](
    shape: Shape,
    dtype: DTypeLike | None = ...,
    *,
    device: Any = ...,
    out_sharding: Any = ...,
) -> Array[Shape]: ...
@overload
def empty(
    shape: Sequence[int] | int,
    dtype: DTypeLike | None = ...,
    *,
    device: Any = ...,
    out_sharding: Any = ...,
) -> Array[IntTuple]: ...
@overload
def full(
    shape: tuple[()],
    fill_value: Any,
    dtype: DTypeLike | None = ...,
    *,
    device: Any = ...,
) -> Array[[]]: ...
@overload
def full[N: IntVar](
    shape: Int[N], fill_value: Any, dtype: DTypeLike | None = ..., *, device: Any = ...
) -> Array[[N]]: ...
@overload
def full[Shape: _Shape](
    shape: Shape, fill_value: Any, dtype: DTypeLike | None = ..., *, device: Any = ...
) -> Array[Shape]: ...
@overload
def full(
    shape: Sequence[int] | int,
    fill_value: Any,
    dtype: DTypeLike | None = ...,
    *,
    device: Any = ...,
) -> Array[IntTuple]: ...

# `_like` constructors
@overload
def empty_like[Shape: _Shape](
    prototype: Array[Shape],
    dtype: DTypeLike | None = ...,
    shape: None = None,
    *,
    device: Any = ...,
) -> Array[Shape]: ...
@overload
def empty_like[N: IntVar](
    prototype: Any,
    dtype: DTypeLike | None = ...,
    shape: Int[N] = ...,
    *,
    device: Any = ...,
) -> Array[[N]]: ...
@overload
def empty_like[Shape: _Shape](
    prototype: Any,
    dtype: DTypeLike | None = ...,
    shape: Shape = ...,
    *,
    device: Any = ...,
) -> Array[Shape]: ...
@overload
def empty_like(
    prototype: Any,
    dtype: DTypeLike | None = ...,
    shape: Sequence[int] | int | None = None,
    *,
    device: Any = ...,
) -> Array[IntTuple]: ...
@overload
def zeros_like[Shape: _Shape](
    a: Array[Shape],
    dtype: DTypeLike | None = ...,
    shape: None = None,
    *,
    device: Any = ...,
    out_sharding: Any = ...,
) -> Array[Shape]: ...
@overload
def zeros_like[N: IntVar](
    a: Any,
    dtype: DTypeLike | None = ...,
    shape: Int[N] = ...,
    *,
    device: Any = ...,
    out_sharding: Any = ...,
) -> Array[[N]]: ...
@overload
def zeros_like[Shape: _Shape](
    a: Any,
    dtype: DTypeLike | None = ...,
    shape: Shape = ...,
    *,
    device: Any = ...,
    out_sharding: Any = ...,
) -> Array[Shape]: ...
@overload
def zeros_like(
    a: Any,
    dtype: DTypeLike | None = ...,
    shape: Sequence[int] | int | None = None,
    *,
    device: Any = ...,
    out_sharding: Any = ...,
) -> Array[IntTuple]: ...
@overload
def ones_like[Shape: _Shape](
    a: Array[Shape],
    dtype: DTypeLike | None = ...,
    shape: None = None,
    *,
    device: Any = ...,
    out_sharding: Any = ...,
) -> Array[Shape]: ...
@overload
def ones_like[N: IntVar](
    a: Any,
    dtype: DTypeLike | None = ...,
    shape: Int[N] = ...,
    *,
    device: Any = ...,
    out_sharding: Any = ...,
) -> Array[[N]]: ...
@overload
def ones_like[Shape: _Shape](
    a: Any,
    dtype: DTypeLike | None = ...,
    shape: Shape = ...,
    *,
    device: Any = ...,
    out_sharding: Any = ...,
) -> Array[Shape]: ...
@overload
def ones_like(
    a: Any,
    dtype: DTypeLike | None = ...,
    shape: Sequence[int] | int | None = None,
    *,
    device: Any = ...,
    out_sharding: Any = ...,
) -> Array[IntTuple]: ...
@overload
def full_like[Shape: _Shape](
    a: Array[Shape],
    fill_value: Any,
    dtype: DTypeLike | None = ...,
    shape: None = None,
    *,
    device: Any = ...,
    out_sharding: Any = ...,
) -> Array[Shape]: ...
@overload
def full_like[N: IntVar](
    a: Any,
    fill_value: Any,
    dtype: DTypeLike | None = ...,
    shape: Int[N] = ...,
    *,
    device: Any = ...,
    out_sharding: Any = ...,
) -> Array[[N]]: ...
@overload
def full_like[Shape: _Shape](
    a: Any,
    fill_value: Any,
    dtype: DTypeLike | None = ...,
    shape: Shape = ...,
    *,
    device: Any = ...,
    out_sharding: Any = ...,
) -> Array[Shape]: ...
@overload
def full_like(
    a: Any,
    fill_value: Any,
    dtype: DTypeLike | None = ...,
    shape: Sequence[int] | int | None = None,
    *,
    device: Any = ...,
    out_sharding: Any = ...,
) -> Array[IntTuple]: ...

# `arange`, `linspace`, `logspace`, `geomspace`
@overload
def arange[N: IntVar](
    start: Int[N], *, dtype: DTypeLike | None = ..., device: Any = ...
) -> Array[[N]]: ...
@overload
def arange(
    start: float, *, dtype: DTypeLike | None = ..., device: Any = ...
) -> Array[[int]]: ...
@overload
def arange(
    start: int | float,
    stop: int | float,
    step: int | float = ...,
    dtype: DTypeLike | None = ...,
) -> Array[[int]]: ...
@overload
def linspace[N: IntVar](
    start: Any,
    stop: Any,
    num: Int[N],
    endpoint: bool = True,
    retstep: Literal[False] = False,
    dtype: DTypeLike | None = None,
    axis: int = 0,
    *,
    device: Any = None,
) -> Array[[N]]: ...
@overload
def linspace[N: IntVar](
    start: Any,
    stop: Any,
    num: Int[N],
    endpoint: bool,
    retstep: Literal[True],
    dtype: DTypeLike | None = None,
    axis: int = 0,
    *,
    device: Any = None,
) -> tuple[Array[[N]], Array[[]]]: ...
@overload
def linspace(
    start: Any,
    stop: Any,
    num: int = 50,
    endpoint: bool = True,
    retstep: Literal[False] = False,
    dtype: DTypeLike | None = None,
    axis: int = 0,
    *,
    device: Any = None,
) -> Array[[int]]: ...
@overload
def linspace(
    start: Any,
    stop: Any,
    num: int,
    endpoint: bool,
    retstep: Literal[True],
    dtype: DTypeLike | None = None,
    axis: int = 0,
    *,
    device: Any = None,
) -> tuple[Array[[int]], Array[[]]]: ...
@overload
def linspace(
    start: Any,
    stop: Any,
    num: int = 50,
    endpoint: bool = True,
    retstep: bool = False,
    dtype: DTypeLike | None = None,
    axis: int = 0,
    *,
    device: Any = None,
) -> Array[IntTuple] | tuple[Array[IntTuple], Array[[]]]: ...
@overload
def logspace[N: IntVar](
    start: Any,
    stop: Any,
    num: Int[N],
    endpoint: bool = True,
    base: Any = 10.0,
    dtype: DTypeLike | None = None,
    axis: int = 0,
) -> Array[[N]]: ...
@overload
def logspace(
    start: Any,
    stop: Any,
    num: int = 50,
    endpoint: bool = True,
    base: Any = 10.0,
    dtype: DTypeLike | None = None,
    axis: int = 0,
) -> Array[[int]]: ...
@overload
def geomspace[N: IntVar](
    start: Any,
    stop: Any,
    num: Int[N],
    endpoint: bool = True,
    dtype: DTypeLike | None = None,
    axis: int = 0,
) -> Array[[N]]: ...
@overload
def geomspace(
    start: Any,
    stop: Any,
    num: int = 50,
    endpoint: bool = True,
    dtype: DTypeLike | None = None,
    axis: int = 0,
) -> Array[[int]]: ...

# `eye`, `identity`, `diag`, `diagflat`, `tri`, `tril`, `triu`, `vander`
@overload
def eye[N: IntVar](
    N: Int[N],
    M: None = ...,
    k: int = ...,
    dtype: DTypeLike | None = ...,
    *,
    device: Any = ...,
) -> Array[[N, N]]: ...
@overload
def eye[N: IntVar, M: IntVar](
    N: Int[N],
    M: Int[M],
    k: int = ...,
    dtype: DTypeLike | None = ...,
    *,
    device: Any = ...,
) -> Array[[N, M]]: ...
def identity[N: IntVar](
    n: Int[N], dtype: DTypeLike | None = ..., *, device: Any = ...
) -> Array[[N, N]]: ...
@overload
def diag[N: IntVar](v: Array[[N]], k: int = 0) -> Array[[N, N]]: ...
@overload
def diag[N: IntVar, M: IntVar](
    v: Array[[N, M]], k: int = 0
) -> Array[[int_min(Int[N], Int[M])]]: ...
@overload
def diag(v: Any, k: int = 0) -> Array[IntTuple]: ...
@overload
def diagflat[N: IntVar](v: Array[[N]], k: int = 0) -> Array[[N, N]]: ...
@overload
def diagflat(v: Any, k: int = 0) -> Array[IntTuple]: ...
@overload
def tri[N: IntVar](
    N: Int[N], M: None = None, k: int = 0, dtype: DTypeLike | None = None
) -> Array[[N, N]]: ...
@overload
def tri[N: IntVar, M: IntVar](
    N: Int[N], M: Int[M], k: int = 0, dtype: DTypeLike | None = None
) -> Array[[N, M]]: ...
@overload
def tri(
    N: int, M: int | None = None, k: int = 0, dtype: DTypeLike | None = None
) -> Array[IntTuple]: ...
def tril[Shape: _Shape](m: Array[Shape], k: int = 0) -> Array[Shape]: ...
def triu[Shape: _Shape](m: Array[Shape], k: int = 0) -> Array[Shape]: ...
@overload
def vander[M: IntVar](
    x: Array[[M]], N: None = None, increasing: bool = False
) -> Array[[M, M]]: ...
@overload
def vander[M: IntVar, N: IntVar](
    x: Array[[M]], N: Int[N], increasing: bool = False
) -> Array[[M, N]]: ...
@overload
def vander(
    x: Any, N: int | None = None, increasing: bool = False
) -> Array[IntTuple]: ...

# `indices`, `meshgrid`
@overload
def indices[N: IntVar](
    dimensions: IntTuple[N],
    dtype: DTypeLike | None = None,
    sparse: Literal[False] = False,
) -> Array[[1, N]]: ...
@overload
def indices[N: IntVar, M: IntVar](
    dimensions: IntTuple[N, M],
    dtype: DTypeLike | None = None,
    sparse: Literal[False] = False,
) -> Array[[2, N, M]]: ...
@overload
def indices[N: IntVar, M: IntVar, K: IntVar](
    dimensions: IntTuple[N, M, K],
    dtype: DTypeLike | None = None,
    sparse: Literal[False] = False,
) -> Array[[3, N, M, K]]: ...
@overload
def indices(
    dimensions: Sequence[int], dtype: DTypeLike | None = None, sparse: bool = False
) -> Array[IntTuple] | tuple[Array[IntTuple], ...]: ...
@overload
def meshgrid[N: IntVar, M: IntVar](
    x1: Array[[N]],
    x2: Array[[M]],
    /,
    *,
    copy: bool = True,
    sparse: Literal[False] = False,
    indexing: Literal["xy"] = "xy",
) -> tuple[Array[[M, N]], Array[[M, N]]]: ...
@overload
def meshgrid[N: IntVar, M: IntVar](
    x1: Array[[N]],
    x2: Array[[M]],
    /,
    *,
    copy: bool = True,
    sparse: Literal[False] = False,
    indexing: Literal["ij"],
) -> tuple[Array[[N, M]], Array[[N, M]]]: ...
@overload
def meshgrid[N: IntVar, M: IntVar, K: IntVar](
    x1: Array[[N]],
    x2: Array[[M]],
    x3: Array[[K]],
    /,
    *,
    copy: bool = True,
    sparse: Literal[False] = False,
    indexing: Literal["xy"] = "xy",
) -> tuple[Array[[M, N, K]], Array[[M, N, K]], Array[[M, N, K]]]: ...
@overload
def meshgrid[N: IntVar, M: IntVar, K: IntVar](
    x1: Array[[N]],
    x2: Array[[M]],
    x3: Array[[K]],
    /,
    *,
    copy: bool = True,
    sparse: Literal[False] = False,
    indexing: Literal["ij"],
) -> tuple[Array[[N, M, K]], Array[[N, M, K]], Array[[N, M, K]]]: ...
@overload
def meshgrid(
    *xi: Any, copy: bool = True, sparse: bool = False, indexing: str = "xy"
) -> tuple[Array[IntTuple], ...]: ...

# `from_*` constructors
def from_dlpack(
    x: Any, /, *, device: Any = None, copy: bool | None = None
) -> Array[IntTuple]: ...
def frombuffer(
    buffer: Any, dtype: DTypeLike = float, count: int = -1, offset: int = 0
) -> Array[IntTuple]: ...
def fromfile(*args: Any, **kwargs: Any) -> Array[IntTuple]: ...
@overload
def fromfunction[N: IntVar](
    function: Callable[..., Any],
    shape: IntTuple[N],
    *,
    dtype: DTypeLike = float,
    **kwargs: Any,
) -> Array[[N]]: ...
@overload
def fromfunction[N: IntVar, M: IntVar](
    function: Callable[..., Any],
    shape: IntTuple[N, M],
    *,
    dtype: DTypeLike = float,
    **kwargs: Any,
) -> Array[[N, M]]: ...
@overload
def fromfunction[N: IntVar, M: IntVar, K: IntVar](
    function: Callable[..., Any],
    shape: IntTuple[N, M, K],
    *,
    dtype: DTypeLike = float,
    **kwargs: Any,
) -> Array[[N, M, K]]: ...
@overload
def fromfunction(
    function: Callable[..., Any],
    shape: Sequence[int],
    *,
    dtype: DTypeLike = float,
    **kwargs: Any,
) -> Array[IntTuple]: ...
def fromiter(*args: Any, **kwargs: Any) -> Array[IntTuple]: ...
def fromstring(
    string: str, dtype: DTypeLike = float, count: int = -1, *, sep: str
) -> Array[IntTuple]: ...

# Window functions
@overload
def bartlett[N: IntVar](M: Int[N]) -> Array[[N]]: ...
@overload
def bartlett(M: int) -> Array[IntTuple]: ...
@overload
def blackman[N: IntVar](M: Int[N]) -> Array[[N]]: ...
@overload
def blackman(M: int) -> Array[IntTuple]: ...
@overload
def hamming[N: IntVar](M: Int[N]) -> Array[[N]]: ...
@overload
def hamming(M: int) -> Array[IntTuple]: ...
@overload
def hanning[N: IntVar](M: Int[N]) -> Array[[N]]: ...
@overload
def hanning(M: int) -> Array[IntTuple]: ...
@overload
def kaiser[N: IntVar](M: Int[N], beta: Any) -> Array[[N]]: ...
@overload
def kaiser(M: int, beta: Any) -> Array[IntTuple]: ...

# Shape-preserving elementwise unary functions.
def abs[Shape: _Shape](x: Array[Shape], /) -> Array[Shape]: ...
def absolute[Shape: _Shape](x: Array[Shape], /) -> Array[Shape]: ...
def acos[Shape: _Shape](x: Array[Shape], /) -> Array[Shape]: ...
def acosh[Shape: _Shape](x: Array[Shape], /) -> Array[Shape]: ...
def angle[Shape: _Shape](z: Array[Shape], deg: bool = False) -> Array[Shape]: ...
def arccos[Shape: _Shape](x: Array[Shape], /) -> Array[Shape]: ...
def arccosh[Shape: _Shape](x: Array[Shape], /) -> Array[Shape]: ...
def arcsin[Shape: _Shape](x: Array[Shape], /) -> Array[Shape]: ...
def arcsinh[Shape: _Shape](x: Array[Shape], /) -> Array[Shape]: ...
def arctan[Shape: _Shape](x: Array[Shape], /) -> Array[Shape]: ...
def arctanh[Shape: _Shape](x: Array[Shape], /) -> Array[Shape]: ...
def around[Shape: _Shape](a: Array[Shape], decimals: int = 0) -> Array[Shape]: ...
def asin[Shape: _Shape](x: Array[Shape], /) -> Array[Shape]: ...
def asinh[Shape: _Shape](x: Array[Shape], /) -> Array[Shape]: ...
def atan[Shape: _Shape](x: Array[Shape], /) -> Array[Shape]: ...
def atanh[Shape: _Shape](x: Array[Shape], /) -> Array[Shape]: ...
def bitwise_count[Shape: _Shape](x: Array[Shape], /) -> Array[Shape]: ...
def bitwise_invert[Shape: _Shape](x: Array[Shape], /) -> Array[Shape]: ...
def bitwise_not[Shape: _Shape](x: Array[Shape], /) -> Array[Shape]: ...
def cbrt[Shape: _Shape](x: Array[Shape], /) -> Array[Shape]: ...
def ceil[Shape: _Shape](x: Array[Shape], /) -> Array[Shape]: ...
def conj[Shape: _Shape](x: Array[Shape], /) -> Array[Shape]: ...
def conjugate[Shape: _Shape](x: Array[Shape], /) -> Array[Shape]: ...
def cos[Shape: _Shape](x: Array[Shape], /) -> Array[Shape]: ...
def cosh[Shape: _Shape](x: Array[Shape], /) -> Array[Shape]: ...
def deg2rad[Shape: _Shape](x: Array[Shape], /) -> Array[Shape]: ...
def degrees[Shape: _Shape](x: Array[Shape], /) -> Array[Shape]: ...
def exp[Shape: _Shape](x: Array[Shape], /) -> Array[Shape]: ...
def exp2[Shape: _Shape](x: Array[Shape], /) -> Array[Shape]: ...
def expm1[Shape: _Shape](x: Array[Shape], /) -> Array[Shape]: ...
def fabs[Shape: _Shape](x: Array[Shape], /) -> Array[Shape]: ...
def floor[Shape: _Shape](x: Array[Shape], /) -> Array[Shape]: ...
def frexp[Shape: _Shape](x: Array[Shape], /) -> tuple[Array[Shape], Array[Shape]]: ...
def i0[Shape: _Shape](x: Array[Shape], /) -> Array[Shape]: ...
def imag[Shape: _Shape](val: Array[Shape], /) -> Array[Shape]: ...
def invert[Shape: _Shape](x: Array[Shape], /) -> Array[Shape]: ...
def log[Shape: _Shape](x: Array[Shape], /) -> Array[Shape]: ...
def log10[Shape: _Shape](x: Array[Shape], /) -> Array[Shape]: ...
def log1p[Shape: _Shape](x: Array[Shape], /) -> Array[Shape]: ...
def log2[Shape: _Shape](x: Array[Shape], /) -> Array[Shape]: ...
def modf[Shape: _Shape](x: Array[Shape], /) -> tuple[Array[Shape], Array[Shape]]: ...
def negative[Shape: _Shape](x: Array[Shape], /) -> Array[Shape]: ...
def positive[Shape: _Shape](x: Array[Shape], /) -> Array[Shape]: ...
def rad2deg[Shape: _Shape](x: Array[Shape], /) -> Array[Shape]: ...
def radians[Shape: _Shape](x: Array[Shape], /) -> Array[Shape]: ...
def real[Shape: _Shape](val: Array[Shape], /) -> Array[Shape]: ...
def reciprocal[Shape: _Shape](x: Array[Shape], /) -> Array[Shape]: ...
def rint[Shape: _Shape](x: Array[Shape], /) -> Array[Shape]: ...
def round[Shape: _Shape](a: Array[Shape], decimals: int = 0) -> Array[Shape]: ...
def sign[Shape: _Shape](x: Array[Shape], /) -> Array[Shape]: ...
def signbit[Shape: _Shape](x: Array[Shape], /) -> Array[Shape]: ...
def sin[Shape: _Shape](x: Array[Shape], /) -> Array[Shape]: ...
def sinc[Shape: _Shape](x: Array[Shape], /) -> Array[Shape]: ...
def sinh[Shape: _Shape](x: Array[Shape], /) -> Array[Shape]: ...
def spacing[Shape: _Shape](x: Array[Shape], /) -> Array[Shape]: ...
def sqrt[Shape: _Shape](x: Array[Shape], /) -> Array[Shape]: ...
def square[Shape: _Shape](x: Array[Shape], /) -> Array[Shape]: ...
def tan[Shape: _Shape](x: Array[Shape], /) -> Array[Shape]: ...
def tanh[Shape: _Shape](x: Array[Shape], /) -> Array[Shape]: ...
def trunc[Shape: _Shape](x: Array[Shape], /) -> Array[Shape]: ...
def unwrap[Shape: _Shape](
    p: Array[Shape],
    discont: Any = None,
    axis: int = -1,
    period: Any = ...,
) -> Array[Shape]: ...

# Broadcasting elementwise binary functions. Each takes a scalar in either
# position as well as an array: rejecting `jnp.add(a, 1)` would flag valid code.
@overload
def add[Shape: _Shape](
    x1: Array[Shape], x2: int | float | complex, /
) -> Array[Shape]: ...
@overload
def add[Shape: _Shape](
    x1: int | float | complex, x2: Array[Shape], /
) -> Array[Shape]: ...
@overload
def add[Shape1: _Shape, Shape2: _Shape](
    x1: Array[Shape1], x2: Array[Shape2], /
) -> Array[broadcast(Shape1, Shape2)]: ...
@overload
def arctan2[Shape: _Shape](
    x1: Array[Shape], x2: int | float | complex, /
) -> Array[Shape]: ...
@overload
def arctan2[Shape: _Shape](
    x1: int | float | complex, x2: Array[Shape], /
) -> Array[Shape]: ...
@overload
def arctan2[Shape1: _Shape, Shape2: _Shape](
    x1: Array[Shape1], x2: Array[Shape2], /
) -> Array[broadcast(Shape1, Shape2)]: ...
@overload
def atan2[Shape: _Shape](
    x1: Array[Shape], x2: int | float | complex, /
) -> Array[Shape]: ...
@overload
def atan2[Shape: _Shape](
    x1: int | float | complex, x2: Array[Shape], /
) -> Array[Shape]: ...
@overload
def atan2[Shape1: _Shape, Shape2: _Shape](
    x1: Array[Shape1], x2: Array[Shape2], /
) -> Array[broadcast(Shape1, Shape2)]: ...
@overload
def bitwise_and[Shape: _Shape](
    x1: Array[Shape], x2: int | float | complex, /
) -> Array[Shape]: ...
@overload
def bitwise_and[Shape: _Shape](
    x1: int | float | complex, x2: Array[Shape], /
) -> Array[Shape]: ...
@overload
def bitwise_and[Shape1: _Shape, Shape2: _Shape](
    x1: Array[Shape1], x2: Array[Shape2], /
) -> Array[broadcast(Shape1, Shape2)]: ...
@overload
def bitwise_left_shift[Shape: _Shape](
    x1: Array[Shape], x2: int | float | complex, /
) -> Array[Shape]: ...
@overload
def bitwise_left_shift[Shape: _Shape](
    x1: int | float | complex, x2: Array[Shape], /
) -> Array[Shape]: ...
@overload
def bitwise_left_shift[Shape1: _Shape, Shape2: _Shape](
    x1: Array[Shape1], x2: Array[Shape2], /
) -> Array[broadcast(Shape1, Shape2)]: ...
@overload
def bitwise_or[Shape: _Shape](
    x1: Array[Shape], x2: int | float | complex, /
) -> Array[Shape]: ...
@overload
def bitwise_or[Shape: _Shape](
    x1: int | float | complex, x2: Array[Shape], /
) -> Array[Shape]: ...
@overload
def bitwise_or[Shape1: _Shape, Shape2: _Shape](
    x1: Array[Shape1], x2: Array[Shape2], /
) -> Array[broadcast(Shape1, Shape2)]: ...
@overload
def bitwise_right_shift[Shape: _Shape](
    x1: Array[Shape], x2: int | float | complex, /
) -> Array[Shape]: ...
@overload
def bitwise_right_shift[Shape: _Shape](
    x1: int | float | complex, x2: Array[Shape], /
) -> Array[Shape]: ...
@overload
def bitwise_right_shift[Shape1: _Shape, Shape2: _Shape](
    x1: Array[Shape1], x2: Array[Shape2], /
) -> Array[broadcast(Shape1, Shape2)]: ...
@overload
def bitwise_xor[Shape: _Shape](
    x1: Array[Shape], x2: int | float | complex, /
) -> Array[Shape]: ...
@overload
def bitwise_xor[Shape: _Shape](
    x1: int | float | complex, x2: Array[Shape], /
) -> Array[Shape]: ...
@overload
def bitwise_xor[Shape1: _Shape, Shape2: _Shape](
    x1: Array[Shape1], x2: Array[Shape2], /
) -> Array[broadcast(Shape1, Shape2)]: ...
@overload
def copysign[Shape: _Shape](
    x1: Array[Shape], x2: int | float | complex, /
) -> Array[Shape]: ...
@overload
def copysign[Shape: _Shape](
    x1: int | float | complex, x2: Array[Shape], /
) -> Array[Shape]: ...
@overload
def copysign[Shape1: _Shape, Shape2: _Shape](
    x1: Array[Shape1], x2: Array[Shape2], /
) -> Array[broadcast(Shape1, Shape2)]: ...
@overload
def divide[Shape: _Shape](
    x1: Array[Shape], x2: int | float | complex, /
) -> Array[Shape]: ...
@overload
def divide[Shape: _Shape](
    x1: int | float | complex, x2: Array[Shape], /
) -> Array[Shape]: ...
@overload
def divide[Shape1: _Shape, Shape2: _Shape](
    x1: Array[Shape1], x2: Array[Shape2], /
) -> Array[broadcast(Shape1, Shape2)]: ...
@overload
def divmod[Shape: _Shape](
    x1: Array[Shape], x2: int | float | complex, /
) -> tuple[Array[Shape], Array[Shape]]: ...
@overload
def divmod[Shape: _Shape](
    x1: int | float | complex, x2: Array[Shape], /
) -> tuple[Array[Shape], Array[Shape]]: ...
@overload
def divmod[Shape1: _Shape, Shape2: _Shape](
    x1: Array[Shape1], x2: Array[Shape2], /
) -> tuple[Array[broadcast(Shape1, Shape2)], Array[broadcast(Shape1, Shape2)]]: ...
@overload
def float_power[Shape: _Shape](
    x1: Array[Shape], x2: int | float | complex, /
) -> Array[Shape]: ...
@overload
def float_power[Shape: _Shape](
    x1: int | float | complex, x2: Array[Shape], /
) -> Array[Shape]: ...
@overload
def float_power[Shape1: _Shape, Shape2: _Shape](
    x1: Array[Shape1], x2: Array[Shape2], /
) -> Array[broadcast(Shape1, Shape2)]: ...
@overload
def floor_divide[Shape: _Shape](
    x1: Array[Shape], x2: int | float | complex, /
) -> Array[Shape]: ...
@overload
def floor_divide[Shape: _Shape](
    x1: int | float | complex, x2: Array[Shape], /
) -> Array[Shape]: ...
@overload
def floor_divide[Shape1: _Shape, Shape2: _Shape](
    x1: Array[Shape1], x2: Array[Shape2], /
) -> Array[broadcast(Shape1, Shape2)]: ...
@overload
def fmod[Shape: _Shape](
    x1: Array[Shape], x2: int | float | complex, /
) -> Array[Shape]: ...
@overload
def fmod[Shape: _Shape](
    x1: int | float | complex, x2: Array[Shape], /
) -> Array[Shape]: ...
@overload
def fmod[Shape1: _Shape, Shape2: _Shape](
    x1: Array[Shape1], x2: Array[Shape2], /
) -> Array[broadcast(Shape1, Shape2)]: ...
@overload
def gcd[Shape: _Shape](
    x1: Array[Shape], x2: int | float | complex, /
) -> Array[Shape]: ...
@overload
def gcd[Shape: _Shape](
    x1: int | float | complex, x2: Array[Shape], /
) -> Array[Shape]: ...
@overload
def gcd[Shape1: _Shape, Shape2: _Shape](
    x1: Array[Shape1], x2: Array[Shape2], /
) -> Array[broadcast(Shape1, Shape2)]: ...
@overload
def heaviside[Shape: _Shape](
    x1: Array[Shape], x2: int | float | complex, /
) -> Array[Shape]: ...
@overload
def heaviside[Shape: _Shape](
    x1: int | float | complex, x2: Array[Shape], /
) -> Array[Shape]: ...
@overload
def heaviside[Shape1: _Shape, Shape2: _Shape](
    x1: Array[Shape1], x2: Array[Shape2], /
) -> Array[broadcast(Shape1, Shape2)]: ...
@overload
def hypot[Shape: _Shape](
    x1: Array[Shape], x2: int | float | complex, /
) -> Array[Shape]: ...
@overload
def hypot[Shape: _Shape](
    x1: int | float | complex, x2: Array[Shape], /
) -> Array[Shape]: ...
@overload
def hypot[Shape1: _Shape, Shape2: _Shape](
    x1: Array[Shape1], x2: Array[Shape2], /
) -> Array[broadcast(Shape1, Shape2)]: ...
@overload
def lcm[Shape: _Shape](
    x1: Array[Shape], x2: int | float | complex, /
) -> Array[Shape]: ...
@overload
def lcm[Shape: _Shape](
    x1: int | float | complex, x2: Array[Shape], /
) -> Array[Shape]: ...
@overload
def lcm[Shape1: _Shape, Shape2: _Shape](
    x1: Array[Shape1], x2: Array[Shape2], /
) -> Array[broadcast(Shape1, Shape2)]: ...
@overload
def ldexp[Shape: _Shape](
    x1: Array[Shape], x2: int | float | complex, /
) -> Array[Shape]: ...
@overload
def ldexp[Shape: _Shape](
    x1: int | float | complex, x2: Array[Shape], /
) -> Array[Shape]: ...
@overload
def ldexp[Shape1: _Shape, Shape2: _Shape](
    x1: Array[Shape1], x2: Array[Shape2], /
) -> Array[broadcast(Shape1, Shape2)]: ...
@overload
def left_shift[Shape: _Shape](
    x1: Array[Shape], x2: int | float | complex, /
) -> Array[Shape]: ...
@overload
def left_shift[Shape: _Shape](
    x1: int | float | complex, x2: Array[Shape], /
) -> Array[Shape]: ...
@overload
def left_shift[Shape1: _Shape, Shape2: _Shape](
    x1: Array[Shape1], x2: Array[Shape2], /
) -> Array[broadcast(Shape1, Shape2)]: ...
@overload
def logaddexp[Shape: _Shape](
    x1: Array[Shape], x2: int | float | complex, /
) -> Array[Shape]: ...
@overload
def logaddexp[Shape: _Shape](
    x1: int | float | complex, x2: Array[Shape], /
) -> Array[Shape]: ...
@overload
def logaddexp[Shape1: _Shape, Shape2: _Shape](
    x1: Array[Shape1], x2: Array[Shape2], /
) -> Array[broadcast(Shape1, Shape2)]: ...
@overload
def logaddexp2[Shape: _Shape](
    x1: Array[Shape], x2: int | float | complex, /
) -> Array[Shape]: ...
@overload
def logaddexp2[Shape: _Shape](
    x1: int | float | complex, x2: Array[Shape], /
) -> Array[Shape]: ...
@overload
def logaddexp2[Shape1: _Shape, Shape2: _Shape](
    x1: Array[Shape1], x2: Array[Shape2], /
) -> Array[broadcast(Shape1, Shape2)]: ...
@overload
def cross[
    Shape1: _Shape,
    Shape2: _Shape,
    Axis: Flag[int],
](
    a: Array[Shape1],
    b: Array[Shape2],
    /,
    axisa: int = -1,
    axisb: int = -1,
    axisc: int = -1,
    *,
    axis: Axis,
) -> Array[cross_axis_shape(Shape1, Shape2, Axis)]: ...
@overload
def cross[
    Shape1: _Shape,
    Shape2: _Shape,
    AxisA: Flag[int] = -1,
    AxisB: Flag[int] = -1,
    AxisC: Flag[int] = -1,
](
    a: Array[Shape1],
    b: Array[Shape2],
    /,
    axisa: AxisA = -1,
    axisb: AxisB = -1,
    axisc: AxisC = -1,
    axis: None = None,
) -> Array[cross_axes_shape(Shape1, Shape2, AxisA, AxisB, AxisC)]: ...
@overload
def cross(
    a: Array[Any],
    b: Array[Any],
    /,
    axisa: int = -1,
    axisb: int = -1,
    axisc: int = -1,
    axis: int | None = None,
) -> Array[IntTuple]: ...
@overload
def diagonal[
    Shape: _Shape,
    Offset: Flag[int] = 0,
    Axis1: Flag[int] = 0,
    Axis2: Flag[int] = 1,
](
    a: Array[Shape],
    offset: Offset = 0,
    axis1: Axis1 = 0,
    axis2: Axis2 = 1,
) -> Array[diagonal_shape(Shape, Offset, Axis1, Axis2)]: ...
@overload
def diagonal(
    a: Array[Any],
    offset: int = 0,
    axis1: int = 0,
    axis2: int = 1,
) -> Array[IntTuple]: ...
def dot[LeftShape: _Shape, RightShape: _Shape](
    a: Array[LeftShape],
    b: Array[RightShape],
    *,
    precision: Any = None,
    preferred_element_type: Any = None,
    out_sharding: Any = None,
) -> Array[dot_shape(LeftShape, RightShape)]: ...
@overload
def einsum[Spec: Flag[str], Shapes: IntTuples](
    subscripts: Spec,
    /,
    *operands: Unpack[MapIntTuples[lambda S: Array[S], Shapes]],
    out: None = None,
    optimize: str | bool | Sequence[tuple[int, ...]] = "auto",
    precision: Any = None,
    preferred_element_type: Any = None,
    _dot_general: Any = ...,
    out_sharding: Any = None,
) -> Array[einsum_shape(Spec, Shapes)]: ...
@overload
def einsum(
    subscripts: str,
    /,
    *operands: Array[Any] | Sequence[Any],
    out: None = None,
    optimize: str | bool | Sequence[tuple[int, ...]] = "auto",
    precision: Any = None,
    preferred_element_type: Any = None,
    _dot_general: Any = ...,
    out_sharding: Any = None,
) -> Array[IntTuple]: ...
def einsum_path(
    subscripts: str,
    /,
    *operands: Array[Any] | Sequence[Any],
    optimize: bool | str | Sequence[tuple[int, ...]] = "auto",
) -> tuple[list[tuple[int, ...]], Any]: ...
def inner[LeftShape: _Shape, RightShape: _Shape](
    a: Array[LeftShape],
    b: Array[RightShape],
    *,
    precision: Any = None,
    preferred_element_type: Any = None,
) -> Array[inner_shape(LeftShape, RightShape)]: ...
def kron[AShape: _Shape, BShape: _Shape](
    a: Array[AShape],
    b: Array[BShape],
) -> Array[kron_shape(AShape, BShape)]: ...
def matmul[LeftShape: _Shape, RightShape: _Shape](
    a: Array[LeftShape], b: Array[RightShape]
) -> Array[matmul_shape(LeftShape, RightShape)]: ...
def matvec[LeftShape: _Shape, RightShape: _Shape](
    x1: Array[LeftShape],
    x2: Array[RightShape],
    /,
) -> Array[matvec_shape(LeftShape, RightShape)]: ...
@overload
def outer[M: IntVar, N: IntVar](
    a: Array[[M]],
    b: Array[[N]],
    out: None = None,
) -> Array[[M, N]]: ...
@overload
def outer(
    a: Array[Any] | Sequence[Any],
    b: Array[Any] | Sequence[Any],
    out: None = None,
) -> Array[IntTuple]: ...
@overload
def tensordot[Left: _Shape, Right: _Shape, Dims: Flag[int] = 2](
    a: Array[Left],
    b: Array[Right],
    axes: Dims = 2,
    *,
    precision: Any = None,
    preferred_element_type: Any = None,
    out_sharding: Any = None,
) -> Array[tensordot_shape(Left, Right, Dims)]: ...
@overload
def tensordot(
    a: Array[Any],
    b: Array[Any],
    axes: int | Sequence[int] | Sequence[Sequence[int]] = 2,
    *,
    precision: Any = None,
    preferred_element_type: Any = None,
    out_sharding: Any = None,
) -> Array[IntTuple]: ...
@overload
def trace[
    Shape: _Shape,
    Offset: Flag[int] = 0,
    Axis1: Flag[int] = 0,
    Axis2: Flag[int] = 1,
](
    a: Array[Shape],
    offset: Offset = 0,
    axis1: Axis1 = 0,
    axis2: Axis2 = 1,
    dtype: DTypeLike | None = None,
    out: None = None,
) -> Array[trace_shape(Shape, Offset, Axis1, Axis2)]: ...
@overload
def trace(
    a: Array[Any],
    offset: int = 0,
    axis1: int = 0,
    axis2: int = 1,
    dtype: DTypeLike | None = None,
    out: None = None,
) -> Array[IntTuple]: ...
def vdot[Shape1: _Shape, Shape2: _Shape](
    a: Array[Shape1],
    b: Array[Shape2],
    *,
    precision: Any = None,
    preferred_element_type: Any = None,
) -> Array[[]]: ...
def vecdot[Shape1: _Shape, Shape2: _Shape, Axis: Flag[_Axis] = -1](
    x1: Array[Shape1],
    x2: Array[Shape2],
    /,
    *,
    axis: Axis = -1,
    precision: Any = None,
    preferred_element_type: Any = None,
) -> Array[reduce_shape(broadcast(Shape1, Shape2), Axis, False)]: ...
def vecmat[LeftShape: _Shape, RightShape: _Shape](
    x1: Array[LeftShape],
    x2: Array[RightShape],
    /,
) -> Array[vecmat_shape(LeftShape, RightShape)]: ...
@overload
def maximum[Shape: _Shape](
    x1: Array[Shape], x2: int | float | complex, /
) -> Array[Shape]: ...
@overload
def maximum[Shape: _Shape](
    x1: int | float | complex, x2: Array[Shape], /
) -> Array[Shape]: ...
@overload
def maximum[Shape1: _Shape, Shape2: _Shape](
    x1: Array[Shape1], x2: Array[Shape2], /
) -> Array[broadcast(Shape1, Shape2)]: ...
@overload
def minimum[Shape: _Shape](
    x1: Array[Shape], x2: int | float | complex, /
) -> Array[Shape]: ...
@overload
def minimum[Shape: _Shape](
    x1: int | float | complex, x2: Array[Shape], /
) -> Array[Shape]: ...
@overload
def minimum[Shape1: _Shape, Shape2: _Shape](
    x1: Array[Shape1], x2: Array[Shape2], /
) -> Array[broadcast(Shape1, Shape2)]: ...
@overload
def mod[Shape: _Shape](
    x1: Array[Shape], x2: int | float | complex, /
) -> Array[Shape]: ...
@overload
def mod[Shape: _Shape](
    x1: int | float | complex, x2: Array[Shape], /
) -> Array[Shape]: ...
@overload
def mod[Shape1: _Shape, Shape2: _Shape](
    x1: Array[Shape1], x2: Array[Shape2], /
) -> Array[broadcast(Shape1, Shape2)]: ...
@overload
def multiply[Shape: _Shape](
    x1: Array[Shape], x2: int | float | complex, /
) -> Array[Shape]: ...
@overload
def multiply[Shape: _Shape](
    x1: int | float | complex, x2: Array[Shape], /
) -> Array[Shape]: ...
@overload
def multiply[Shape1: _Shape, Shape2: _Shape](
    x1: Array[Shape1], x2: Array[Shape2], /
) -> Array[broadcast(Shape1, Shape2)]: ...
@overload
def nextafter[Shape: _Shape](
    x1: Array[Shape], x2: int | float | complex, /
) -> Array[Shape]: ...
@overload
def nextafter[Shape: _Shape](
    x1: int | float | complex, x2: Array[Shape], /
) -> Array[Shape]: ...
@overload
def nextafter[Shape1: _Shape, Shape2: _Shape](
    x1: Array[Shape1], x2: Array[Shape2], /
) -> Array[broadcast(Shape1, Shape2)]: ...
@overload
def pow[Shape: _Shape](
    x1: Array[Shape], x2: int | float | complex, /
) -> Array[Shape]: ...
@overload
def pow[Shape: _Shape](
    x1: int | float | complex, x2: Array[Shape], /
) -> Array[Shape]: ...
@overload
def pow[Shape1: _Shape, Shape2: _Shape](
    x1: Array[Shape1], x2: Array[Shape2], /
) -> Array[broadcast(Shape1, Shape2)]: ...
@overload
def power[Shape: _Shape](
    x1: Array[Shape], x2: int | float | complex, /
) -> Array[Shape]: ...
@overload
def power[Shape: _Shape](
    x1: int | float | complex, x2: Array[Shape], /
) -> Array[Shape]: ...
@overload
def power[Shape1: _Shape, Shape2: _Shape](
    x1: Array[Shape1], x2: Array[Shape2], /
) -> Array[broadcast(Shape1, Shape2)]: ...
@overload
def remainder[Shape: _Shape](
    x1: Array[Shape], x2: int | float | complex, /
) -> Array[Shape]: ...
@overload
def remainder[Shape: _Shape](
    x1: int | float | complex, x2: Array[Shape], /
) -> Array[Shape]: ...
@overload
def remainder[Shape1: _Shape, Shape2: _Shape](
    x1: Array[Shape1], x2: Array[Shape2], /
) -> Array[broadcast(Shape1, Shape2)]: ...
@overload
def right_shift[Shape: _Shape](
    x1: Array[Shape], x2: int | float | complex, /
) -> Array[Shape]: ...
@overload
def right_shift[Shape: _Shape](
    x1: int | float | complex, x2: Array[Shape], /
) -> Array[Shape]: ...
@overload
def right_shift[Shape1: _Shape, Shape2: _Shape](
    x1: Array[Shape1], x2: Array[Shape2], /
) -> Array[broadcast(Shape1, Shape2)]: ...
@overload
def subtract[Shape: _Shape](
    x1: Array[Shape], x2: int | float | complex, /
) -> Array[Shape]: ...
@overload
def subtract[Shape: _Shape](
    x1: int | float | complex, x2: Array[Shape], /
) -> Array[Shape]: ...
@overload
def subtract[Shape1: _Shape, Shape2: _Shape](
    x1: Array[Shape1], x2: Array[Shape2], /
) -> Array[broadcast(Shape1, Shape2)]: ...
@overload
def true_divide[Shape: _Shape](
    x1: Array[Shape], x2: int | float | complex, /
) -> Array[Shape]: ...
@overload
def true_divide[Shape: _Shape](
    x1: int | float | complex, x2: Array[Shape], /
) -> Array[Shape]: ...
@overload
def true_divide[Shape1: _Shape, Shape2: _Shape](
    x1: Array[Shape1], x2: Array[Shape2], /
) -> Array[broadcast(Shape1, Shape2)]: ...
@overload
def transpose[Shape: _Shape](
    a: Array[Shape], axes: None = None
) -> Array[reverse_shape(Shape)]: ...
@overload
def transpose[Shape: _Shape, Axes: Flag[_Axis]](
    a: Array[Shape], axes: Axes = None
) -> Array[permute_shape(Shape, Axes)]: ...
@overload
def transpose[Shape: _Shape](
    a: Array[Shape], axes: Sequence[int]
) -> Array[IntTuple]: ...
@overload
def permute_dims[Shape: _Shape, Axes: Flag[_Axis]](
    a: Array[Shape], /, axes: Axes
) -> Array[permute_shape(Shape, Axes)]: ...
@overload
def permute_dims[Shape: _Shape](
    a: Array[Shape], /, axes: Sequence[int]
) -> Array[IntTuple]: ...
def matrix_transpose[Batch: IntTuple, M: IntVar, N: IntVar](
    x: Array[[*Elements[Batch], M, N]],
    /,
) -> Array[[*Elements[Batch], N, M]]: ...

# A single int or tuple, matching JAX: the free function is not variadic, so
# `jnp.reshape(a, 2, 3)` is an error there. `Array.reshape` is the variadic one.
@overload
def reshape[Shape: _Shape, NewShape: Flag[_NewShape]](
    a: Array[Shape],
    shape: NewShape,
    order: str = ...,
    *,
    copy: bool | None = ...,
    out_sharding: Any = ...,
) -> Array[reshape_shape(Shape, NewShape)]: ...
@overload
def reshape[NewShape: _Shape](
    a: Array[Any],
    shape: NewShape,
    order: str = ...,
    *,
    copy: bool | None = ...,
    out_sharding: Any = ...,
) -> Array[NewShape]: ...
@overload
def reshape(
    a: Array[Any],
    shape: Sequence[int],
    order: str = ...,
    *,
    copy: bool | None = ...,
    out_sharding: Any = ...,
) -> Array[IntTuple]: ...
def ravel[Shape: _Shape](
    a: Array[Shape],
    order: str = "C",
) -> Array[ravel_shape(Shape)]: ...
@overload
def squeeze[Shape: _Shape, Axis: Flag[_Axis] = None](
    a: Array[Shape],
    axis: Axis = None,
) -> Array[squeeze_shape(Shape, Axis)]: ...
@overload
def squeeze(
    a: Array[Any],
    axis: Sequence[int] | None = None,
) -> Array[IntTuple]: ...
@overload
def expand_dims[Shape: _Shape, Axis: Flag[int]](
    a: Array[Shape],
    axis: Axis,
) -> Array[expand_dims_shape(Shape, Axis)]: ...
@overload
def expand_dims(
    a: Array[Any],
    axis: int | Sequence[int],
) -> Array[IntTuple]: ...
@overload
def broadcast_to[Shape: _Shape, TargetShape: Flag[_NewShape]](
    array: Array[Shape],
    shape: TargetShape,
) -> Array[broadcast_to_shape(Shape, TargetShape)]: ...
@overload
def broadcast_to(
    array: Array[Any],
    shape: Sequence[int] | int,
) -> Array[IntTuple]: ...
@overload
def broadcast_arrays[Shape: _Shape](
    a: Array[Shape],
    /,
) -> tuple[Array[Shape]]: ...
@overload
def broadcast_arrays[Shape: _Shape](
    a1: Array[Shape],
    a2: _Scalar,
    /,
) -> tuple[Array[Shape], Array[Shape]]: ...
@overload
def broadcast_arrays[Shape: _Shape](
    a1: _Scalar,
    a2: Array[Shape],
    /,
) -> tuple[Array[Shape], Array[Shape]]: ...
@overload
def broadcast_arrays[Shape1: _Shape, Shape2: _Shape](
    a1: Array[Shape1],
    a2: Array[Shape2],
    /,
) -> tuple[Array[broadcast(Shape1, Shape2)], Array[broadcast(Shape1, Shape2)]]: ...
@overload
def broadcast_arrays(*args: Any) -> tuple[Array[IntTuple], ...]: ...
def broadcast_shapes(*shapes: Sequence[int]) -> tuple[int, ...]: ...
@overload
def concatenate[Shapes: IntTuples, Axis: Flag[int] = 0](
    arrays: MapIntTuples[lambda S: Array[S], Shapes],
    axis: Axis = 0,
    dtype: DTypeLike | None = None,
) -> Array[concatenate_shape(Shapes, Axis)]: ...
@overload
def concatenate(
    arrays: Any,
    axis: int | None = 0,
    dtype: DTypeLike | None = None,
) -> Array[IntTuple]: ...

concat = concatenate

@overload
def append[Shape1: _Shape, Shape2: _Shape, Axis: Flag[int | None] = None](
    arr: Array[Shape1],
    values: Array[Shape2],
    axis: Axis = None,
) -> Array[append_shape(Shape1, Shape2, Axis)]: ...
@overload
def append(
    arr: Any,
    values: Any,
    axis: int | None = None,
) -> Array[IntTuple]: ...
@overload
def stack[Shapes: IntTuples, Axis: Flag[int] = 0](
    arrays: MapIntTuples[lambda S: Array[S], Shapes],
    axis: Axis = 0,
    dtype: DTypeLike | None = None,
    *,
    out: Any = None,
) -> Array[stack_shape(Shapes, Axis)]: ...
@overload
def stack(
    arrays: Any,
    axis: int = 0,
    dtype: DTypeLike | None = None,
    *,
    out: Any = None,
) -> Array[IntTuple]: ...
@overload
def vstack[Shapes: IntTuples](
    tup: MapIntTuples[lambda S: Array[S], Shapes],
    *,
    dtype: DTypeLike | None = None,
) -> Array[vstack_shape(Shapes)]: ...
@overload
def vstack(
    tup: Any,
    *,
    dtype: DTypeLike | None = None,
) -> Array[IntTuple]: ...
@overload
def hstack[Shapes: IntTuples](
    tup: MapIntTuples[lambda S: Array[S], Shapes],
    *,
    dtype: DTypeLike | None = None,
) -> Array[hstack_shape(Shapes)]: ...
@overload
def hstack(
    tup: Any,
    *,
    dtype: DTypeLike | None = None,
) -> Array[IntTuple]: ...
@overload
def column_stack[Shapes: IntTuples](
    tup: MapIntTuples[lambda S: Array[S], Shapes],
) -> Array[column_stack_shape(Shapes)]: ...
@overload
def column_stack(
    tup: Any,
) -> Array[IntTuple]: ...
@overload
def dstack[Shapes: IntTuples](
    tup: MapIntTuples[lambda S: Array[S], Shapes],
    *,
    dtype: DTypeLike | None = None,
) -> Array[dstack_shape(Shapes)]: ...
@overload
def dstack(
    tup: Any,
    *,
    dtype: DTypeLike | None = None,
) -> Array[IntTuple]: ...
def block(arrays: Any) -> Array[IntTuple]: ...
def array_split[Batch: IntTuple, M: IntVar](
    ary: Array[[*Elements[Batch], M]],
    indices_or_sections: int | Sequence[int] | Array[Any],
    axis: int = 0,
) -> list[Array[IntTuple]]: ...
def split[Batch: IntTuple, M: IntVar](
    ary: Array[[*Elements[Batch], M]],
    indices_or_sections: int | Sequence[int] | Array[Any],
    axis: int = 0,
) -> list[Array[IntTuple]]: ...
def dsplit[Batch: IntTuple, M: IntVar, N: IntVar, P: IntVar](
    ary: Array[[*Elements[Batch], M, N, P]],
    indices_or_sections: int | Sequence[int] | Array[Any],
) -> list[Array[IntTuple]]: ...
def hsplit[Batch: IntTuple, M: IntVar](
    ary: Array[[*Elements[Batch], M]],
    indices_or_sections: int | Sequence[int] | Array[Any],
) -> list[Array[IntTuple]]: ...
def vsplit[Batch: IntTuple, M: IntVar](
    ary: Array[[*Elements[Batch], M]],
    indices_or_sections: int | Sequence[int] | Array[Any],
) -> list[Array[IntTuple]]: ...
def unstack[Batch: IntTuple, M: IntVar](
    x: Array[[*Elements[Batch], M]],
    /,
    *,
    axis: int = 0,
) -> tuple[Array[IntTuple], ...]: ...
def pad(
    array: Array[Any] | _Scalar,
    pad_width: Any,
    mode: str | Callable[..., Any] = "constant",
    **kwargs: Any,
) -> Array[IntTuple]: ...
def repeat(
    a: Array[Any] | _Scalar,
    repeats: Array[Any] | int | Sequence[int],
    axis: int | None = None,
    *,
    total_repeat_length: int | None = None,
) -> Array[IntTuple]: ...
@overload
def resize(a: Array[Any] | _Scalar, new_shape: tuple[()]) -> Array[[]]: ...
@overload
def resize[N: IntVar](a: Array[Any] | _Scalar, new_shape: Int[N]) -> Array[[N]]: ...
@overload
def resize[Shape: _Shape](
    a: Array[Any] | _Scalar, new_shape: Shape
) -> Array[Shape]: ...
@overload
def resize(
    a: Array[Any] | _Scalar, new_shape: Sequence[int] | int
) -> Array[IntTuple]: ...
def tile(
    A: Array[Any] | _Scalar,
    reps: int | Sequence[int],
) -> Array[IntTuple]: ...
@overload
def swapaxes[Shape: _Shape, Axis1: Flag[int], Axis2: Flag[int]](
    a: Array[Shape],
    axis1: Axis1,
    axis2: Axis2,
) -> Array[swapaxes_shape(Shape, Axis1, Axis2)]: ...
@overload
def swapaxes(
    a: Array[Any],
    axis1: int,
    axis2: int,
) -> Array[IntTuple]: ...
@overload
def moveaxis[Shape: _Shape, Source: Flag[int], Destination: Flag[int]](
    a: Array[Shape],
    source: Source,
    destination: Destination,
) -> Array[moveaxis_shape(Shape, Source, Destination)]: ...
@overload
def moveaxis(
    a: Array[Any],
    source: int | Sequence[int],
    destination: int | Sequence[int],
) -> Array[IntTuple]: ...
@overload
def rollaxis[Shape: _Shape, Axis: Flag[int], Start: Flag[int] = 0](
    a: Array[Shape],
    axis: Axis,
    start: Start = 0,
) -> Array[rollaxis_shape(Shape, Axis, Start)]: ...
@overload
def rollaxis(
    a: Array[Any],
    axis: int,
    start: int = 0,
) -> Array[IntTuple]: ...
@overload
def flip[Shape: _Shape, Axis: Flag[_Axis] = None](
    m: Array[Shape],
    axis: Axis = None,
) -> Array[flip_shape(Shape, Axis)]: ...
@overload
def flip[Shape: _Shape](
    m: Array[Shape],
    axis: Sequence[int] | None = None,
) -> Array[Shape]: ...
def fliplr[Batch: IntTuple, M: IntVar, N: IntVar](
    m: Array[[*Elements[Batch], M, N]],
) -> Array[[*Elements[Batch], M, N]]: ...
def flipud[Batch: IntTuple, M: IntVar](
    m: Array[[*Elements[Batch], M]],
) -> Array[[*Elements[Batch], M]]: ...
@overload
def roll[Shape: _Shape, Axis: Flag[_Axis] = None](
    a: Array[Shape],
    shift: Any,
    axis: Axis = None,
) -> Array[roll_shape(Shape, Axis)]: ...
@overload
def roll[Shape: _Shape](
    a: Array[Shape],
    shift: Any,
    axis: Sequence[int] | None = None,
) -> Array[Shape]: ...
@overload
def rot90[Shape: _Shape, K: Flag[int] = 1, Axes: Flag[tuple[int, int]] = (0, 1)](
    m: Array[Shape],
    k: K = 1,
    axes: Axes = (0, 1),
) -> Array[rot90_shape(Shape, K, Axes)]: ...
@overload
def rot90(
    m: Array[Any],
    k: int = 1,
    axes: tuple[int, int] = (0, 1),
) -> Array[IntTuple]: ...
@overload
def atleast_1d(ary: _Scalar, /) -> Array[[1]]: ...
@overload
def atleast_1d[Shape: _Shape](
    ary: Array[Shape], /
) -> Array[atleast_1d_shape(Shape)]: ...
@overload
def atleast_1d(ary: Any, /) -> Array[IntTuple]: ...
@overload
def atleast_1d(*arys: Any) -> list[Array[IntTuple]]: ...
@overload
def atleast_2d(ary: _Scalar, /) -> Array[[1, 1]]: ...
@overload
def atleast_2d[Shape: _Shape](
    ary: Array[Shape], /
) -> Array[atleast_2d_shape(Shape)]: ...
@overload
def atleast_2d(ary: Any, /) -> Array[IntTuple]: ...
@overload
def atleast_2d(*arys: Any) -> list[Array[IntTuple]]: ...
@overload
def atleast_3d(ary: _Scalar, /) -> Array[[1, 1, 1]]: ...
@overload
def atleast_3d[Shape: _Shape](
    ary: Array[Shape], /
) -> Array[atleast_3d_shape(Shape)]: ...
@overload
def atleast_3d(ary: Any, /) -> Array[IntTuple]: ...
@overload
def atleast_3d(*arys: Any) -> list[Array[IntTuple]]: ...

# JAX accepts any integer sequence for an axis, but only a tuple is a Flag
# domain, so any other sequence yields a gradual shape. Rejecting it would flag
# valid code. The exact overload is declared first so that a tuple resolves to
# it rather than being absorbed by the fallback.
@overload
def sum[Shape: _Shape, Axis: Flag[_Axis], KeepDims: Flag[bool]](
    a: Array[Shape],
    axis: Axis = None,
    *,
    keepdims: KeepDims = False,
    dtype: DTypeLike | None = ...,
    out: Any = ...,
    initial: Any = ...,
    where: Any = ...,
    promote_integers: bool = ...,
) -> Array[reduce_shape(Shape, Axis, KeepDims)]: ...
@overload
def sum[Shape: _Shape](
    a: Array[Shape],
    axis: Sequence[int],
    *,
    keepdims: bool = False,
    dtype: DTypeLike | None = ...,
    out: Any = ...,
    initial: Any = ...,
    where: Any = ...,
    promote_integers: bool = ...,
) -> Array[IntTuple]: ...
@overload
def prod[Shape: _Shape, Axis: Flag[_Axis], KeepDims: Flag[bool]](
    a: Array[Shape],
    axis: Axis = None,
    *,
    keepdims: KeepDims = False,
    dtype: DTypeLike | None = ...,
    out: Any = ...,
    initial: Any = ...,
    where: Any = ...,
    promote_integers: bool = ...,
) -> Array[reduce_shape(Shape, Axis, KeepDims)]: ...
@overload
def prod[Shape: _Shape](
    a: Array[Shape],
    axis: Sequence[int],
    *,
    keepdims: bool = False,
    dtype: DTypeLike | None = ...,
    out: Any = ...,
    initial: Any = ...,
    where: Any = ...,
    promote_integers: bool = ...,
) -> Array[IntTuple]: ...
@overload
def mean[Shape: _Shape, Axis: Flag[_Axis], KeepDims: Flag[bool]](
    a: Array[Shape],
    axis: Axis = None,
    *,
    keepdims: KeepDims = False,
    dtype: DTypeLike | None = ...,
    out: Any = ...,
    initial: Any = ...,
    where: Any = ...,
    promote_integers: bool = ...,
) -> Array[reduce_shape(Shape, Axis, KeepDims)]: ...
@overload
def mean[Shape: _Shape](
    a: Array[Shape],
    axis: Sequence[int],
    *,
    keepdims: bool = False,
    dtype: DTypeLike | None = ...,
    out: Any = ...,
    initial: Any = ...,
    where: Any = ...,
    promote_integers: bool = ...,
) -> Array[IntTuple]: ...
@overload
def max[Shape: _Shape, Axis: Flag[_Axis], KeepDims: Flag[bool]](
    a: Array[Shape],
    axis: Axis = None,
    *,
    keepdims: KeepDims = False,
    dtype: DTypeLike | None = ...,
    out: Any = ...,
    initial: Any = ...,
    where: Any = ...,
    promote_integers: bool = ...,
) -> Array[reduce_shape(Shape, Axis, KeepDims)]: ...
@overload
def max[Shape: _Shape](
    a: Array[Shape],
    axis: Sequence[int],
    *,
    keepdims: bool = False,
    dtype: DTypeLike | None = ...,
    out: Any = ...,
    initial: Any = ...,
    where: Any = ...,
    promote_integers: bool = ...,
) -> Array[IntTuple]: ...
@overload
def min[Shape: _Shape, Axis: Flag[_Axis], KeepDims: Flag[bool]](
    a: Array[Shape],
    axis: Axis = None,
    *,
    keepdims: KeepDims = False,
    dtype: DTypeLike | None = ...,
    out: Any = ...,
    initial: Any = ...,
    where: Any = ...,
    promote_integers: bool = ...,
) -> Array[reduce_shape(Shape, Axis, KeepDims)]: ...
@overload
def min[Shape: _Shape](
    a: Array[Shape],
    axis: Sequence[int],
    *,
    keepdims: bool = False,
    dtype: DTypeLike | None = ...,
    out: Any = ...,
    initial: Any = ...,
    where: Any = ...,
    promote_integers: bool = ...,
) -> Array[IntTuple]: ...

# Boolean reductions
@overload
def all[Shape: _Shape, Axis: Flag[_Axis], KeepDims: Flag[bool]](
    a: Array[Shape],
    axis: Axis = None,
    out: Any = None,
    keepdims: KeepDims = False,
    *,
    where: Any = None,
) -> Array[reduce_shape(Shape, Axis, KeepDims)]: ...
@overload
def all[Shape: _Shape](
    a: Array[Shape],
    axis: Sequence[int],
    out: Any = None,
    keepdims: bool = False,
    *,
    where: Any = None,
) -> Array[IntTuple]: ...
@overload
def any[Shape: _Shape, Axis: Flag[_Axis], KeepDims: Flag[bool]](
    a: Array[Shape],
    axis: Axis = None,
    out: Any = None,
    keepdims: KeepDims = False,
    *,
    where: Any = None,
) -> Array[reduce_shape(Shape, Axis, KeepDims)]: ...
@overload
def any[Shape: _Shape](
    a: Array[Shape],
    axis: Sequence[int],
    out: Any = None,
    keepdims: bool = False,
    *,
    where: Any = None,
) -> Array[IntTuple]: ...

# Count nonzero
@overload
def count_nonzero[Shape: _Shape, Axis: Flag[_Axis], KeepDims: Flag[bool]](
    a: Array[Shape],
    axis: Axis = None,
    keepdims: KeepDims = False,
) -> Array[reduce_shape(Shape, Axis, KeepDims)]: ...
@overload
def count_nonzero[Shape: _Shape](
    a: Array[Shape],
    axis: Sequence[int],
    keepdims: bool = False,
) -> Array[IntTuple]: ...

# amax / amin aliases
@overload
def amax[Shape: _Shape, Axis: Flag[_Axis], KeepDims: Flag[bool]](
    a: Array[Shape],
    axis: Axis = None,
    out: Any = None,
    keepdims: KeepDims = False,
    initial: Any = None,
    where: Any = None,
) -> Array[reduce_shape(Shape, Axis, KeepDims)]: ...
@overload
def amax[Shape: _Shape](
    a: Array[Shape],
    axis: Sequence[int],
    out: Any = None,
    keepdims: bool = False,
    initial: Any = None,
    where: Any = None,
) -> Array[IntTuple]: ...
@overload
def amin[Shape: _Shape, Axis: Flag[_Axis], KeepDims: Flag[bool]](
    a: Array[Shape],
    axis: Axis = None,
    out: Any = None,
    keepdims: KeepDims = False,
    initial: Any = None,
    where: Any = None,
) -> Array[reduce_shape(Shape, Axis, KeepDims)]: ...
@overload
def amin[Shape: _Shape](
    a: Array[Shape],
    axis: Sequence[int],
    out: Any = None,
    keepdims: bool = False,
    initial: Any = None,
    where: Any = None,
) -> Array[IntTuple]: ...

# Standard deviation & variance
@overload
def std[Shape: _Shape, Axis: Flag[_Axis], KeepDims: Flag[bool]](
    a: Array[Shape],
    axis: Axis = None,
    dtype: DTypeLike | None = None,
    out: Any = None,
    ddof: int = 0,
    keepdims: KeepDims = False,
    *,
    where: Any = None,
    mean: Any = None,
    correction: Any = None,
) -> Array[reduce_shape(Shape, Axis, KeepDims)]: ...
@overload
def std[Shape: _Shape](
    a: Array[Shape],
    axis: Sequence[int],
    dtype: DTypeLike | None = None,
    out: Any = None,
    ddof: int = 0,
    keepdims: bool = False,
    *,
    where: Any = None,
    mean: Any = None,
    correction: Any = None,
) -> Array[IntTuple]: ...
@overload
def var[Shape: _Shape, Axis: Flag[_Axis], KeepDims: Flag[bool]](
    a: Array[Shape],
    axis: Axis = None,
    dtype: DTypeLike | None = None,
    out: Any = None,
    ddof: int = 0,
    keepdims: KeepDims = False,
    *,
    where: Any = None,
    mean: Any = None,
    correction: Any = None,
) -> Array[reduce_shape(Shape, Axis, KeepDims)]: ...
@overload
def var[Shape: _Shape](
    a: Array[Shape],
    axis: Sequence[int],
    dtype: DTypeLike | None = None,
    out: Any = None,
    ddof: int = 0,
    keepdims: bool = False,
    *,
    where: Any = None,
    mean: Any = None,
    correction: Any = None,
) -> Array[IntTuple]: ...

# Peak-to-peak (ptp)
@overload
def ptp[Shape: _Shape, Axis: Flag[_Axis], KeepDims: Flag[bool]](
    a: Array[Shape],
    axis: Axis = None,
    out: Any = None,
    keepdims: KeepDims = False,
) -> Array[reduce_shape(Shape, Axis, KeepDims)]: ...
@overload
def ptp[Shape: _Shape](
    a: Array[Shape],
    axis: Sequence[int],
    out: Any = None,
    keepdims: bool = False,
) -> Array[IntTuple]: ...

# Median
@overload
def median[Shape: _Shape, Axis: Flag[_Axis], KeepDims: Flag[bool]](
    a: Array[Shape],
    axis: Axis = None,
    out: Any = None,
    overwrite_input: bool = False,
    keepdims: KeepDims = False,
) -> Array[reduce_shape(Shape, Axis, KeepDims)]: ...
@overload
def median[Shape: _Shape](
    a: Array[Shape],
    axis: Sequence[int],
    out: Any = None,
    overwrite_input: bool = False,
    keepdims: bool = False,
) -> Array[IntTuple]: ...

# NaN-safe reductions
@overload
def nanmax[Shape: _Shape, Axis: Flag[_Axis], KeepDims: Flag[bool]](
    a: Array[Shape],
    axis: Axis = None,
    out: Any = None,
    keepdims: KeepDims = False,
    initial: Any = None,
    where: Any = None,
) -> Array[reduce_shape(Shape, Axis, KeepDims)]: ...
@overload
def nanmax[Shape: _Shape](
    a: Array[Shape],
    axis: Sequence[int],
    out: Any = None,
    keepdims: bool = False,
    initial: Any = None,
    where: Any = None,
) -> Array[IntTuple]: ...
@overload
def nanmin[Shape: _Shape, Axis: Flag[_Axis], KeepDims: Flag[bool]](
    a: Array[Shape],
    axis: Axis = None,
    out: Any = None,
    keepdims: KeepDims = False,
    initial: Any = None,
    where: Any = None,
) -> Array[reduce_shape(Shape, Axis, KeepDims)]: ...
@overload
def nanmin[Shape: _Shape](
    a: Array[Shape],
    axis: Sequence[int],
    out: Any = None,
    keepdims: bool = False,
    initial: Any = None,
    where: Any = None,
) -> Array[IntTuple]: ...
@overload
def nansum[Shape: _Shape, Axis: Flag[_Axis], KeepDims: Flag[bool]](
    a: Array[Shape],
    axis: Axis = None,
    dtype: DTypeLike | None = None,
    out: Any = None,
    keepdims: KeepDims = False,
    initial: Any = None,
    where: Any = None,
) -> Array[reduce_shape(Shape, Axis, KeepDims)]: ...
@overload
def nansum[Shape: _Shape](
    a: Array[Shape],
    axis: Sequence[int],
    dtype: DTypeLike | None = None,
    out: Any = None,
    keepdims: bool = False,
    initial: Any = None,
    where: Any = None,
) -> Array[IntTuple]: ...
@overload
def nanprod[Shape: _Shape, Axis: Flag[_Axis], KeepDims: Flag[bool]](
    a: Array[Shape],
    axis: Axis = None,
    dtype: DTypeLike | None = None,
    out: Any = None,
    keepdims: KeepDims = False,
    initial: Any = None,
    where: Any = None,
) -> Array[reduce_shape(Shape, Axis, KeepDims)]: ...
@overload
def nanprod[Shape: _Shape](
    a: Array[Shape],
    axis: Sequence[int],
    dtype: DTypeLike | None = None,
    out: Any = None,
    keepdims: bool = False,
    initial: Any = None,
    where: Any = None,
) -> Array[IntTuple]: ...
@overload
def nanmean[Shape: _Shape, Axis: Flag[_Axis], KeepDims: Flag[bool]](
    a: Array[Shape],
    axis: Axis = None,
    dtype: DTypeLike | None = None,
    out: Any = None,
    keepdims: KeepDims = False,
    where: Any = None,
) -> Array[reduce_shape(Shape, Axis, KeepDims)]: ...
@overload
def nanmean[Shape: _Shape](
    a: Array[Shape],
    axis: Sequence[int],
    dtype: DTypeLike | None = None,
    out: Any = None,
    keepdims: bool = False,
    where: Any = None,
) -> Array[IntTuple]: ...
@overload
def nanstd[Shape: _Shape, Axis: Flag[_Axis], KeepDims: Flag[bool]](
    a: Array[Shape],
    axis: Axis = None,
    dtype: DTypeLike | None = None,
    out: Any = None,
    ddof: int = 0,
    keepdims: KeepDims = False,
    where: Any = None,
    mean: Any = None,
) -> Array[reduce_shape(Shape, Axis, KeepDims)]: ...
@overload
def nanstd[Shape: _Shape](
    a: Array[Shape],
    axis: Sequence[int],
    dtype: DTypeLike | None = None,
    out: Any = None,
    ddof: int = 0,
    keepdims: bool = False,
    where: Any = None,
    mean: Any = None,
) -> Array[IntTuple]: ...
@overload
def nanvar[Shape: _Shape, Axis: Flag[_Axis], KeepDims: Flag[bool]](
    a: Array[Shape],
    axis: Axis = None,
    dtype: DTypeLike | None = None,
    out: Any = None,
    ddof: int = 0,
    keepdims: KeepDims = False,
    where: Any = None,
    mean: Any = None,
) -> Array[reduce_shape(Shape, Axis, KeepDims)]: ...
@overload
def nanvar[Shape: _Shape](
    a: Array[Shape],
    axis: Sequence[int],
    dtype: DTypeLike | None = None,
    out: Any = None,
    ddof: int = 0,
    keepdims: bool = False,
    where: Any = None,
    mean: Any = None,
) -> Array[IntTuple]: ...
@overload
def nanmedian[Shape: _Shape, Axis: Flag[_Axis], KeepDims: Flag[bool]](
    a: Array[Shape],
    axis: Axis = None,
    out: Any = None,
    overwrite_input: bool = False,
    keepdims: KeepDims = False,
) -> Array[reduce_shape(Shape, Axis, KeepDims)]: ...
@overload
def nanmedian[Shape: _Shape](
    a: Array[Shape],
    axis: Sequence[int],
    out: Any = None,
    overwrite_input: bool = False,
    keepdims: bool = False,
) -> Array[IntTuple]: ...

# Average
@overload
def average[Shape: _Shape, Axis: Flag[_Axis], KeepDims: Flag[bool]](
    a: Array[Shape],
    axis: Axis = None,
    weights: Any = None,
    returned: Literal[False] = False,
    keepdims: KeepDims = False,
) -> Array[reduce_shape(Shape, Axis, KeepDims)]: ...
@overload
def average[Shape: _Shape, Axis: Flag[_Axis], KeepDims: Flag[bool]](
    a: Array[Shape],
    axis: Axis = None,
    weights: Any = None,
    returned: Literal[True] = ...,
    keepdims: KeepDims = False,
) -> tuple[
    Array[reduce_shape(Shape, Axis, KeepDims)],
    Array[reduce_shape(Shape, Axis, KeepDims)],
]: ...
@overload
def average[Shape: _Shape](
    a: Array[Shape],
    axis: Sequence[int],
    weights: Any = None,
    returned: bool = False,
    keepdims: bool = False,
) -> Array[IntTuple] | tuple[Array[IntTuple], Array[IntTuple]]: ...

# Arg reductions
@overload
def argmax[Shape: _Shape, Axis: Flag[_Axis], KeepDims: Flag[bool]](
    a: Array[Shape],
    axis: Axis = None,
    out: Any = None,
    keepdims: KeepDims = False,
) -> Array[reduce_shape(Shape, Axis, KeepDims)]: ...
@overload
def argmax(
    a: Array[Any],
    axis: int | None = None,
    out: Any = None,
    keepdims: bool | None = None,
) -> Array[IntTuple]: ...
@overload
def argmin[Shape: _Shape, Axis: Flag[_Axis], KeepDims: Flag[bool]](
    a: Array[Shape],
    axis: Axis = None,
    out: Any = None,
    keepdims: KeepDims = False,
) -> Array[reduce_shape(Shape, Axis, KeepDims)]: ...
@overload
def argmin(
    a: Array[Any],
    axis: int | None = None,
    out: Any = None,
    keepdims: bool | None = None,
) -> Array[IntTuple]: ...
@overload
def nanargmax[Shape: _Shape, Axis: Flag[_Axis], KeepDims: Flag[bool]](
    a: Array[Shape],
    axis: Axis = None,
    out: Any = None,
    keepdims: KeepDims = False,
) -> Array[reduce_shape(Shape, Axis, KeepDims)]: ...
@overload
def nanargmax(
    a: Array[Any],
    axis: int | None = None,
    out: Any = None,
    keepdims: bool | None = None,
) -> Array[IntTuple]: ...
@overload
def nanargmin[Shape: _Shape, Axis: Flag[_Axis], KeepDims: Flag[bool]](
    a: Array[Shape],
    axis: Axis = None,
    out: Any = None,
    keepdims: KeepDims = False,
) -> Array[reduce_shape(Shape, Axis, KeepDims)]: ...
@overload
def nanargmin(
    a: Array[Any],
    axis: int | None = None,
    out: Any = None,
    keepdims: bool | None = None,
) -> Array[IntTuple]: ...

# Cumulative operations
@overload
def cumsum[Shape: _Shape](
    a: Array[Shape],
    axis: int,
    dtype: DTypeLike | None = None,
    out: Any = None,
) -> Array[Shape]: ...
@overload
def cumsum(
    a: Array[Any],
    axis: None = None,
    dtype: DTypeLike | None = None,
    out: Any = None,
) -> Array[IntTuple]: ...
@overload
def cumprod[Shape: _Shape](
    a: Array[Shape],
    axis: int,
    dtype: DTypeLike | None = None,
    out: Any = None,
) -> Array[Shape]: ...
@overload
def cumprod(
    a: Array[Any],
    axis: None = None,
    dtype: DTypeLike | None = None,
    out: Any = None,
) -> Array[IntTuple]: ...
@overload
def cumulative_sum[Shape: _Shape](
    x: Array[Shape],
    /,
    *,
    axis: int,
    dtype: DTypeLike | None = None,
    include_initial: Literal[False] = False,
) -> Array[Shape]: ...
@overload
def cumulative_sum(
    x: Array[Any],
    /,
    *,
    axis: int | None = None,
    dtype: DTypeLike | None = None,
    include_initial: bool = False,
) -> Array[IntTuple]: ...
@overload
def cumulative_prod[Shape: _Shape](
    x: Array[Shape],
    /,
    *,
    axis: int,
    dtype: DTypeLike | None = None,
    include_initial: Literal[False] = False,
) -> Array[Shape]: ...
@overload
def cumulative_prod(
    x: Array[Any],
    /,
    *,
    axis: int | None = None,
    dtype: DTypeLike | None = None,
    include_initial: bool = False,
) -> Array[IntTuple]: ...
@overload
def nancumsum[Shape: _Shape](
    a: Array[Shape],
    axis: int,
    dtype: DTypeLike | None = None,
    out: Any = None,
) -> Array[Shape]: ...
@overload
def nancumsum(
    a: Array[Any],
    axis: None = None,
    dtype: DTypeLike | None = None,
    out: Any = None,
) -> Array[IntTuple]: ...
@overload
def nancumprod[Shape: _Shape](
    a: Array[Shape],
    axis: int,
    dtype: DTypeLike | None = None,
    out: Any = None,
) -> Array[Shape]: ...
@overload
def nancumprod(
    a: Array[Any],
    axis: None = None,
    dtype: DTypeLike | None = None,
    out: Any = None,
) -> Array[IntTuple]: ...

# Quantile & Percentile
@overload
def quantile[Shape: _Shape, Axis: Flag[_Axis], KeepDims: Flag[bool]](
    a: Array[Shape],
    q: int | float,
    axis: Axis = None,
    out: Any = None,
    overwrite_input: bool = False,
    method: str = "linear",
    keepdims: KeepDims = False,
    *,
    weights: Any = None,
) -> Array[reduce_shape(Shape, Axis, KeepDims)]: ...
@overload
def quantile(
    a: Array[Any],
    q: Any,
    axis: Any = None,
    out: Any = None,
    overwrite_input: bool = False,
    method: str = "linear",
    keepdims: bool = False,
    *,
    weights: Any = None,
) -> Array[IntTuple]: ...
@overload
def percentile[Shape: _Shape, Axis: Flag[_Axis], KeepDims: Flag[bool]](
    a: Array[Shape],
    q: int | float,
    axis: Axis = None,
    out: Any = None,
    overwrite_input: bool = False,
    method: str = "linear",
    keepdims: KeepDims = False,
    *,
    weights: Any = None,
    out_sharding: Any = None,
) -> Array[reduce_shape(Shape, Axis, KeepDims)]: ...
@overload
def percentile(
    a: Array[Any],
    q: Any,
    axis: Any = None,
    out: Any = None,
    overwrite_input: bool = False,
    method: str = "linear",
    keepdims: bool = False,
    *,
    weights: Any = None,
    out_sharding: Any = None,
) -> Array[IntTuple]: ...
@overload
def nanquantile[Shape: _Shape, Axis: Flag[_Axis], KeepDims: Flag[bool]](
    a: Array[Shape],
    q: int | float,
    axis: Axis = None,
    out: Any = None,
    overwrite_input: bool = False,
    method: str = "linear",
    keepdims: KeepDims = False,
    *,
    weights: Any = None,
) -> Array[reduce_shape(Shape, Axis, KeepDims)]: ...
@overload
def nanquantile(
    a: Array[Any],
    q: Any,
    axis: Any = None,
    out: Any = None,
    overwrite_input: bool = False,
    method: str = "linear",
    keepdims: bool = False,
    *,
    weights: Any = None,
) -> Array[IntTuple]: ...
@overload
def nanpercentile[Shape: _Shape, Axis: Flag[_Axis], KeepDims: Flag[bool]](
    a: Array[Shape],
    q: int | float,
    axis: Axis = None,
    out: Any = None,
    overwrite_input: bool = False,
    method: str = "linear",
    keepdims: KeepDims = False,
    *,
    weights: Any = None,
) -> Array[reduce_shape(Shape, Axis, KeepDims)]: ...
@overload
def nanpercentile(
    a: Array[Any],
    q: Any,
    axis: Any = None,
    out: Any = None,
    overwrite_input: bool = False,
    method: str = "linear",
    keepdims: bool = False,
    *,
    weights: Any = None,
) -> Array[IntTuple]: ...

# Differences & Calculus
def diff(
    a: Array[Any],
    n: int = 1,
    axis: int = -1,
    prepend: Any = None,
    append: Any = None,
) -> Array[IntTuple]: ...
def ediff1d(
    ary: Array[Any],
    to_end: Any = None,
    to_begin: Any = None,
) -> Array[IntTuple]: ...
@overload
def gradient(
    f: Array[Any],
    *varargs: Any,
    axis: int,
    edge_order: int | None = None,
) -> Array[IntTuple]: ...
@overload
def gradient(
    f: Array[Any],
    *varargs: Any,
    axis: Sequence[int] | None = None,
    edge_order: int | None = None,
) -> list[Array[IntTuple]]: ...
@overload
def trapezoid[Shape: _Shape, Axis: Flag[_Axis]](
    y: Array[Shape],
    x: Any = None,
    dx: Any = 1.0,
    axis: Axis = -1,
) -> Array[reduce_shape(Shape, Axis, False)]: ...
@overload
def trapezoid(
    y: Array[Any],
    x: Any = None,
    dx: Any = 1.0,
    axis: int = -1,
) -> Array[IntTuple]: ...
def corrcoef(
    x: Array[Any],
    y: Any = None,
    rowvar: bool = True,
    dtype: DTypeLike | None = None,
) -> Array[IntTuple]: ...
def cov(
    m: Array[Any],
    y: Any = None,
    rowvar: bool = True,
    bias: bool = False,
    ddof: int | None = None,
    fweights: Any = None,
    aweights: Any = None,
    dtype: DTypeLike | None = None,
) -> Array[IntTuple]: ...

# Logic and Comparison
def allclose(
    a: Array | _Scalar,
    b: Array | _Scalar,
    rtol: Any = 1e-05,
    atol: Any = 1e-08,
    equal_nan: bool = False,
) -> Array[[]]: ...
def array_equal(
    a1: Array | _Scalar, a2: Array | _Scalar, equal_nan: bool = False
) -> Array[[]]: ...
def array_equiv(a1: Array | _Scalar, a2: Array | _Scalar) -> Array[[]]: ...
@overload
def equal[Shape: _Shape](x1: Array[Shape], x2: _Scalar, /) -> Array[Shape]: ...
@overload
def equal[Shape: _Shape](x1: _Scalar, x2: Array[Shape], /) -> Array[Shape]: ...
@overload
def equal[Shape1: _Shape, Shape2: _Shape](
    x1: Array[Shape1], x2: Array[Shape2], /
) -> Array[broadcast(Shape1, Shape2)]: ...
@overload
def equal(x1: _Scalar, x2: _Scalar, /) -> Array[[]]: ...
@overload
def equal(x1: Any, x2: Any, /) -> Array[IntTuple]: ...
@overload
def greater[Shape: _Shape](x1: Array[Shape], x2: _Scalar, /) -> Array[Shape]: ...
@overload
def greater[Shape: _Shape](x1: _Scalar, x2: Array[Shape], /) -> Array[Shape]: ...
@overload
def greater[Shape1: _Shape, Shape2: _Shape](
    x1: Array[Shape1], x2: Array[Shape2], /
) -> Array[broadcast(Shape1, Shape2)]: ...
@overload
def greater(x1: _Scalar, x2: _Scalar, /) -> Array[[]]: ...
@overload
def greater(x1: Any, x2: Any, /) -> Array[IntTuple]: ...
@overload
def greater_equal[Shape: _Shape](x1: Array[Shape], x2: _Scalar, /) -> Array[Shape]: ...
@overload
def greater_equal[Shape: _Shape](x1: _Scalar, x2: Array[Shape], /) -> Array[Shape]: ...
@overload
def greater_equal[Shape1: _Shape, Shape2: _Shape](
    x1: Array[Shape1], x2: Array[Shape2], /
) -> Array[broadcast(Shape1, Shape2)]: ...
@overload
def greater_equal(x1: _Scalar, x2: _Scalar, /) -> Array[[]]: ...
@overload
def greater_equal(x1: Any, x2: Any, /) -> Array[IntTuple]: ...
@overload
def isclose[Shape: _Shape](
    a: Array[Shape],
    b: _Scalar,
    rtol: Any = 1e-05,
    atol: Any = 1e-08,
    equal_nan: bool = False,
) -> Array[Shape]: ...
@overload
def isclose[Shape: _Shape](
    a: _Scalar,
    b: Array[Shape],
    rtol: Any = 1e-05,
    atol: Any = 1e-08,
    equal_nan: bool = False,
) -> Array[Shape]: ...
@overload
def isclose[Shape1: _Shape, Shape2: _Shape](
    a: Array[Shape1],
    b: Array[Shape2],
    rtol: Any = 1e-05,
    atol: Any = 1e-08,
    equal_nan: bool = False,
) -> Array[broadcast(Shape1, Shape2)]: ...
@overload
def isclose(
    a: _Scalar,
    b: _Scalar,
    rtol: Any = 1e-05,
    atol: Any = 1e-08,
    equal_nan: bool = False,
) -> Array[[]]: ...
@overload
def isclose(
    a: Any,
    b: Any,
    rtol: Any = 1e-05,
    atol: Any = 1e-08,
    equal_nan: bool = False,
) -> Array[IntTuple]: ...
@overload
def iscomplex[Shape: _Shape](x: Array[Shape], /) -> Array[Shape]: ...
@overload
def iscomplex(x: _Scalar, /) -> Array[[]]: ...
@overload
def iscomplex(x: Any, /) -> Array[IntTuple]: ...
def iscomplexobj(x: Any) -> bool: ...
@overload
def isfinite[Shape: _Shape](x: Array[Shape], /) -> Array[Shape]: ...
@overload
def isfinite(x: _Scalar, /) -> Array[[]]: ...
@overload
def isfinite(x: Any, /) -> Array[IntTuple]: ...
@overload
def isinf[Shape: _Shape](x: Array[Shape], /) -> Array[Shape]: ...
@overload
def isinf(x: _Scalar, /) -> Array[[]]: ...
@overload
def isinf(x: Any, /) -> Array[IntTuple]: ...
@overload
def isnan[Shape: _Shape](x: Array[Shape], /) -> Array[Shape]: ...
@overload
def isnan(x: _Scalar, /) -> Array[[]]: ...
@overload
def isnan(x: Any, /) -> Array[IntTuple]: ...
@overload
def isneginf[Shape: _Shape](x: Array[Shape], /, out: Any = None) -> Array[Shape]: ...
@overload
def isneginf(x: _Scalar, /, out: Any = None) -> Array[[]]: ...
@overload
def isneginf(x: Any, /, out: Any = None) -> Array[IntTuple]: ...
@overload
def isposinf[Shape: _Shape](x: Array[Shape], /, out: Any = None) -> Array[Shape]: ...
@overload
def isposinf(x: _Scalar, /, out: Any = None) -> Array[[]]: ...
@overload
def isposinf(x: Any, /, out: Any = None) -> Array[IntTuple]: ...
@overload
def isreal[Shape: _Shape](x: Array[Shape], /) -> Array[Shape]: ...
@overload
def isreal(x: _Scalar, /) -> Array[[]]: ...
@overload
def isreal(x: Any, /) -> Array[IntTuple]: ...
def isrealobj(x: Any) -> bool: ...
def isscalar(element: Any) -> bool: ...
def iterable(y: Any) -> bool: ...
@overload
def less[Shape: _Shape](x1: Array[Shape], x2: _Scalar, /) -> Array[Shape]: ...
@overload
def less[Shape: _Shape](x1: _Scalar, x2: Array[Shape], /) -> Array[Shape]: ...
@overload
def less[Shape1: _Shape, Shape2: _Shape](
    x1: Array[Shape1], x2: Array[Shape2], /
) -> Array[broadcast(Shape1, Shape2)]: ...
@overload
def less(x1: _Scalar, x2: _Scalar, /) -> Array[[]]: ...
@overload
def less(x1: Any, x2: Any, /) -> Array[IntTuple]: ...
@overload
def less_equal[Shape: _Shape](x1: Array[Shape], x2: _Scalar, /) -> Array[Shape]: ...
@overload
def less_equal[Shape: _Shape](x1: _Scalar, x2: Array[Shape], /) -> Array[Shape]: ...
@overload
def less_equal[Shape1: _Shape, Shape2: _Shape](
    x1: Array[Shape1], x2: Array[Shape2], /
) -> Array[broadcast(Shape1, Shape2)]: ...
@overload
def less_equal(x1: _Scalar, x2: _Scalar, /) -> Array[[]]: ...
@overload
def less_equal(x1: Any, x2: Any, /) -> Array[IntTuple]: ...
@overload
def logical_and[Shape: _Shape](x1: Array[Shape], x2: _Scalar, /) -> Array[Shape]: ...
@overload
def logical_and[Shape: _Shape](x1: _Scalar, x2: Array[Shape], /) -> Array[Shape]: ...
@overload
def logical_and[Shape1: _Shape, Shape2: _Shape](
    x1: Array[Shape1], x2: Array[Shape2], /
) -> Array[broadcast(Shape1, Shape2)]: ...
@overload
def logical_and(x1: _Scalar, x2: _Scalar, /) -> Array[[]]: ...
@overload
def logical_and(x1: Any, x2: Any, /) -> Array[IntTuple]: ...
@overload
def logical_not[Shape: _Shape](x: Array[Shape], /) -> Array[Shape]: ...
@overload
def logical_not(x: _Scalar, /) -> Array[[]]: ...
@overload
def logical_not(x: Any, /) -> Array[IntTuple]: ...
@overload
def logical_or[Shape: _Shape](x1: Array[Shape], x2: _Scalar, /) -> Array[Shape]: ...
@overload
def logical_or[Shape: _Shape](x1: _Scalar, x2: Array[Shape], /) -> Array[Shape]: ...
@overload
def logical_or[Shape1: _Shape, Shape2: _Shape](
    x1: Array[Shape1], x2: Array[Shape2], /
) -> Array[broadcast(Shape1, Shape2)]: ...
@overload
def logical_or(x1: _Scalar, x2: _Scalar, /) -> Array[[]]: ...
@overload
def logical_or(x1: Any, x2: Any, /) -> Array[IntTuple]: ...
@overload
def logical_xor[Shape: _Shape](x1: Array[Shape], x2: _Scalar, /) -> Array[Shape]: ...
@overload
def logical_xor[Shape: _Shape](x1: _Scalar, x2: Array[Shape], /) -> Array[Shape]: ...
@overload
def logical_xor[Shape1: _Shape, Shape2: _Shape](
    x1: Array[Shape1], x2: Array[Shape2], /
) -> Array[broadcast(Shape1, Shape2)]: ...
@overload
def logical_xor(x1: _Scalar, x2: _Scalar, /) -> Array[[]]: ...
@overload
def logical_xor(x1: Any, x2: Any, /) -> Array[IntTuple]: ...
@overload
def not_equal[Shape: _Shape](x1: Array[Shape], x2: _Scalar, /) -> Array[Shape]: ...
@overload
def not_equal[Shape: _Shape](x1: _Scalar, x2: Array[Shape], /) -> Array[Shape]: ...
@overload
def not_equal[Shape1: _Shape, Shape2: _Shape](
    x1: Array[Shape1], x2: Array[Shape2], /
) -> Array[broadcast(Shape1, Shape2)]: ...
@overload
def not_equal(x1: _Scalar, x2: _Scalar, /) -> Array[[]]: ...
@overload
def not_equal(x1: Any, x2: Any, /) -> Array[IntTuple]: ...

# Sorting and Partitioning
@overload
def sort[Shape: _Shape, Axis: Flag[int | None] = -1](
    a: Array[Shape],
    axis: Axis = -1,
    *,
    kind: None = None,
    order: None = None,
    stable: bool = True,
    descending: bool = False,
) -> Array[sort_shape(Shape, Axis)]: ...
@overload
def sort(
    a: Any,
    axis: int | None = -1,
    *,
    kind: None = None,
    order: None = None,
    stable: bool = True,
    descending: bool = False,
) -> Array[IntTuple]: ...
@overload
def argsort[Shape: _Shape, Axis: Flag[int | None] = -1](
    a: Array[Shape],
    axis: Axis = -1,
    *,
    kind: None = None,
    order: None = None,
    stable: bool = True,
    descending: bool = False,
    dtype: DTypeLike | None = None,
) -> Array[sort_shape(Shape, Axis)]: ...
@overload
def argsort(
    a: Any,
    axis: int | None = -1,
    *,
    kind: None = None,
    order: None = None,
    stable: bool = True,
    descending: bool = False,
    dtype: DTypeLike | None = None,
) -> Array[IntTuple]: ...
@overload
def sort_complex[Shape: _Shape](a: Array[Shape]) -> Array[Shape]: ...
@overload
def sort_complex(a: Any) -> Array[IntTuple]: ...
@overload
def partition[Shape: _Shape, Axis: Flag[int] = -1](
    a: Array[Shape],
    kth: int | Sequence[int],
    axis: Axis = -1,
) -> Array[sort_shape(Shape, Axis)]: ...
@overload
def partition(
    a: Any,
    kth: int | Sequence[int],
    axis: int = -1,
) -> Array[IntTuple]: ...
@overload
def argpartition[Shape: _Shape, Axis: Flag[int] = -1](
    a: Array[Shape],
    kth: int | Sequence[int],
    axis: Axis = -1,
) -> Array[sort_shape(Shape, Axis)]: ...
@overload
def argpartition(
    a: Any,
    kth: int | Sequence[int],
    axis: int = -1,
) -> Array[IntTuple]: ...
@overload
def lexsort[Shape: _Shape](
    keys: Sequence[Array[Shape]],
    axis: int = -1,
) -> Array[Shape]: ...
@overload
def lexsort(
    keys: Any,
    axis: int = -1,
) -> Array[IntTuple]: ...
@overload
def top_k[Shape: _Shape, K: Flag[int], Axis: Flag[int] = -1](
    a: Array[Shape],
    k: K,
    /,
    *,
    axis: Axis = -1,
    mode: str = "largest",
    sorted: bool = True,
) -> tuple[
    Array[top_k_shape(Shape, K, Axis)],
    Array[top_k_shape(Shape, K, Axis)],
]: ...
@overload
def top_k(
    a: Any,
    k: int,
    /,
    *,
    axis: int = -1,
    mode: str = "largest",
    sorted: bool = True,
) -> tuple[Array[IntTuple], Array[IntTuple]]: ...

# Searching
@overload
def searchsorted(
    a: Any,
    v: _Scalar,
    side: str = "left",
    sorter: Any = None,
    *,
    method: str = "scan",
) -> Array[[]]: ...
@overload
def searchsorted[Shape: _Shape](
    a: Any,
    v: Array[Shape],
    side: str = "left",
    sorter: Any = None,
    *,
    method: str = "scan",
) -> Array[Shape]: ...
@overload
def searchsorted(
    a: Any,
    v: Any,
    side: str = "left",
    sorter: Any = None,
    *,
    method: str = "scan",
) -> Array[IntTuple]: ...
@overload
def nonzero[Size: IntVar](
    a: Any,
    *,
    size: Int[Size],
    fill_value: Any = None,
) -> tuple[Array[[Size]], ...]: ...
@overload
def nonzero(
    a: Any,
    *,
    size: int | None = None,
    fill_value: Any = None,
) -> tuple[Array[IntTuple], ...]: ...
@overload
def flatnonzero[Size: IntVar](
    a: Any,
    *,
    size: Int[Size],
    fill_value: Any = None,
) -> Array[[Size]]: ...
@overload
def flatnonzero(
    a: Any,
    *,
    size: int | None = None,
    fill_value: Any = None,
) -> Array[IntTuple]: ...
@overload
def argwhere[N: IntVar, Size: IntVar](
    a: Array[[N]],
    *,
    size: Int[Size],
    fill_value: Any = None,
) -> Array[[Size, 1]]: ...
@overload
def argwhere[M: IntVar, N: IntVar, Size: IntVar](
    a: Array[[M, N]],
    *,
    size: Int[Size],
    fill_value: Any = None,
) -> Array[[Size, 2]]: ...
@overload
def argwhere[L: IntVar, M: IntVar, N: IntVar, Size: IntVar](
    a: Array[[L, M, N]],
    *,
    size: Int[Size],
    fill_value: Any = None,
) -> Array[[Size, 3]]: ...
@overload
def argwhere[K: IntVar, L: IntVar, M: IntVar, N: IntVar, Size: IntVar](
    a: Array[[K, L, M, N]],
    *,
    size: Int[Size],
    fill_value: Any = None,
) -> Array[[Size, 4]]: ...
@overload
def argwhere[Size: IntVar](
    a: Any,
    *,
    size: Int[Size],
    fill_value: Any = None,
) -> Array[[Size, int]]: ...
@overload
def argwhere(
    a: Any,
    *,
    size: int | None = None,
    fill_value: Any = None,
) -> Array[IntTuple]: ...
@overload
def nan_to_num[Shape: _Shape](
    x: Array[Shape],
    copy: bool = True,
    nan: _Scalar = 0.0,
    posinf: _Scalar | None = None,
    neginf: _Scalar | None = None,
) -> Array[Shape]: ...
@overload
def nan_to_num(
    x: Any,
    copy: bool = True,
    nan: _Scalar = 0.0,
    posinf: _Scalar | None = None,
    neginf: _Scalar | None = None,
) -> Array[IntTuple]: ...
@overload
def digitize[Shape: _Shape](
    x: Array[Shape],
    bins: Any,
    right: bool = False,
    *,
    method: str | None = None,
) -> Array[Shape]: ...
@overload
def digitize(
    x: Any,
    bins: Any,
    right: bool = False,
    *,
    method: str | None = None,
) -> Array[IntTuple]: ...
@overload
def where[Size: IntVar](
    condition: Any,
    *,
    size: Int[Size],
    fill_value: Any = None,
) -> tuple[Array[[Size]], ...]: ...
@overload
def where(
    condition: Any,
    *,
    size: int | None = None,
    fill_value: Any = None,
) -> tuple[Array[IntTuple], ...]: ...
@overload
def where[Shape: _Shape](
    condition: Array[Shape],
    x: _Scalar,
    y: _Scalar,
    /,
) -> Array[Shape]: ...
@overload
def where[CondShape: _Shape, XShape: _Shape](
    condition: Array[CondShape],
    x: Array[XShape],
    y: _Scalar,
    /,
) -> Array[broadcast(CondShape, XShape)]: ...
@overload
def where[CondShape: _Shape, YShape: _Shape](
    condition: Array[CondShape],
    x: _Scalar,
    y: Array[YShape],
    /,
) -> Array[broadcast(CondShape, YShape)]: ...
@overload
def where[CondShape: _Shape, XShape: _Shape, YShape: _Shape](
    condition: Array[CondShape],
    x: Array[XShape],
    y: Array[YShape],
    /,
) -> Array[broadcast(broadcast(CondShape, XShape), YShape)]: ...
@overload
def where(
    condition: Any,
    x: Any = None,
    y: Any = None,
    /,
    *,
    size: int | None = None,
    fill_value: Any = None,
) -> Any: ...

# Selection & Clipping
@overload
def bincount[Length: IntVar](
    x: Any,
    weights: Any = None,
    minlength: int = 0,
    *,
    length: Int[Length],
    out_sharding: Any = None,
) -> Array[[Length]]: ...
@overload
def bincount(
    x: Any,
    weights: Any = None,
    minlength: int = 0,
    *,
    length: int | None = None,
    out_sharding: Any = None,
) -> Array[IntTuple]: ...
@overload
def choose[Shape: _Shape](
    a: Array[Shape],
    choices: Sequence[Array[Shape] | _Scalar],
    out: Any = None,
    mode: str = "raise",
) -> Array[Shape]: ...
@overload
def choose(
    a: Any,
    choices: Any,
    out: Any = None,
    mode: str = "raise",
) -> Array[IntTuple]: ...
@overload
def clip[Shape: _Shape](
    a: Array[Shape],
    min: _Scalar | None = None,
    max: _Scalar | None = None,
) -> Array[Shape]: ...
@overload
def clip[Shape1: _Shape, Shape2: _Shape](
    a: Array[Shape1],
    min: Array[Shape2],
    max: _Scalar | None = None,
) -> Array[broadcast(Shape1, Shape2)]: ...
@overload
def clip[Shape1: _Shape, Shape3: _Shape](
    a: Array[Shape1],
    min: _Scalar | None,
    max: Array[Shape3],
) -> Array[broadcast(Shape1, Shape3)]: ...
@overload
def clip[Shape1: _Shape, Shape2: _Shape, Shape3: _Shape](
    a: Array[Shape1],
    min: Array[Shape2],
    max: Array[Shape3],
) -> Array[broadcast(broadcast(Shape1, Shape2), Shape3)]: ...
@overload
def clip(
    a: Any,
    min: Any = None,
    max: Any = None,
) -> Array[IntTuple]: ...
@overload
def fmax[Shape: _Shape](x1: Array[Shape], x2: _Scalar, /) -> Array[Shape]: ...
@overload
def fmax[Shape: _Shape](x1: _Scalar, x2: Array[Shape], /) -> Array[Shape]: ...
@overload
def fmax[Shape1: _Shape, Shape2: _Shape](
    x1: Array[Shape1], x2: Array[Shape2], /
) -> Array[broadcast(Shape1, Shape2)]: ...
@overload
def fmax(x1: Any, x2: Any, /) -> Array[IntTuple]: ...
@overload
def fmin[Shape: _Shape](x1: Array[Shape], x2: _Scalar, /) -> Array[Shape]: ...
@overload
def fmin[Shape: _Shape](x1: _Scalar, x2: Array[Shape], /) -> Array[Shape]: ...
@overload
def fmin[Shape1: _Shape, Shape2: _Shape](
    x1: Array[Shape1], x2: Array[Shape2], /
) -> Array[broadcast(Shape1, Shape2)]: ...
@overload
def fmin(x1: Any, x2: Any, /) -> Array[IntTuple]: ...
@overload
def piecewise[Shape: _Shape](
    x: Array[Shape],
    condlist: Array[Any] | Sequence[Array[Any] | bool],
    funclist: Sequence[Any],
    *args: Any,
    **kw: Any,
) -> Array[Shape]: ...
@overload
def piecewise(
    x: Any,
    condlist: Any,
    funclist: Any,
    *args: Any,
    **kw: Any,
) -> Array[IntTuple]: ...
@overload
def select[Shape: _Shape](
    condlist: Sequence[Any],
    choicelist: Sequence[Array[Shape]],
    default: _Scalar | Array[Shape] = 0,
) -> Array[Shape]: ...
@overload
def select(
    condlist: Any,
    choicelist: Any,
    default: Any = 0,
) -> Array[IntTuple]: ...

# Set-like operations
@overload
def intersect1d[Size: IntVar](
    ar1: Any,
    ar2: Any,
    assume_unique: bool = False,
    return_indices: Literal[False] = False,
    *,
    size: Int[Size],
    fill_value: Any = None,
) -> Array[[Size]]: ...
@overload
def intersect1d[Size: IntVar](
    ar1: Any,
    ar2: Any,
    assume_unique: bool = False,
    return_indices: Literal[True] = ...,
    *,
    size: Int[Size],
    fill_value: Any = None,
) -> tuple[Array[[Size]], Array[[Size]], Array[[Size]]]: ...
@overload
def intersect1d(
    ar1: Any,
    ar2: Any,
    assume_unique: bool = False,
    return_indices: Literal[True] = ...,
    *,
    size: int | None = None,
    fill_value: Any = None,
) -> tuple[Array[IntTuple], Array[IntTuple], Array[IntTuple]]: ...
@overload
def intersect1d(
    ar1: Any,
    ar2: Any,
    assume_unique: bool = False,
    return_indices: bool = False,
    *,
    size: int | None = None,
    fill_value: Any = None,
) -> Array[IntTuple] | tuple[Array[IntTuple], Array[IntTuple], Array[IntTuple]]: ...
@overload
def isin[Shape: _Shape](
    element: Array[Shape],
    test_elements: Any,
    assume_unique: bool = False,
    invert: bool = False,
    *,
    method: str = "auto",
) -> Array[Shape]: ...
@overload
def isin(
    element: Any,
    test_elements: Any,
    assume_unique: bool = False,
    invert: bool = False,
    *,
    method: str = "auto",
) -> Array[IntTuple]: ...
@overload
def setdiff1d[Size: IntVar](
    ar1: Any,
    ar2: Any,
    assume_unique: bool = False,
    *,
    size: Int[Size],
    fill_value: Any = None,
) -> Array[[Size]]: ...
@overload
def setdiff1d(
    ar1: Any,
    ar2: Any,
    assume_unique: bool = False,
    *,
    size: int | None = None,
    fill_value: Any = None,
) -> Array[IntTuple]: ...
@overload
def setxor1d[Size: IntVar](
    ar1: Any,
    ar2: Any,
    assume_unique: bool = False,
    *,
    size: Int[Size],
    fill_value: Any = None,
) -> Array[[Size]]: ...
@overload
def setxor1d(
    ar1: Any,
    ar2: Any,
    assume_unique: bool = False,
    *,
    size: int | None = None,
    fill_value: Any = None,
) -> Array[IntTuple]: ...
@overload
def union1d[Size: IntVar](
    ar1: Any,
    ar2: Any,
    *,
    size: Int[Size],
    fill_value: Any = None,
) -> Array[[Size]]: ...
@overload
def union1d(
    ar1: Any,
    ar2: Any,
    *,
    size: int | None = None,
    fill_value: Any = None,
) -> Array[IntTuple]: ...
@overload
def unique[Size: IntVar](
    ar: Any,
    return_index: Literal[False] = False,
    return_inverse: Literal[False] = False,
    return_counts: Literal[False] = False,
    axis: int | None = None,
    *,
    equal_nan: bool = True,
    size: Int[Size],
    fill_value: Any = None,
    sorted: bool = True,
) -> Array[[Size]]: ...
@overload
def unique(
    ar: Any,
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
    values: Array[IntTuple]
    indices: Array[IntTuple]
    inverse_indices: Array[IntTuple]
    counts: Array[IntTuple]

class _UniqueCountsResult(NamedTuple):
    values: Array[IntTuple]
    counts: Array[IntTuple]

class _UniqueInverseResult(NamedTuple):
    values: Array[IntTuple]
    inverse_indices: Array[IntTuple]

def unique_all(
    x: Any,
    /,
    *,
    size: int | None = None,
    fill_value: Any = None,
) -> _UniqueAllResult: ...
def unique_counts(
    x: Any,
    /,
    *,
    size: int | None = None,
    fill_value: Any = None,
) -> _UniqueCountsResult: ...
def unique_inverse(
    x: Any,
    /,
    *,
    size: int | None = None,
    fill_value: Any = None,
) -> _UniqueInverseResult: ...
@overload
def unique_values[Size: IntVar](
    x: Any,
    /,
    *,
    size: Int[Size],
    fill_value: Any = None,
) -> Array[[Size]]: ...
@overload
def unique_values(
    x: Any,
    /,
    *,
    size: int | None = None,
    fill_value: Any = None,
) -> Array[IntTuple]: ...

# Indexing, Slicing & Masking
@overload
def compress[Shape: _Shape, Size: Flag[int], Axis: Flag[int | None] = None](
    condition: Any,
    a: Array[Shape],
    axis: Axis = None,
    *,
    size: Size,
    fill_value: Any = 0,
    out: None = None,
) -> Array[compress_shape(Shape, Size, Axis)]: ...
@overload
def compress[Shape: _Shape, Axis: Flag[int | None] = None](
    condition: Any,
    a: Array[Shape],
    axis: Axis = None,
    *,
    size: int | None = None,
    fill_value: Any = 0,
    out: None = None,
) -> Array[IntTuple]: ...
@overload
def compress(
    condition: Any,
    a: Any,
    axis: int | None = None,
    *,
    size: int | None = None,
    fill_value: Any = 0,
    out: None = None,
) -> Array[IntTuple]: ...
def delete(
    arr: Any,
    obj: Any,
    axis: int | None = None,
    *,
    assume_unique_indices: bool = False,
) -> Array[IntTuple]: ...
@overload
def extract[Size: IntVar](
    condition: Any,
    arr: Any,
    *,
    size: Int[Size],
    fill_value: Any = 0,
) -> Array[[Size]]: ...
@overload
def extract(
    condition: Any,
    arr: Any,
    *,
    size: int | None = None,
    fill_value: Any = 0,
) -> Array[IntTuple]: ...
@overload
def fill_diagonal[Shape: _Shape](
    a: Array[Shape],
    val: Any,
    wrap: bool = False,
    *,
    inplace: bool = True,
) -> Array[fill_diagonal_shape(Shape)]: ...
@overload
def fill_diagonal(
    a: Any,
    val: Any,
    wrap: bool = False,
    *,
    inplace: bool = True,
) -> Array[IntTuple]: ...
def insert(
    arr: Any,
    obj: Any,
    values: Any,
    axis: int | None = None,
) -> Array[IntTuple]: ...
@overload
def place[Shape: _Shape](
    arr: Array[Shape],
    mask: Any,
    vals: Any,
    *,
    inplace: bool = True,
) -> Array[Shape]: ...
@overload
def place(
    arr: Any,
    mask: Any,
    vals: Any,
    *,
    inplace: bool = True,
) -> Array[IntTuple]: ...
@overload
def put[Shape: _Shape](
    a: Array[Shape],
    ind: Any,
    v: Any,
    mode: str | None = None,
    *,
    inplace: bool = True,
) -> Array[Shape]: ...
@overload
def put(
    a: Any,
    ind: Any,
    v: Any,
    mode: str | None = None,
    *,
    inplace: bool = True,
) -> Array[IntTuple]: ...
@overload
def put_along_axis[Shape: _Shape](
    arr: Array[Shape],
    indices: Any,
    values: Any,
    axis: int | None,
    inplace: bool = True,
    *,
    mode: str | None = None,
) -> Array[Shape]: ...
@overload
def put_along_axis(
    arr: Any,
    indices: Any,
    values: Any,
    axis: int | None,
    inplace: bool = True,
    *,
    mode: str | None = None,
) -> Array[IntTuple]: ...
@overload
def take[Shape: _Shape, IdxShape: _Shape, Axis: Flag[int | None] = None](
    a: Array[Shape],
    indices: Array[IdxShape],
    axis: Axis = None,
    out: None = None,
    mode: str | None = None,
    unique_indices: bool = False,
    indices_are_sorted: bool = False,
    fill_value: Any = None,
) -> Array[take_shape(Shape, IdxShape, Axis)]: ...
@overload
def take[Shape: _Shape, Axis: Flag[int | None] = None](
    a: Array[Shape],
    indices: int,
    axis: Axis = None,
    out: None = None,
    mode: str | None = None,
    unique_indices: bool = False,
    indices_are_sorted: bool = False,
    fill_value: Any = None,
) -> Array[take_scalar_idx_shape(Shape, Axis)]: ...
@overload
def take(
    a: Any,
    indices: Any,
    axis: int | None = None,
    out: None = None,
    mode: str | None = None,
    unique_indices: bool = False,
    indices_are_sorted: bool = False,
    fill_value: Any = None,
) -> Array[IntTuple]: ...
@overload
def take_along_axis[ArrShape: _Shape, IdxShape: _Shape, Axis: Flag[int | None] = -1](
    arr: Array[ArrShape],
    indices: Array[IdxShape],
    axis: Axis = -1,
    mode: str | None = None,
    fill_value: Any = None,
    *,
    wrap_negative_indices: bool = True,
) -> Array[take_along_axis_shape(ArrShape, IdxShape, Axis)]: ...
@overload
def take_along_axis(
    arr: Any,
    indices: Any,
    axis: int | None = -1,
    mode: str | None = None,
    fill_value: Any = None,
    *,
    wrap_negative_indices: bool = True,
) -> Array[IntTuple]: ...
def trim_zeros(
    filt: Any,
    trim: str = "fb",
    axis: int | Sequence[int] | None = None,
) -> Array[IntTuple]: ...
@overload
def diag_indices[N: IntVar](
    n: Int[N],
    ndim: int = 2,
) -> tuple[Array[[N]], ...]: ...
@overload
def diag_indices(
    n: int,
    ndim: int = 2,
) -> tuple[Array[IntTuple], ...]: ...
@overload
def diag_indices_from[Shape: _Shape](
    arr: Array[Shape],
) -> tuple[Array[diag_indices_from_shape(Shape)], ...]: ...
@overload
def diag_indices_from(
    arr: Any,
) -> tuple[Array[IntTuple], ...]: ...
@overload
def mask_indices[Size: IntVar](
    n: int,
    mask_func: Callable[..., Any],
    k: int = 0,
    *,
    size: Int[Size],
) -> tuple[Array[[Size]], Array[[Size]]]: ...
@overload
def mask_indices(
    n: int,
    mask_func: Callable[..., Any],
    k: int = 0,
    *,
    size: int | None = None,
) -> tuple[Array[IntTuple], Array[IntTuple]]: ...
def ravel_multi_index(
    multi_index: Sequence[Any],
    dims: Sequence[int],
    mode: str = "raise",
    order: str = "C",
    *,
    dtype: Any = None,
) -> Array[IntTuple]: ...
def tril_indices(
    n: int,
    k: int = 0,
    m: int | None = None,
) -> tuple[Array[IntTuple], Array[IntTuple]]: ...
def tril_indices_from(
    arr: Any,
    k: int = 0,
) -> tuple[Array[IntTuple], Array[IntTuple]]: ...
def triu_indices(
    n: int,
    k: int = 0,
    m: int | None = None,
) -> tuple[Array[IntTuple], Array[IntTuple]]: ...
def triu_indices_from(
    arr: Any,
    k: int = 0,
) -> tuple[Array[IntTuple], Array[IntTuple]]: ...
@overload
def unravel_index[Shape: _Shape](
    indices: Array[Shape],
    shape: Any,
) -> tuple[Array[Shape], ...]: ...
@overload
def unravel_index(
    indices: int,
    shape: Any,
) -> tuple[Array[[]], ...]: ...
@overload
def ix_[Shapes: IntTuples](
    *args: Unpack[MapIntTuples[lambda S: Array[S], Shapes]],
) -> MapIntTuples[lambda S: Array[S], ix_shapes(Shapes)]: ...
@overload
def ix_(*args: Array[Any]) -> tuple[Array[IntTuple], ...]: ...

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

@overload
def astype[Shape: _Shape](
    x: Array[Shape],
    dtype: DTypeLike | None,
    /,
    *,
    copy: bool = False,
    device: Any = None,
) -> Array[Shape]: ...
@overload
def astype(
    x: _Scalar,
    dtype: DTypeLike | None,
    /,
    *,
    copy: bool = False,
    device: Any = None,
) -> Array[()]: ...
@overload
def astype(
    x: Any,
    dtype: DTypeLike | None,
    /,
    *,
    copy: bool = False,
    device: Any = None,
) -> Array[IntTuple]: ...
def can_cast(from_: Any, to: DTypeLike, casting: str = "safe") -> bool: ...
def isdtype(
    dtype: DTypeLike, kind: str | DTypeLike | tuple[str | DTypeLike, ...]
) -> bool: ...
def issubdtype(arg1: DTypeLike, arg2: DTypeLike) -> bool: ...
def promote_types(a: DTypeLike, b: DTypeLike) -> Any: ...
def result_type(*args: Any) -> Any: ...

class _Mgrid:
    @overload
    def __getitem__(self, key: slice) -> Array[[Any]]: ...
    @overload
    def __getitem__(self, key: tuple[slice, slice]) -> Array[[2, Any, Any]]: ...
    @overload
    def __getitem__(
        self, key: tuple[slice, slice, slice]
    ) -> Array[[3, Any, Any, Any]]: ...
    @overload
    def __getitem__(self, key: tuple[slice, ...]) -> Array[IntTuple]: ...
    @overload
    def __getitem__(self, key: Any) -> Array[IntTuple]: ...

class _Ogrid:
    @overload
    def __getitem__(self, key: slice) -> Array[[Any]]: ...
    @overload
    def __getitem__(self, key: tuple[slice, slice]) -> list[Array[IntTuple]]: ...
    @overload
    def __getitem__(self, key: tuple[slice, slice, slice]) -> list[Array[IntTuple]]: ...
    @overload
    def __getitem__(self, key: tuple[slice, ...]) -> list[Array[IntTuple]]: ...
    @overload
    def __getitem__(self, key: Any) -> Array[IntTuple] | list[Array[IntTuple]]: ...

mgrid: _Mgrid
ogrid: _Ogrid

class _CClass:
    def __getitem__(self, key: Any) -> Array[IntTuple]: ...

class _RClass:
    def __getitem__(self, key: Any) -> Array[IntTuple]: ...

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

def ndim(a: Any) -> int: ...
@overload
def shape[Shape: _Shape](a: Array[Shape]) -> Shape: ...
@overload
def shape(a: Any) -> tuple[int, ...]: ...
def size(a: Any, axis: int | Sequence[int] | None = None) -> int: ...
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
    func1d: Callable[..., Any], axis: int, arr: Any, *args: Any, **kwargs: Any
) -> Array[IntTuple]: ...
def apply_over_axes(
    func: Callable[[Any, int], Any], a: Any, axes: Sequence[int]
) -> Array[IntTuple]: ...
def frompyfunc(
    func: Callable[..., Any], /, nin: int, nout: int, *, identity: Any = None
) -> ufunc: ...
def vectorize(pyfunc: Any, *, excluded: Any = ..., signature: Any = None) -> Any: ...
def load(file: Any, *args: Any, **kwargs: Any) -> Any: ...

# Bit packing

@overload
def packbits[Shape: _Shape, Axis: Flag[int | None] = None](
    a: Array[Shape],
    axis: Axis = None,
    bitorder: str = "big",
) -> Array[packbits_shape(Shape, Axis)]: ...
@overload
def packbits(
    a: Any,
    axis: int | None = None,
    bitorder: str = "big",
) -> Array[IntTuple]: ...
@overload
def unpackbits[
    Shape: _Shape,
    Axis: Flag[int | None] = None,
    Count: Flag[int | None] = None,
](
    a: Array[Shape],
    axis: Axis = None,
    count: Count = None,
    bitorder: str = "big",
) -> Array[unpackbits_shape(Shape, Axis, Count)]: ...
@overload
def unpackbits(
    a: Any,
    axis: int | None = None,
    count: int | None = None,
    bitorder: str = "big",
) -> Array[IntTuple]: ...

# Interpolation

def interp[Shape: _Shape, N: IntVar](
    x: Array[Shape],
    xp: Array[[N]],
    fp: Array[[N]],
    left: Any = None,
    right: Any = None,
    period: Any = None,
) -> Array[Shape]: ...

# Convolutions & Signal Processing

@overload
def convolve[ShapeA: _Shape, ShapeV: _Shape, Mode: Flag[str] = "full"](
    a: Array[ShapeA],
    v: Array[ShapeV],
    mode: Mode = "full",
    *,
    precision: Any = None,
    preferred_element_type: DTypeLike | None = None,
) -> Array[convolve_shape(ShapeA, ShapeV, Mode)]: ...
@overload
def convolve(
    a: Any,
    v: Any,
    mode: str = "full",
    *,
    precision: Any = None,
    preferred_element_type: DTypeLike | None = None,
) -> Array[IntTuple]: ...
@overload
def correlate[ShapeA: _Shape, ShapeV: _Shape, Mode: Flag[str] = "valid"](
    a: Array[ShapeA],
    v: Array[ShapeV],
    mode: Mode = "valid",
    *,
    precision: Any = None,
    preferred_element_type: DTypeLike | None = None,
) -> Array[convolve_shape(ShapeA, ShapeV, Mode)]: ...
@overload
def correlate(
    a: Any,
    v: Any,
    mode: str = "valid",
    *,
    precision: Any = None,
    preferred_element_type: DTypeLike | None = None,
) -> Array[IntTuple]: ...

# Histograms

@overload
def histogram[Bins: Flag[int] = 10](
    a: Array[Any],
    bins: Bins = 10,
    range: Sequence[Any] | None = None,
    weights: Array[Any] | None = None,
    density: bool | None = None,
) -> tuple[Array[histogram_counts_shape(Bins)], Array[histogram_edges_shape(Bins)]]: ...
@overload
def histogram(
    a: Any,
    bins: Any = 10,
    range: Sequence[Any] | None = None,
    weights: Any = None,
    density: bool | None = None,
) -> tuple[Array[IntTuple], Array[IntTuple]]: ...
@overload
def histogram2d[Bins: Flag[int] = 10](
    x: Array[Any],
    y: Array[Any],
    bins: Bins = 10,
    range: Sequence[Any] | None = None,
    weights: Array[Any] | None = None,
    density: bool | None = None,
) -> tuple[
    Array[histogram2d_counts_shape(Bins)],
    Array[histogram_edges_shape(Bins)],
    Array[histogram_edges_shape(Bins)],
]: ...
@overload
def histogram2d(
    x: Any,
    y: Any,
    bins: Any = 10,
    range: Sequence[Any] | None = None,
    weights: Any = None,
    density: bool | None = None,
) -> tuple[Array[IntTuple], Array[IntTuple], Array[IntTuple]]: ...
@overload
def histogram_bin_edges[Bins: Flag[int] = 10](
    a: Array[Any],
    bins: Bins = 10,
    range: Any = None,
    weights: Array[Any] | None = None,
) -> Array[histogram_edges_shape(Bins)]: ...
@overload
def histogram_bin_edges(
    a: Any,
    bins: Any = 10,
    range: Any = None,
    weights: Any = None,
) -> Array[IntTuple]: ...
def histogramdd(
    sample: Array[Any],
    bins: Any = 10,
    range: Sequence[Any] | None = None,
    weights: Array[Any] | None = None,
    density: bool | None = None,
) -> tuple[Array[IntTuple], list[Array[IntTuple]]]: ...

# Polynomials

@overload
def poly[Shape: _Shape](
    seq_of_zeros: Array[Shape],
) -> Array[poly_shape(Shape)]: ...
@overload
def poly(seq_of_zeros: Any) -> Array[IntTuple]: ...
@overload
def polyadd[Shape1: _Shape, Shape2: _Shape](
    a1: Array[Shape1],
    a2: Array[Shape2],
) -> Array[polyadd_shape(Shape1, Shape2)]: ...
@overload
def polyadd(a1: Any, a2: Any) -> Array[IntTuple]: ...
@overload
def polyder[Shape: _Shape, M: Flag[int] = 1](
    p: Array[Shape],
    m: M = 1,
) -> Array[polyder_shape(Shape, M)]: ...
@overload
def polyder(p: Any, m: int = 1) -> Array[IntTuple]: ...
@overload
def polydiv[Shape1: _Shape, Shape2: _Shape](
    u: Array[Shape1],
    v: Array[Shape2],
    *,
    trim_leading_zeros: Literal[False] = False,
) -> tuple[Array[polydiv_quotient_shape(Shape1, Shape2)], Array[Shape1]]: ...
@overload
def polydiv(
    u: Any,
    v: Any,
    *,
    trim_leading_zeros: bool = False,
) -> tuple[Array[IntTuple], Array[IntTuple]]: ...
@overload
def polyfit[Deg: Flag[int]](
    x: Array[Any],
    y: Array[Any],
    deg: Deg,
    rcond: float | None = None,
    full: Literal[False] = False,
    w: Array[Any] | None = None,
    cov: Literal[False] = False,
) -> Array[polyfit_shape(Deg)]: ...
@overload
def polyfit[Deg: Flag[int]](
    x: Array[Any],
    y: Array[Any],
    deg: Deg,
    rcond: float | None = None,
    full: Literal[False] = False,
    w: Array[Any] | None = None,
    cov: Literal[True, "unscaled"] = ...,
) -> tuple[Array[polyfit_shape(Deg)], Array[polyfit_cov_shape(Deg)]]: ...
@overload
def polyfit(
    x: Array[Any],
    y: Array[Any],
    deg: int,
    rcond: float | None = None,
    full: Literal[True] = ...,
    w: Array[Any] | None = None,
    cov: bool = False,
) -> tuple[Array[IntTuple], ...]: ...
@overload
def polyfit(
    x: Any,
    y: Any,
    deg: int,
    rcond: float | None = None,
    full: bool = False,
    w: Any = None,
    cov: bool | str = False,
) -> Any: ...
@overload
def polyint[Shape: _Shape, M: Flag[int] = 1](
    p: Array[Shape],
    m: M = 1,
    k: int | Array[Any] | None = None,
) -> Array[polyint_shape(Shape, M)]: ...
@overload
def polyint(
    p: Any,
    m: int = 1,
    k: int | Any | None = None,
) -> Array[IntTuple]: ...
@overload
def polymul[Shape1: _Shape, Shape2: _Shape](
    a1: Array[Shape1],
    a2: Array[Shape2],
    *,
    trim_leading_zeros: Literal[False] = False,
) -> Array[convolve_shape(Shape1, Shape2, "full")]: ...
@overload
def polymul(
    a1: Any,
    a2: Any,
    *,
    trim_leading_zeros: bool = False,
) -> Array[IntTuple]: ...
@overload
def polysub[Shape1: _Shape, Shape2: _Shape](
    a1: Array[Shape1],
    a2: Array[Shape2],
) -> Array[polyadd_shape(Shape1, Shape2)]: ...
@overload
def polysub(a1: Any, a2: Any) -> Array[IntTuple]: ...
@overload
def polyval[Shape: _Shape](
    p: Array[Any],
    x: Array[Shape],
    *,
    unroll: int = 16,
) -> Array[Shape]: ...
@overload
def polyval(
    p: Any,
    x: Any,
    *,
    unroll: int = 16,
) -> Array[IntTuple]: ...
def roots(p: Array[Any], *, strip_zeros: bool = True) -> Array[IntTuple]: ...

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
