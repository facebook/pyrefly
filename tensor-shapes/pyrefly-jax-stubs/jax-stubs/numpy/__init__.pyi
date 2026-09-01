# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Literal, overload, Sequence, Unpack

from jax._array import Array as Array, Array as ndarray
from jax._shapes import (
    diagonal_shape,
    dot_shape,
    einsum_shape,
    inner_shape,
    int_min,
    kron_shape,
    matmul_shape,
    matvec_shape,
    permute_shape,
    reduce_shape,
    reshape_shape,
    reverse_shape,
    tensordot_shape,
    trace_shape,
    vecmat_shape,
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

# Ranks 1 through 3 are exact; any other integer sequence, including a longer
# tuple or a list, falls through to a gradual overload rather than being
# rejected.
# TODO(stroxler): Replace these finite tuple-shape constructor overloads with a
# single `Shape: tuple[int, ...]` overload once carrier shapes flow through
# downstream shaped-array operations without degrading to unknown. The NumPy
# stubs carry the same limitation.
@overload
def zeros(shape: tuple[()], dtype: Any = ..., *, device: Any = ...) -> Array[[]]: ...
@overload
def zeros[N: IntVar](
    shape: Int[N], dtype: Any = ..., *, device: Any = ...
) -> Array[[N]]: ...
@overload
def zeros[Shape: _Shape](
    shape: Shape, dtype: Any = ..., *, device: Any = ...
) -> Array[Shape]: ...
@overload
def zeros(
    shape: Sequence[int] | int, dtype: Any = ..., *, device: Any = ...
) -> Array[IntTuple]: ...
@overload
def ones(shape: tuple[()], dtype: Any = ..., *, device: Any = ...) -> Array[[]]: ...
@overload
def ones[N: IntVar](
    shape: Int[N], dtype: Any = ..., *, device: Any = ...
) -> Array[[N]]: ...
@overload
def ones[Shape: _Shape](
    shape: Shape, dtype: Any = ..., *, device: Any = ...
) -> Array[Shape]: ...
@overload
def ones(
    shape: Sequence[int] | int, dtype: Any = ..., *, device: Any = ...
) -> Array[IntTuple]: ...
@overload
def empty[N: IntVar](
    shape: Int[N],
    dtype: Any = ...,
    *,
    device: Any = ...,
    out_sharding: Any = ...,
) -> Array[[N]]: ...
@overload
def empty[Shape: _Shape](
    shape: Shape,
    dtype: Any = ...,
    *,
    device: Any = ...,
    out_sharding: Any = ...,
) -> Array[Shape]: ...
@overload
def empty(
    shape: Sequence[int] | int,
    dtype: Any = ...,
    *,
    device: Any = ...,
    out_sharding: Any = ...,
) -> Array[IntTuple]: ...
@overload
def full(
    shape: tuple[()], fill_value: Any, dtype: Any = ..., *, device: Any = ...
) -> Array[[]]: ...
@overload
def full[N: IntVar](
    shape: Int[N], fill_value: Any, dtype: Any = ..., *, device: Any = ...
) -> Array[[N]]: ...
@overload
def full[Shape: _Shape](
    shape: Shape, fill_value: Any, dtype: Any = ..., *, device: Any = ...
) -> Array[Shape]: ...
@overload
def full(
    shape: Sequence[int] | int,
    fill_value: Any,
    dtype: Any = ...,
    *,
    device: Any = ...,
) -> Array[IntTuple]: ...

# `_like` constructors
@overload
def empty_like[Shape: _Shape](
    prototype: Array[Shape],
    dtype: Any = ...,
    shape: None = None,
    *,
    device: Any = ...,
) -> Array[Shape]: ...
@overload
def empty_like[N: IntVar](
    prototype: Any,
    dtype: Any = ...,
    shape: Int[N] = ...,
    *,
    device: Any = ...,
) -> Array[[N]]: ...
@overload
def empty_like[Shape: _Shape](
    prototype: Any,
    dtype: Any = ...,
    shape: Shape = ...,
    *,
    device: Any = ...,
) -> Array[Shape]: ...
@overload
def empty_like(
    prototype: Any,
    dtype: Any = ...,
    shape: Sequence[int] | int | None = None,
    *,
    device: Any = ...,
) -> Array[IntTuple]: ...
@overload
def zeros_like[Shape: _Shape](
    a: Array[Shape],
    dtype: Any = ...,
    shape: None = None,
    *,
    device: Any = ...,
    out_sharding: Any = ...,
) -> Array[Shape]: ...
@overload
def zeros_like[N: IntVar](
    a: Any,
    dtype: Any = ...,
    shape: Int[N] = ...,
    *,
    device: Any = ...,
    out_sharding: Any = ...,
) -> Array[[N]]: ...
@overload
def zeros_like[Shape: _Shape](
    a: Any,
    dtype: Any = ...,
    shape: Shape = ...,
    *,
    device: Any = ...,
    out_sharding: Any = ...,
) -> Array[Shape]: ...
@overload
def zeros_like(
    a: Any,
    dtype: Any = ...,
    shape: Sequence[int] | int | None = None,
    *,
    device: Any = ...,
    out_sharding: Any = ...,
) -> Array[IntTuple]: ...
@overload
def ones_like[Shape: _Shape](
    a: Array[Shape],
    dtype: Any = ...,
    shape: None = None,
    *,
    device: Any = ...,
    out_sharding: Any = ...,
) -> Array[Shape]: ...
@overload
def ones_like[N: IntVar](
    a: Any,
    dtype: Any = ...,
    shape: Int[N] = ...,
    *,
    device: Any = ...,
    out_sharding: Any = ...,
) -> Array[[N]]: ...
@overload
def ones_like[Shape: _Shape](
    a: Any,
    dtype: Any = ...,
    shape: Shape = ...,
    *,
    device: Any = ...,
    out_sharding: Any = ...,
) -> Array[Shape]: ...
@overload
def ones_like(
    a: Any,
    dtype: Any = ...,
    shape: Sequence[int] | int | None = None,
    *,
    device: Any = ...,
    out_sharding: Any = ...,
) -> Array[IntTuple]: ...
@overload
def full_like[Shape: _Shape](
    a: Array[Shape],
    fill_value: Any,
    dtype: Any = ...,
    shape: None = None,
    *,
    device: Any = ...,
    out_sharding: Any = ...,
) -> Array[Shape]: ...
@overload
def full_like[N: IntVar](
    a: Any,
    fill_value: Any,
    dtype: Any = ...,
    shape: Int[N] = ...,
    *,
    device: Any = ...,
    out_sharding: Any = ...,
) -> Array[[N]]: ...
@overload
def full_like[Shape: _Shape](
    a: Any,
    fill_value: Any,
    dtype: Any = ...,
    shape: Shape = ...,
    *,
    device: Any = ...,
    out_sharding: Any = ...,
) -> Array[Shape]: ...
@overload
def full_like(
    a: Any,
    fill_value: Any,
    dtype: Any = ...,
    shape: Sequence[int] | int | None = None,
    *,
    device: Any = ...,
    out_sharding: Any = ...,
) -> Array[IntTuple]: ...

# `arange`, `linspace`, `logspace`, `geomspace`
@overload
def arange[N: IntVar](
    start: Int[N], *, dtype: Any = ..., device: Any = ...
) -> Array[[N]]: ...
@overload
def arange(start: float, *, dtype: Any = ..., device: Any = ...) -> Array[[int]]: ...
@overload
def arange(
    start: int | float,
    stop: int | float,
    step: int | float = ...,
    dtype: Any = ...,
) -> Array[[int]]: ...
@overload
def linspace[N: IntVar](
    start: Any,
    stop: Any,
    num: Int[N],
    endpoint: bool = True,
    retstep: Literal[False] = False,
    dtype: Any = None,
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
    dtype: Any = None,
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
    dtype: Any = None,
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
    dtype: Any = None,
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
    dtype: Any = None,
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
    dtype: Any = None,
    axis: int = 0,
) -> Array[[N]]: ...
@overload
def logspace(
    start: Any,
    stop: Any,
    num: int = 50,
    endpoint: bool = True,
    base: Any = 10.0,
    dtype: Any = None,
    axis: int = 0,
) -> Array[[int]]: ...
@overload
def geomspace[N: IntVar](
    start: Any,
    stop: Any,
    num: Int[N],
    endpoint: bool = True,
    dtype: Any = None,
    axis: int = 0,
) -> Array[[N]]: ...
@overload
def geomspace(
    start: Any,
    stop: Any,
    num: int = 50,
    endpoint: bool = True,
    dtype: Any = None,
    axis: int = 0,
) -> Array[[int]]: ...

# `eye`, `identity`, `diag`, `diagflat`, `tri`, `tril`, `triu`, `vander`
@overload
def eye[N: IntVar](
    N: Int[N], M: None = ..., k: int = ..., dtype: Any = ..., *, device: Any = ...
) -> Array[[N, N]]: ...
@overload
def eye[N: IntVar, M: IntVar](
    N: Int[N], M: Int[M], k: int = ..., dtype: Any = ..., *, device: Any = ...
) -> Array[[N, M]]: ...
def identity[N: IntVar](
    n: Int[N], dtype: Any = ..., *, device: Any = ...
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
    N: Int[N], M: None = None, k: int = 0, dtype: Any = None
) -> Array[[N, N]]: ...
@overload
def tri[N: IntVar, M: IntVar](
    N: Int[N], M: Int[M], k: int = 0, dtype: Any = None
) -> Array[[N, M]]: ...
@overload
def tri(
    N: int, M: int | None = None, k: int = 0, dtype: Any = None
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
    dtype: Any = None,
    sparse: Literal[False] = False,
) -> Array[[1, N]]: ...
@overload
def indices[N: IntVar, M: IntVar](
    dimensions: IntTuple[N, M],
    dtype: Any = None,
    sparse: Literal[False] = False,
) -> Array[[2, N, M]]: ...
@overload
def indices[N: IntVar, M: IntVar, K: IntVar](
    dimensions: IntTuple[N, M, K],
    dtype: Any = None,
    sparse: Literal[False] = False,
) -> Array[[3, N, M, K]]: ...
@overload
def indices(
    dimensions: Sequence[int], dtype: Any = None, sparse: bool = False
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
    buffer: Any, dtype: Any = float, count: int = -1, offset: int = 0
) -> Array[IntTuple]: ...
def fromfile(*args: Any, **kwargs: Any) -> Array[IntTuple]: ...
@overload
def fromfunction[N: IntVar](
    function: Callable[..., Any],
    shape: IntTuple[N],
    *,
    dtype: Any = float,
    **kwargs: Any,
) -> Array[[N]]: ...
@overload
def fromfunction[N: IntVar, M: IntVar](
    function: Callable[..., Any],
    shape: IntTuple[N, M],
    *,
    dtype: Any = float,
    **kwargs: Any,
) -> Array[[N, M]]: ...
@overload
def fromfunction[N: IntVar, M: IntVar, K: IntVar](
    function: Callable[..., Any],
    shape: IntTuple[N, M, K],
    *,
    dtype: Any = float,
    **kwargs: Any,
) -> Array[[N, M, K]]: ...
@overload
def fromfunction(
    function: Callable[..., Any],
    shape: Sequence[int],
    *,
    dtype: Any = float,
    **kwargs: Any,
) -> Array[IntTuple]: ...
def fromiter(*args: Any, **kwargs: Any) -> Array[IntTuple]: ...
def fromstring(
    string: str, dtype: Any = float, count: int = -1, *, sep: str
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

# Shape semantics, including vector and batched operands, come from the gufunc-backed helper.
def cross[Shape1: _Shape, Shape2: _Shape](
    a: Array[Shape1],
    b: Array[Shape2],
    /,
    axisa: int = -1,
    axisb: int = -1,
    axisc: int = -1,
    axis: int | None = None,
) -> Array[broadcast(Shape1, Shape2)]: ...
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
    dtype: Any = None,
    out: None = None,
) -> Array[trace_shape(Shape, Offset, Axis1, Axis2)]: ...
@overload
def trace(
    a: Array[Any],
    offset: int = 0,
    axis1: int = 0,
    axis2: int = 1,
    dtype: Any = None,
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
def reshape(
    a: Array[Any],
    shape: Sequence[int],
    order: str = ...,
    *,
    copy: bool | None = ...,
    out_sharding: Any = ...,
) -> Array[IntTuple]: ...

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
    dtype: Any = ...,
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
    dtype: Any = ...,
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
    dtype: Any = ...,
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
    dtype: Any = ...,
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
    dtype: Any = ...,
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
    dtype: Any = ...,
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
    dtype: Any = ...,
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
    dtype: Any = ...,
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
    dtype: Any = ...,
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
    dtype: Any = ...,
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
    dtype: Any = None,
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
    dtype: Any = None,
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
    dtype: Any = None,
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
    dtype: Any = None,
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
    dtype: Any = None,
    out: Any = None,
    keepdims: KeepDims = False,
    initial: Any = None,
    where: Any = None,
) -> Array[reduce_shape(Shape, Axis, KeepDims)]: ...
@overload
def nansum[Shape: _Shape](
    a: Array[Shape],
    axis: Sequence[int],
    dtype: Any = None,
    out: Any = None,
    keepdims: bool = False,
    initial: Any = None,
    where: Any = None,
) -> Array[IntTuple]: ...
@overload
def nanprod[Shape: _Shape, Axis: Flag[_Axis], KeepDims: Flag[bool]](
    a: Array[Shape],
    axis: Axis = None,
    dtype: Any = None,
    out: Any = None,
    keepdims: KeepDims = False,
    initial: Any = None,
    where: Any = None,
) -> Array[reduce_shape(Shape, Axis, KeepDims)]: ...
@overload
def nanprod[Shape: _Shape](
    a: Array[Shape],
    axis: Sequence[int],
    dtype: Any = None,
    out: Any = None,
    keepdims: bool = False,
    initial: Any = None,
    where: Any = None,
) -> Array[IntTuple]: ...
@overload
def nanmean[Shape: _Shape, Axis: Flag[_Axis], KeepDims: Flag[bool]](
    a: Array[Shape],
    axis: Axis = None,
    dtype: Any = None,
    out: Any = None,
    keepdims: KeepDims = False,
    where: Any = None,
) -> Array[reduce_shape(Shape, Axis, KeepDims)]: ...
@overload
def nanmean[Shape: _Shape](
    a: Array[Shape],
    axis: Sequence[int],
    dtype: Any = None,
    out: Any = None,
    keepdims: bool = False,
    where: Any = None,
) -> Array[IntTuple]: ...
@overload
def nanstd[Shape: _Shape, Axis: Flag[_Axis], KeepDims: Flag[bool]](
    a: Array[Shape],
    axis: Axis = None,
    dtype: Any = None,
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
    dtype: Any = None,
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
    dtype: Any = None,
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
    dtype: Any = None,
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
    dtype: Any = None,
    out: Any = None,
) -> Array[Shape]: ...
@overload
def cumsum(
    a: Array[Any],
    axis: None = None,
    dtype: Any = None,
    out: Any = None,
) -> Array[IntTuple]: ...
@overload
def cumprod[Shape: _Shape](
    a: Array[Shape],
    axis: int,
    dtype: Any = None,
    out: Any = None,
) -> Array[Shape]: ...
@overload
def cumprod(
    a: Array[Any],
    axis: None = None,
    dtype: Any = None,
    out: Any = None,
) -> Array[IntTuple]: ...
@overload
def cumulative_sum[Shape: _Shape](
    x: Array[Shape],
    /,
    *,
    axis: int,
    dtype: Any = None,
    include_initial: Literal[False] = False,
) -> Array[Shape]: ...
@overload
def cumulative_sum(
    x: Array[Any],
    /,
    *,
    axis: int | None = None,
    dtype: Any = None,
    include_initial: bool = False,
) -> Array[IntTuple]: ...
@overload
def cumulative_prod[Shape: _Shape](
    x: Array[Shape],
    /,
    *,
    axis: int,
    dtype: Any = None,
    include_initial: Literal[False] = False,
) -> Array[Shape]: ...
@overload
def cumulative_prod(
    x: Array[Any],
    /,
    *,
    axis: int | None = None,
    dtype: Any = None,
    include_initial: bool = False,
) -> Array[IntTuple]: ...
@overload
def nancumsum[Shape: _Shape](
    a: Array[Shape],
    axis: int,
    dtype: Any = None,
    out: Any = None,
) -> Array[Shape]: ...
@overload
def nancumsum(
    a: Array[Any],
    axis: None = None,
    dtype: Any = None,
    out: Any = None,
) -> Array[IntTuple]: ...
@overload
def nancumprod[Shape: _Shape](
    a: Array[Shape],
    axis: int,
    dtype: Any = None,
    out: Any = None,
) -> Array[Shape]: ...
@overload
def nancumprod(
    a: Array[Any],
    axis: None = None,
    dtype: Any = None,
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
    dtype: Any = None,
) -> Array[IntTuple]: ...
def cov(
    m: Array[Any],
    y: Any = None,
    rowvar: bool = True,
    bias: bool = False,
    ddof: int | None = None,
    fweights: Any = None,
    aweights: Any = None,
    dtype: Any = None,
) -> Array[IntTuple]: ...

float32: Any
float64: Any
int32: Any
int64: Any
bool_: Any
