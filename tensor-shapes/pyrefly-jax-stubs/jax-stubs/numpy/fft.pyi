# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, overload, Sequence

from jax._array import Array as _Array, ArrayLike as _ArrayLike
from jax._shapes import (
    fft_shape,
    fftfreq_shape,
    irfft_shape,
    rfft_shape,
    rfftfreq_shape,
)
from jax._src.lib import Device as _Device
from jax.sharding import Sharding as _Sharding
from jax.typing import DTypeLike
from shape_extensions import Flag, Int, IntTuple

type _Shape = IntTuple

# 1D FFT operations
def fft[
    Shape: _Shape = [],
    N: Int | None = None,
    Dim: Flag[int] = -1,
](
    a: _ArrayLike[Shape],
    n: N = None,
    axis: Dim = -1,
    norm: str | None = None,
) -> _Array[fft_shape(Shape, N, Dim)]: ...
def ifft[
    Shape: _Shape = [],
    N: Int | None = None,
    Dim: Flag[int] = -1,
](
    a: _ArrayLike[Shape],
    n: N = None,
    axis: Dim = -1,
    norm: str | None = None,
) -> _Array[fft_shape(Shape, N, Dim)]: ...
def rfft[
    Shape: _Shape = [],
    N: Int | None = None,
    Dim: Flag[int] = -1,
](
    a: _ArrayLike[Shape],
    n: N = None,
    axis: Dim = -1,
    norm: str | None = None,
) -> _Array[rfft_shape(Shape, N, Dim)]: ...
def irfft[
    Shape: _Shape = [],
    N: Int | None = None,
    Dim: Flag[int] = -1,
](
    a: _ArrayLike[Shape],
    n: N = None,
    axis: Dim = -1,
    norm: str | None = None,
) -> _Array[irfft_shape(Shape, N, Dim)]: ...
def hfft[
    Shape: _Shape = [],
    N: Int | None = None,
    Dim: Flag[int] = -1,
](
    a: _ArrayLike[Shape],
    n: N = None,
    axis: Dim = -1,
    norm: str | None = None,
) -> _Array[irfft_shape(Shape, N, Dim)]: ...
def ihfft[
    Shape: _Shape = [],
    N: Int | None = None,
    Dim: Flag[int] = -1,
](
    a: _ArrayLike[Shape],
    n: N = None,
    axis: Dim = -1,
    norm: str | None = None,
) -> _Array[rfft_shape(Shape, N, Dim)]: ...

# 2D FFT operations
@overload
def fft2[Shape: _Shape = []](
    a: _ArrayLike[Shape],
    s: None = None,
    axes: Sequence[int] = (-2, -1),
    norm: str | None = None,
) -> _Array[Shape]: ...
@overload
def fft2(
    a: _ArrayLike[Any],
    s: Sequence[int] | None = None,
    axes: Sequence[int] = (-2, -1),
    norm: str | None = None,
) -> _Array[IntTuple]: ...
@overload
def ifft2[Shape: _Shape = []](
    a: _ArrayLike[Shape],
    s: None = None,
    axes: Sequence[int] = (-2, -1),
    norm: str | None = None,
) -> _Array[Shape]: ...
@overload
def ifft2(
    a: _ArrayLike[Any],
    s: Sequence[int] | None = None,
    axes: Sequence[int] = (-2, -1),
    norm: str | None = None,
) -> _Array[IntTuple]: ...
@overload
def rfft2[Shape: _Shape = []](
    a: _ArrayLike[Shape],
    s: None = None,
    axes: Sequence[int] = (-2, -1),
    norm: str | None = None,
) -> _Array[rfft_shape(Shape, None, -1)]: ...
@overload
def rfft2(
    a: _ArrayLike[Any],
    s: Sequence[int] | None = None,
    axes: Sequence[int] = (-2, -1),
    norm: str | None = None,
) -> _Array[IntTuple]: ...
@overload
def irfft2[Shape: _Shape = []](
    a: _ArrayLike[Shape],
    s: None = None,
    axes: Sequence[int] = (-2, -1),
    norm: str | None = None,
) -> _Array[irfft_shape(Shape, None, -1)]: ...
@overload
def irfft2(
    a: _ArrayLike[Any],
    s: Sequence[int] | None = None,
    axes: Sequence[int] = (-2, -1),
    norm: str | None = None,
) -> _Array[IntTuple]: ...

# ND FFT operations
@overload
def fftn[Shape: _Shape = []](
    a: _ArrayLike[Shape],
    s: None = None,
    axes: Sequence[int] | None = None,
    norm: str | None = None,
) -> _Array[Shape]: ...
@overload
def fftn(
    a: _ArrayLike[Any],
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: str | None = None,
) -> _Array[IntTuple]: ...
@overload
def ifftn[Shape: _Shape = []](
    a: _ArrayLike[Shape],
    s: None = None,
    axes: Sequence[int] | None = None,
    norm: str | None = None,
) -> _Array[Shape]: ...
@overload
def ifftn(
    a: _ArrayLike[Any],
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: str | None = None,
) -> _Array[IntTuple]: ...
@overload
def rfftn[Shape: _Shape = []](
    a: _ArrayLike[Shape],
    s: None = None,
    axes: None = None,
    norm: str | None = None,
) -> _Array[rfft_shape(Shape, None, -1)]: ...
@overload
def rfftn(
    a: _ArrayLike[Any],
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: str | None = None,
) -> _Array[IntTuple]: ...
@overload
def irfftn[Shape: _Shape = []](
    a: _ArrayLike[Shape],
    s: None = None,
    axes: None = None,
    norm: str | None = None,
) -> _Array[irfft_shape(Shape, None, -1)]: ...
@overload
def irfftn(
    a: _ArrayLike[Any],
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: str | None = None,
) -> _Array[IntTuple]: ...

# Frequency helpers
def fftfreq[N: Int](
    n: N,
    d: Any = 1.0,
    *,
    dtype: DTypeLike | None = None,
    device: _Device | _Sharding | None = None,
) -> _Array[fftfreq_shape(N)]: ...
def rfftfreq[N: Int](
    n: N,
    d: Any = 1.0,
    *,
    dtype: DTypeLike | None = None,
    device: _Device | _Sharding | None = None,
) -> _Array[rfftfreq_shape(N)]: ...

# Shift helpers
def fftshift[Shape: _Shape = []](
    x: _ArrayLike[Shape],
    axes: None | int | Sequence[int] = None,
) -> _Array[Shape]: ...
def ifftshift[Shape: _Shape = []](
    x: _ArrayLike[Shape],
    axes: None | int | Sequence[int] = None,
) -> _Array[Shape]: ...
