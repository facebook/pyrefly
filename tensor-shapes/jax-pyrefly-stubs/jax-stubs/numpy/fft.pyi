# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Sequence

from jax._array import Array as _Array, ArrayLike as _ArrayLike
from jax._shapes import (
    fft2_shape,
    fft_shape,
    fftfreq_shape,
    fftn_shape,
    irfft2_shape,
    irfft_shape,
    irfftn_shape,
    rfft2_shape,
    rfft_shape,
    rfftfreq_shape,
    rfftn_shape,
)
from jax._src.lib import Device as _Device
from jax.sharding import Sharding as _Sharding
from jax.typing import DTypeLike
from shape_extensions import Flag, Int, IntTuple

type _Shape = IntTuple
type _Axis = int | tuple[int, ...] | None

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
def fft2[
    Shape: _Shape = [],
    S: Flag[_Axis] = None,
    Axes: Flag[_Axis] = (-2, -1),
](
    a: _ArrayLike[Shape],
    s: S = None,
    axes: Axes = (-2, -1),
    norm: str | None = None,
) -> _Array[fft2_shape(Shape, S, Axes)]: ...
def ifft2[
    Shape: _Shape = [],
    S: Flag[_Axis] = None,
    Axes: Flag[_Axis] = (-2, -1),
](
    a: _ArrayLike[Shape],
    s: S = None,
    axes: Axes = (-2, -1),
    norm: str | None = None,
) -> _Array[fft2_shape(Shape, S, Axes)]: ...
def rfft2[
    Shape: _Shape = [],
    S: Flag[_Axis] = None,
    Axes: Flag[_Axis] = (-2, -1),
](
    a: _ArrayLike[Shape],
    s: S = None,
    axes: Axes = (-2, -1),
    norm: str | None = None,
) -> _Array[rfft2_shape(Shape, S, Axes)]: ...
def irfft2[
    Shape: _Shape = [],
    S: Flag[_Axis] = None,
    Axes: Flag[_Axis] = (-2, -1),
](
    a: _ArrayLike[Shape],
    s: S = None,
    axes: Axes = (-2, -1),
    norm: str | None = None,
) -> _Array[irfft2_shape(Shape, S, Axes)]: ...

# ND FFT operations
def fftn[
    Shape: _Shape = [],
    S: Flag[_Axis] = None,
    Axes: Flag[_Axis] = None,
](
    a: _ArrayLike[Shape],
    s: S = None,
    axes: Axes = None,
    norm: str | None = None,
) -> _Array[fftn_shape(Shape, S, Axes)]: ...
def ifftn[
    Shape: _Shape = [],
    S: Flag[_Axis] = None,
    Axes: Flag[_Axis] = None,
](
    a: _ArrayLike[Shape],
    s: S = None,
    axes: Axes = None,
    norm: str | None = None,
) -> _Array[fftn_shape(Shape, S, Axes)]: ...
def rfftn[
    Shape: _Shape = [],
    S: Flag[_Axis] = None,
    Axes: Flag[_Axis] = None,
](
    a: _ArrayLike[Shape],
    s: S = None,
    axes: Axes = None,
    norm: str | None = None,
) -> _Array[rfftn_shape(Shape, S, Axes)]: ...
def irfftn[
    Shape: _Shape = [],
    S: Flag[_Axis] = None,
    Axes: Flag[_Axis] = None,
](
    a: _ArrayLike[Shape],
    s: S = None,
    axes: Axes = None,
    norm: str | None = None,
) -> _Array[irfftn_shape(Shape, S, Axes)]: ...

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
