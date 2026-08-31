# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Sequence, overload

from jax._array import Array
from jax._shapes import (
    fft_n_shape,
    fftfreq_shape,
    irfft_n_shape,
    irfft_shape,
    rfft_n_shape,
    rfft_shape,
    rfftfreq_shape,
)
from shape_extensions import Flag, IntTuple

# 1D FFT operations
@overload
def fft[Shape: IntTuple](
    a: Array[Shape],
    n: None = None,
    axis: int = -1,
    norm: str | None = None,
) -> Array[Shape]: ...
@overload
def fft[Shape: IntTuple, N: Flag[int], Dim: Flag[int]](
    a: Array[Shape],
    n: N,
    axis: Dim = -1,
    norm: str | None = None,
) -> Array[fft_n_shape(Shape, N, Dim)]: ...
@overload
def fft(
    a: Array,
    n: int | None = None,
    axis: int = -1,
    norm: str | None = None,
) -> Array: ...
@overload
def ifft[Shape: IntTuple](
    a: Array[Shape],
    n: None = None,
    axis: int = -1,
    norm: str | None = None,
) -> Array[Shape]: ...
@overload
def ifft[Shape: IntTuple, N: Flag[int], Dim: Flag[int]](
    a: Array[Shape],
    n: N,
    axis: Dim = -1,
    norm: str | None = None,
) -> Array[fft_n_shape(Shape, N, Dim)]: ...
@overload
def ifft(
    a: Array,
    n: int | None = None,
    axis: int = -1,
    norm: str | None = None,
) -> Array: ...
@overload
def rfft[Shape: IntTuple, Dim: Flag[int]](
    a: Array[Shape],
    n: None = None,
    axis: Dim = -1,
    norm: str | None = None,
) -> Array[rfft_shape(Shape, Dim)]: ...
@overload
def rfft[Shape: IntTuple, N: Flag[int], Dim: Flag[int]](
    a: Array[Shape],
    n: N,
    axis: Dim = -1,
    norm: str | None = None,
) -> Array[rfft_n_shape(Shape, N, Dim)]: ...
@overload
def rfft(
    a: Array,
    n: int | None = None,
    axis: int = -1,
    norm: str | None = None,
) -> Array: ...
@overload
def irfft[Shape: IntTuple, Dim: Flag[int]](
    a: Array[Shape],
    n: None = None,
    axis: Dim = -1,
    norm: str | None = None,
) -> Array[irfft_shape(Shape, Dim)]: ...
@overload
def irfft[Shape: IntTuple, N: Flag[int], Dim: Flag[int]](
    a: Array[Shape],
    n: N,
    axis: Dim = -1,
    norm: str | None = None,
) -> Array[irfft_n_shape(Shape, N, Dim)]: ...
@overload
def irfft(
    a: Array,
    n: int | None = None,
    axis: int = -1,
    norm: str | None = None,
) -> Array: ...
@overload
def hfft[Shape: IntTuple, Dim: Flag[int]](
    a: Array[Shape],
    n: None = None,
    axis: Dim = -1,
    norm: str | None = None,
) -> Array[irfft_shape(Shape, Dim)]: ...
@overload
def hfft[Shape: IntTuple, N: Flag[int], Dim: Flag[int]](
    a: Array[Shape],
    n: N,
    axis: Dim = -1,
    norm: str | None = None,
) -> Array[irfft_n_shape(Shape, N, Dim)]: ...
@overload
def hfft(
    a: Array,
    n: int | None = None,
    axis: int = -1,
    norm: str | None = None,
) -> Array: ...
@overload
def ihfft[Shape: IntTuple, Dim: Flag[int]](
    a: Array[Shape],
    n: None = None,
    axis: Dim = -1,
    norm: str | None = None,
) -> Array[rfft_shape(Shape, Dim)]: ...
@overload
def ihfft[Shape: IntTuple, N: Flag[int], Dim: Flag[int]](
    a: Array[Shape],
    n: N,
    axis: Dim = -1,
    norm: str | None = None,
) -> Array[rfft_n_shape(Shape, N, Dim)]: ...
@overload
def ihfft(
    a: Array,
    n: int | None = None,
    axis: int = -1,
    norm: str | None = None,
) -> Array: ...

# 2D FFT operations
@overload
def fft2[Shape: IntTuple](
    a: Array[Shape],
    s: None = None,
    axes: Sequence[int] = (-2, -1),
    norm: str | None = None,
) -> Array[Shape]: ...
@overload
def fft2(
    a: Array,
    s: Sequence[int] | None = None,
    axes: Sequence[int] = (-2, -1),
    norm: str | None = None,
) -> Array: ...
@overload
def ifft2[Shape: IntTuple](
    a: Array[Shape],
    s: None = None,
    axes: Sequence[int] = (-2, -1),
    norm: str | None = None,
) -> Array[Shape]: ...
@overload
def ifft2(
    a: Array,
    s: Sequence[int] | None = None,
    axes: Sequence[int] = (-2, -1),
    norm: str | None = None,
) -> Array: ...
@overload
def rfft2[Shape: IntTuple](
    a: Array[Shape],
    s: None = None,
    axes: Sequence[int] = (-2, -1),
    norm: str | None = None,
) -> Array[rfft_shape(Shape, -1)]: ...
@overload
def rfft2(
    a: Array,
    s: Sequence[int] | None = None,
    axes: Sequence[int] = (-2, -1),
    norm: str | None = None,
) -> Array: ...
@overload
def irfft2[Shape: IntTuple](
    a: Array[Shape],
    s: None = None,
    axes: Sequence[int] = (-2, -1),
    norm: str | None = None,
) -> Array[irfft_shape(Shape, -1)]: ...
@overload
def irfft2(
    a: Array,
    s: Sequence[int] | None = None,
    axes: Sequence[int] = (-2, -1),
    norm: str | None = None,
) -> Array: ...

# ND FFT operations
@overload
def fftn[Shape: IntTuple](
    a: Array[Shape],
    s: None = None,
    axes: Sequence[int] | None = None,
    norm: str | None = None,
) -> Array[Shape]: ...
@overload
def fftn(
    a: Array,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: str | None = None,
) -> Array: ...
@overload
def ifftn[Shape: IntTuple](
    a: Array[Shape],
    s: None = None,
    axes: Sequence[int] | None = None,
    norm: str | None = None,
) -> Array[Shape]: ...
@overload
def ifftn(
    a: Array,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: str | None = None,
) -> Array: ...
@overload
def rfftn[Shape: IntTuple](
    a: Array[Shape],
    s: None = None,
    axes: None = None,
    norm: str | None = None,
) -> Array[rfft_shape(Shape, -1)]: ...
@overload
def rfftn(
    a: Array,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: str | None = None,
) -> Array: ...
@overload
def irfftn[Shape: IntTuple](
    a: Array[Shape],
    s: None = None,
    axes: None = None,
    norm: str | None = None,
) -> Array[irfft_shape(Shape, -1)]: ...
@overload
def irfftn(
    a: Array,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: str | None = None,
) -> Array: ...

# Frequency helpers
@overload
def fftfreq[N: Flag[int]](
    n: N,
    d: Any = 1.0,
    *,
    dtype: Any = None,
    device: Any = None,
) -> Array[fftfreq_shape(N)]: ...
@overload
def fftfreq(
    n: int,
    d: Any = 1.0,
    *,
    dtype: Any = None,
    device: Any = None,
) -> Array[IntTuple]: ...
@overload
def rfftfreq[N: Flag[int]](
    n: N,
    d: Any = 1.0,
    *,
    dtype: Any = None,
    device: Any = None,
) -> Array[rfftfreq_shape(N)]: ...
@overload
def rfftfreq(
    n: int,
    d: Any = 1.0,
    *,
    dtype: Any = None,
    device: Any = None,
) -> Array[IntTuple]: ...

# Shift helpers
def fftshift[Shape: IntTuple](
    x: Array[Shape],
    axes: None | int | Sequence[int] = None,
) -> Array[Shape]: ...
def ifftshift[Shape: IntTuple](
    x: Array[Shape],
    axes: None | int | Sequence[int] = None,
) -> Array[Shape]: ...
