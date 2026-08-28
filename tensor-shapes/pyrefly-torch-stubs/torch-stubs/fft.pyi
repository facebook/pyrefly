# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Type stubs for torch.fft module (Phase 6: FFT Operations)
from typing import overload

from shape_extensions import Flag, Int as _Int, IntTuple, IntVar
from torch import Tensor
from torch._shapes import (
    irfft_literal_shape,
    irfft_n_shape,
    irfft_shape,
    rfft_literal_shape,
    rfft_n_shape,
    rfft_shape,
)

# 1D FFT operations
def fft[Shape: IntTuple](
    input: Tensor[Shape], n: int = None, dim: int = -1, norm: str = None
) -> Tensor[Shape]: ...
def ifft[Shape: IntTuple](
    input: Tensor[Shape], n: int = None, dim: int = -1, norm: str = None
) -> Tensor[Shape]: ...
@overload
def rfft[Shape: IntTuple, Dim: Flag[int]](
    self: Tensor[Shape], n: None = None, dim: Dim = -1, norm: str = None
) -> Tensor[rfft_shape(Shape, Dim)]: ...
@overload
def rfft[Shape: IntTuple, N: IntVar, Dim: Flag[int]](
    self: Tensor[Shape], n: _Int[N], dim: Dim = -1, norm: str = None
) -> Tensor[rfft_n_shape(Shape, N, Dim)]: ...
@overload
def rfft[Shape: IntTuple, N: Flag[int], Dim: Flag[int]](
    self: Tensor[Shape], n: N, dim: Dim = -1, norm: str = None
) -> Tensor[rfft_literal_shape(Shape, N, Dim)]: ...
@overload
def rfft(self: Tensor, n: int, dim: int = -1, norm: str = None) -> Tensor: ...
@overload
def irfft[Shape: IntTuple, Dim: Flag[int]](
    self: Tensor[Shape], n: None = None, dim: Dim = -1, norm: str = None
) -> Tensor[irfft_shape(Shape, Dim)]: ...
@overload
def irfft[Shape: IntTuple, N: IntVar, Dim: Flag[int]](
    self: Tensor[Shape], n: _Int[N], dim: Dim = -1, norm: str = None
) -> Tensor[irfft_n_shape(Shape, N, Dim)]: ...
@overload
def irfft[Shape: IntTuple, N: Flag[int], Dim: Flag[int]](
    self: Tensor[Shape], n: N, dim: Dim = -1, norm: str = None
) -> Tensor[irfft_literal_shape(Shape, N, Dim)]: ...
@overload
def irfft(self: Tensor, n: int, dim: int = -1, norm: str = None) -> Tensor: ...
@overload
def hfft[Shape: IntTuple, Dim: Flag[int]](
    self: Tensor[Shape], n: None = None, dim: Dim = -1, norm: str = None
) -> Tensor[irfft_shape(Shape, Dim)]: ...
@overload
def hfft[Shape: IntTuple, N: IntVar, Dim: Flag[int]](
    self: Tensor[Shape], n: _Int[N], dim: Dim = -1, norm: str = None
) -> Tensor[irfft_n_shape(Shape, N, Dim)]: ...
@overload
def hfft[Shape: IntTuple, N: Flag[int], Dim: Flag[int]](
    self: Tensor[Shape], n: N, dim: Dim = -1, norm: str = None
) -> Tensor[irfft_literal_shape(Shape, N, Dim)]: ...
@overload
def hfft(self: Tensor, n: int, dim: int = -1, norm: str = None) -> Tensor: ...
@overload
def ihfft[Shape: IntTuple, Dim: Flag[int]](
    self: Tensor[Shape], n: None = None, dim: Dim = -1, norm: str = None
) -> Tensor[rfft_shape(Shape, Dim)]: ...
@overload
def ihfft[Shape: IntTuple, N: IntVar, Dim: Flag[int]](
    self: Tensor[Shape], n: _Int[N], dim: Dim = -1, norm: str = None
) -> Tensor[rfft_n_shape(Shape, N, Dim)]: ...
@overload
def ihfft[Shape: IntTuple, N: Flag[int], Dim: Flag[int]](
    self: Tensor[Shape], n: N, dim: Dim = -1, norm: str = None
) -> Tensor[rfft_literal_shape(Shape, N, Dim)]: ...
@overload
def ihfft(self: Tensor, n: int, dim: int = -1, norm: str = None) -> Tensor: ...

# 2D FFT operations
def fft2[Shape: IntTuple](
    input: Tensor[Shape],
    s: tuple[int, int] = None,
    dim: tuple[int, int] = (-2, -1),
    norm: str = None,
) -> Tensor[Shape]: ...
def ifft2[Shape: IntTuple](
    input: Tensor[Shape],
    s: tuple[int, int] = None,
    dim: tuple[int, int] = (-2, -1),
    norm: str = None,
) -> Tensor[Shape]: ...
def rfft2(
    input: Tensor,
    s: tuple[int, int] = None,
    dim: tuple[int, int] = (-2, -1),
    norm: str = None,
) -> Tensor: ...
def irfft2(
    input: Tensor,
    s: tuple[int, int] = None,
    dim: tuple[int, int] = (-2, -1),
    norm: str = None,
) -> Tensor: ...

# ND FFT operations
def fftn[Shape: IntTuple](
    input: Tensor[Shape],
    s: tuple[int, ...] = None,
    dim: tuple[int, ...] = None,
    norm: str = None,
) -> Tensor[Shape]: ...
def ifftn[Shape: IntTuple](
    input: Tensor[Shape],
    s: tuple[int, ...] = None,
    dim: tuple[int, ...] = None,
    norm: str = None,
) -> Tensor[Shape]: ...
def rfftn(
    input: Tensor,
    s: tuple[int, ...] = None,
    dim: tuple[int, ...] = None,
    norm: str = None,
) -> Tensor: ...
def irfftn(
    input: Tensor,
    s: tuple[int, ...] = None,
    dim: tuple[int, ...] = None,
    norm: str = None,
) -> Tensor: ...

# FFT shift operations
def fftshift[Shape: IntTuple](
    input: Tensor[Shape], dim: int | tuple[int, ...] = None
) -> Tensor[Shape]: ...
def ifftshift[Shape: IntTuple](
    input: Tensor[Shape], dim: int | tuple[int, ...] = None
) -> Tensor[Shape]: ...
