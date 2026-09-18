# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Type stubs for torch.fft module (Phase 6: FFT Operations)
from typing import TYPE_CHECKING

from shape_extensions import Flag, IntTuple
from torch import Tensor
from torch._shapes import irfft_shape, rfft_shape

if TYPE_CHECKING:
    from shape_extensions import Int as _Int

# 1D FFT operations
def fft[Shape: IntTuple](
    input: Tensor[Shape], n: int = None, dim: int = -1, norm: str = None
) -> Tensor[Shape]: ...
def ifft[Shape: IntTuple](
    input: Tensor[Shape], n: int = None, dim: int = -1, norm: str = None
) -> Tensor[Shape]: ...
def rfft[Shape: IntTuple, N: _Int | None, Dim: Flag[int]](
    input: Tensor[Shape], n: N = None, dim: Dim = -1, norm: str = None
) -> Tensor[rfft_shape(Shape, N, Dim)]: ...
def irfft[Shape: IntTuple, N: _Int | None, Dim: Flag[int]](
    input: Tensor[Shape], n: N = None, dim: Dim = -1, norm: str = None
) -> Tensor[irfft_shape(Shape, N, Dim)]: ...
def hfft[Shape: IntTuple, N: _Int | None, Dim: Flag[int]](
    input: Tensor[Shape], n: N = None, dim: Dim = -1, norm: str = None
) -> Tensor[irfft_shape(Shape, N, Dim)]: ...
def ihfft[Shape: IntTuple, N: _Int | None, Dim: Flag[int]](
    input: Tensor[Shape], n: N = None, dim: Dim = -1, norm: str = None
) -> Tensor[rfft_shape(Shape, N, Dim)]: ...

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
