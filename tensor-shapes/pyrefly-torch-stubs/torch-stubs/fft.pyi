# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Type stubs for torch.fft module (Phase 6: FFT Operations)
from typing import Any, Literal, overload, TYPE_CHECKING

from shape_extensions import Flag, IntTuple, IntVar
from torch import Tensor as Tensor
from torch._shapes import (
    fft_shape,
    hermitian_fft_shape,
    irfft2_default_shape,
    irfft_shape,
    rfft2_default_shape,
    rfft_shape,
)

if TYPE_CHECKING:
    from shape_extensions import Int as _Int

# 1D FFT operations
def fft[Shape: IntTuple, N: _Int | None, Dim: Flag[int]](
    input: Tensor[Shape],
    n: N = None,
    dim: Dim = -1,
    norm: str | None = None,
) -> Tensor[fft_shape(Shape, N, Dim)]: ...
def ifft[Shape: IntTuple, N: _Int | None, Dim: Flag[int]](
    input: Tensor[Shape],
    n: N = None,
    dim: Dim = -1,
    norm: str | None = None,
) -> Tensor[fft_shape(Shape, N, Dim)]: ...
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
@overload
def fft2[Shape: IntTuple](
    input: Tensor[Shape],
    s: None = None,
    dim: tuple[int, int] = (-2, -1),
    norm: str | None = None,
) -> Tensor[Shape]: ...
@overload
def fft2(
    input: Tensor,
    s: tuple[int, int] | None = None,
    dim: tuple[int, int] = (-2, -1),
    norm: str | None = None,
) -> Tensor[IntTuple]: ...
@overload
def ifft2[Shape: IntTuple](
    input: Tensor[Shape],
    s: None = None,
    dim: tuple[int, int] = (-2, -1),
    norm: str | None = None,
) -> Tensor[Shape]: ...
@overload
def ifft2(
    input: Tensor,
    s: tuple[int, int] | None = None,
    dim: tuple[int, int] = (-2, -1),
    norm: str | None = None,
) -> Tensor[IntTuple]: ...
@overload
def rfft2[Shape: IntTuple](
    input: Tensor[Shape],
    s: None = None,
    dim: tuple[Literal[-2], Literal[-1]] = (-2, -1),
    norm: str | None = None,
) -> Tensor[rfft2_default_shape(Shape)]: ...
@overload
def rfft2(
    input: Tensor,
    s: tuple[int, int] | None = None,
    dim: tuple[int, int] = (-2, -1),
    norm: str | None = None,
) -> Tensor[IntTuple]: ...
@overload
def irfft2[Shape: IntTuple](
    input: Tensor[Shape],
    s: None = None,
    dim: tuple[Literal[-2], Literal[-1]] = (-2, -1),
    norm: str | None = None,
) -> Tensor[irfft2_default_shape(Shape)]: ...
@overload
def irfft2(
    input: Tensor,
    s: tuple[int, int] | None = None,
    dim: tuple[int, int] = (-2, -1),
    norm: str | None = None,
) -> Tensor[IntTuple]: ...

# ND FFT operations
@overload
def fftn[Shape: IntTuple](
    input: Tensor[Shape],
    s: None = None,
    dim: tuple[int, ...] | None = None,
    norm: str | None = None,
) -> Tensor[Shape]: ...
@overload
def fftn(
    input: Tensor,
    s: tuple[int, ...] | None = None,
    dim: tuple[int, ...] | None = None,
    norm: str | None = None,
) -> Tensor[IntTuple]: ...
@overload
def ifftn[Shape: IntTuple](
    input: Tensor[Shape],
    s: None = None,
    dim: tuple[int, ...] | None = None,
    norm: str | None = None,
) -> Tensor[Shape]: ...
@overload
def ifftn(
    input: Tensor,
    s: tuple[int, ...] | None = None,
    dim: tuple[int, ...] | None = None,
    norm: str | None = None,
) -> Tensor[IntTuple]: ...
@overload
def rfftn[Shape: IntTuple](
    input: Tensor[Shape],
    s: None = None,
    dim: None = None,
    norm: str | None = None,
) -> Tensor[rfft_shape(Shape, None, -1)]: ...
@overload
def rfftn(
    input: Tensor,
    s: tuple[int, ...] | None = None,
    dim: tuple[int, ...] | None = None,
    norm: str | None = None,
) -> Tensor[IntTuple]: ...
@overload
def irfftn[Shape: IntTuple](
    input: Tensor[Shape],
    s: None = None,
    dim: None = None,
    norm: str | None = None,
) -> Tensor[irfft_shape(Shape, None, -1)]: ...
@overload
def irfftn(
    input: Tensor,
    s: tuple[int, ...] | None = None,
    dim: tuple[int, ...] | None = None,
    norm: str | None = None,
) -> Tensor[IntTuple]: ...

# FFT shift operations
def fftshift[Shape: IntTuple](
    input: Tensor[Shape], dim: int | tuple[int, ...] = None
) -> Tensor[Shape]: ...
def ifftshift[Shape: IntTuple](
    input: Tensor[Shape], dim: int | tuple[int, ...] = None
) -> Tensor[Shape]: ...
def fftfreq[N: IntVar](
    n: _Int[N],
    d: float = 1.0,
    *,
    out: Tensor | None = None,
    dtype: Any = None,
    layout: Any = ...,
    device: Any = None,
    requires_grad: bool = False,
) -> Tensor[[N]]: ...
@overload
def hfft2[Shape: IntTuple, Dim: Flag[tuple[int, ...]]](
    input: Tensor[Shape],
    s: None = None,
    dim: Dim = (-2, -1),
    norm: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor[hermitian_fft_shape(Shape, Shape, Dim, False, False)]: ...
@overload
def hfft2[Shape: IntTuple, Sizes: IntTuple, Dim: Flag[tuple[int, ...]]](
    input: Tensor[Shape],
    s: Sizes,
    dim: Dim = (-2, -1),
    norm: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor[hermitian_fft_shape(Shape, Sizes, Dim, False, True)]: ...
@overload
def ihfft2[Shape: IntTuple, Dim: Flag[tuple[int, ...]]](
    input: Tensor[Shape],
    s: None = None,
    dim: Dim = (-2, -1),
    norm: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor[hermitian_fft_shape(Shape, Shape, Dim, True, False)]: ...
@overload
def ihfft2[Shape: IntTuple, Sizes: IntTuple, Dim: Flag[tuple[int, ...]]](
    input: Tensor[Shape],
    s: Sizes,
    dim: Dim = (-2, -1),
    norm: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor[hermitian_fft_shape(Shape, Sizes, Dim, True, True)]: ...
@overload
def hfftn[Shape: IntTuple, Dim: Flag[tuple[int, ...] | None]](
    input: Tensor[Shape],
    s: None = None,
    dim: Dim = None,
    norm: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor[hermitian_fft_shape(Shape, Shape, Dim, False, False)]: ...
@overload
def hfftn[Shape: IntTuple, Sizes: IntTuple, Dim: Flag[tuple[int, ...] | None]](
    input: Tensor[Shape],
    s: Sizes,
    dim: Dim = None,
    norm: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor[hermitian_fft_shape(Shape, Sizes, Dim, False, True)]: ...
@overload
def ihfftn[Shape: IntTuple, Dim: Flag[tuple[int, ...] | None]](
    input: Tensor[Shape],
    s: None = None,
    dim: Dim = None,
    norm: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor[hermitian_fft_shape(Shape, Shape, Dim, True, False)]: ...
@overload
def ihfftn[Shape: IntTuple, Sizes: IntTuple, Dim: Flag[tuple[int, ...] | None]](
    input: Tensor[Shape],
    s: Sizes,
    dim: Dim = None,
    norm: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor[hermitian_fft_shape(Shape, Sizes, Dim, True, True)]: ...
def rfftfreq[N: IntVar](
    n: _Int[N],
    d: float = 1.0,
    *,
    out: Tensor | None = None,
    dtype: Any = None,
    layout: Any = ...,
    device: Any = None,
    requires_grad: bool = False,
) -> Tensor[[N // 2 + 1]]: ...
