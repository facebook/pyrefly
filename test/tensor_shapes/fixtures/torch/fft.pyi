# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Type stubs for torch.fft module (Phase 6: FFT Operations)
from torch import Tensor

# 1D FFT operations
def fft[*Shape](
    input: Tensor[*Shape], n: int = None, dim: int = -1, norm: str = None
) -> Tensor[*Shape]: ...
def ifft[*Shape](
    input: Tensor[*Shape], n: int = None, dim: int = -1, norm: str = None
) -> Tensor[*Shape]: ...
def rfft(self: Tensor, n: int = None, dim: int = -1, norm: str = None) -> Tensor: ...
def irfft(self: Tensor, n: int = None, dim: int = -1, norm: str = None) -> Tensor: ...
def hfft(self: Tensor, n: int = None, dim: int = -1, norm: str = None) -> Tensor: ...
def ihfft(self: Tensor, n: int = None, dim: int = -1, norm: str = None) -> Tensor: ...

# 2D FFT operations
def fft2[*Shape](
    input: Tensor[*Shape],
    s: tuple[int, int] = None,
    dim: tuple[int, int] = (-2, -1),
    norm: str = None,
) -> Tensor[*Shape]: ...
def ifft2[*Shape](
    input: Tensor[*Shape],
    s: tuple[int, int] = None,
    dim: tuple[int, int] = (-2, -1),
    norm: str = None,
) -> Tensor[*Shape]: ...
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
def fftn[*Shape](
    input: Tensor[*Shape],
    s: tuple[int, ...] = None,
    dim: tuple[int, ...] = None,
    norm: str = None,
) -> Tensor[*Shape]: ...
def ifftn[*Shape](
    input: Tensor[*Shape],
    s: tuple[int, ...] = None,
    dim: tuple[int, ...] = None,
    norm: str = None,
) -> Tensor[*Shape]: ...
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
def fftshift[*Shape](
    input: Tensor[*Shape], dim: int | tuple[int, ...] = None
) -> Tensor[*Shape]: ...
def ifftshift[*Shape](
    input: Tensor[*Shape], dim: int | tuple[int, ...] = None
) -> Tensor[*Shape]: ...
