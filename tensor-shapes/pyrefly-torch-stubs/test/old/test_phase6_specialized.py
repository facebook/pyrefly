# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Phase 6: Specialized operations tests (FFT, Loss, Padding, Random, Properties)
from typing import Any, assert_type, Literal

import torch
import torch.fft
import torch.nn
from shape_extensions import Int, IntVar
from torch import Tensor

# ==== FFT Operations ====


def test_fft_1d():
    """1D FFT"""
    x: Tensor[[10]] = torch.randn(10)
    result = torch.fft.fft(x)
    # Preserves shape
    assert_type(result, Tensor[[10]])


def test_ifft_1d():
    """1D inverse FFT"""
    x: Tensor[[8]] = torch.randn(8)
    result = torch.fft.ifft(x)
    # Preserves shape
    assert_type(result, Tensor[[8]])


def test_fft2_2d():
    """2D FFT"""
    x: Tensor[[4, 5]] = torch.randn(4, 5)
    result = torch.fft.fft2(x)
    # Preserves shape
    assert_type(result, Tensor[[4, 5]])


def test_fftn_3d():
    """ND FFT"""
    x: Tensor[[2, 3, 4]] = torch.randn(2, 3, 4)
    result = torch.fft.fftn(x)
    # Preserves shape
    assert_type(result, Tensor[[2, 3, 4]])


def test_rfft():
    """Real FFT (dimension changes)"""
    x: Tensor[[10]] = torch.randn(10)
    result = torch.fft.rfft(x)
    # Real FFT: [10] -> [6] (n//2 + 1 = 10//2 + 1 = 6)
    assert_type(result, Tensor[[6]])


def test_rfft_2d():
    """Real FFT on 2D tensor"""
    x: Tensor[[4, 8]] = torch.randn(4, 8)
    result = torch.fft.rfft(x, dim=1)
    # Real FFT along dim 1: [4, 8] -> [4, 5] (8//2 + 1 = 5)
    assert_type(result, Tensor[[4, 5]])


def test_irfft():
    """Inverse real FFT (dimension changes)"""
    x: Tensor[[6]] = torch.randn(6)
    result = torch.fft.irfft(x)
    # Inverse real FFT: [6] -> [10] (2*(n-1) = 2*(6-1) = 10)
    assert_type(result, Tensor[[10]])


def test_real_fft_axes_and_lengths():
    x: Tensor[[4, 10, 6]] = torch.randn(4, 10, 6)
    assert_type(torch.fft.rfft(x, dim=-2), Tensor[[4, 6, 6]])
    assert_type(torch.fft.rfft(x, n=None, dim=-2), Tensor[[4, 6, 6]])
    assert_type(torch.fft.rfft(x, n=8, dim=0), Tensor[[5, 10, 6]])
    assert_type(torch.fft.ihfft(x, n=8, dim=0), Tensor[[5, 10, 6]])
    assert_type(torch.fft.irfft(x, n=12, dim=1), Tensor[[4, 12, 6]])
    assert_type(torch.fft.hfft(x, n=12, dim=1), Tensor[[4, 12, 6]])
    assert_type(torch.fft.irfft(x, n=None, dim=1), Tensor[[4, 18, 6]])
    assert_type(torch.fft.hfft(x, dim=0), Tensor[[6, 10, 6]])
    assert_type(torch.fft.hfft(x, n=None, dim=0), Tensor[[6, 10, 6]])
    assert_type(torch.fft.ihfft(x, dim=1), Tensor[[4, 6, 6]])
    assert_type(torch.fft.ihfft(x, n=None, dim=1), Tensor[[4, 6, 6]])


def test_real_fft_symbolic_n[N: IntVar](x: Tensor[[3, 7]], n: Int[N]):
    assert_type(torch.fft.rfft(x, n=n, dim=0), Tensor[[N // 2 + 1, 7]])
    assert_type(torch.fft.ihfft(x, n=n, dim=0), Tensor[[N // 2 + 1, 7]])
    assert_type(torch.fft.irfft(x, n=n, dim=-1), Tensor[[3, N]])
    assert_type(torch.fft.hfft(x, n=n, dim=-1), Tensor[[3, N]])


def test_real_fft_known_shape_gradual_n(
    x: Tensor[[4, 10, 6]], n: int, optional_n: int | None
) -> None:
    # Forward arithmetic and inverse transforms both preserve gradual dimensions.
    assert_type(torch.fft.rfft(x, n=n, dim=0), Tensor[[int, 10, 6]])
    assert_type(torch.fft.ihfft(x, n=n, dim=0), Tensor[[int, 10, 6]])
    assert_type(torch.fft.irfft(x, n=n, dim=1), Tensor[[4, int, 6]])
    assert_type(torch.fft.hfft(x, n=n, dim=1), Tensor[[4, int, 6]])
    # TODO(stroxler): Preserve known axes by evaluating both branches for an optional value.
    assert_type(torch.fft.hfft(x, n=optional_n, dim=1), Tensor)
    assert_type(torch.fft.ihfft(x, n=optional_n, dim=0), Tensor)


def test_real_fft_gradual_dim(x: Tensor[[4, 10, 6]], dim: int) -> None:
    assert_type(torch.fft.rfft(x, dim=dim), Tensor)
    assert_type(torch.fft.irfft(x, n=12, dim=dim), Tensor)


def test_real_fft_any(x: Tensor[[4, 10, 6]], value: Any) -> None:
    assert_type(torch.fft.rfft(x, n=value), Tensor)
    assert_type(torch.fft.irfft(x, dim=value), Tensor)


def test_real_fft_gradual_input(x: Tensor, n: int, optional_n: int | None):
    assert_type(torch.fft.rfft(x), Tensor)
    assert_type(torch.fft.rfft(x, dim=0), Tensor)
    assert_type(torch.fft.rfft(x, n=None), Tensor)
    assert_type(torch.fft.irfft(x, n=n), Tensor)
    assert_type(torch.fft.hfft(x, n=optional_n), Tensor)


def test_fftshift():
    """FFT shift"""
    x: Tensor[[3, 4]] = torch.randn(3, 4)
    result = torch.fft.fftshift(x)
    # Preserves shape
    assert_type(result, Tensor[[3, 4]])


def test_ifftshift():
    """Inverse FFT shift"""
    x: Tensor[[5, 6]] = torch.randn(5, 6)
    result = torch.fft.ifftshift(x)
    # Preserves shape
    assert_type(result, Tensor[[5, 6]])


# ==== Random Sampling Operations ====


def test_bernoulli():
    """Bernoulli sampling"""
    x: Tensor[[3, 4]] = torch.randn(3, 4)
    result = torch.bernoulli(x)
    # Preserves shape
    assert_type(result, Tensor[[3, 4]])


def test_bernoulli_method():
    """Bernoulli sampling as method"""
    x: Tensor[[2, 5]] = torch.randn(2, 5)
    result = x.bernoulli()
    # Preserves shape
    assert_type(result, Tensor[[2, 5]])


def test_bernoulli_inplace():
    """Bernoulli sampling in-place"""
    x: Tensor[[4, 3]] = torch.randn(4, 3)
    result = x.bernoulli_()
    # Preserves shape
    assert_type(result, Tensor[[4, 3]])


def test_normal_inplace():
    """Normal distribution sampling in-place"""
    x: Tensor[[3, 4]] = torch.randn(3, 4)
    result = x.normal_()
    # Preserves shape
    assert_type(result, Tensor[[3, 4]])


def test_poisson():
    """Poisson sampling"""
    x: Tensor[[2, 3]] = torch.randn(2, 3)
    result = torch.poisson(x)
    # Preserves shape
    assert_type(result, Tensor[[2, 3]])


def test_random_inplace():
    """Random integers in-place"""
    x: Tensor[[3, 3]] = torch.randn(3, 3)
    result = x.random_()
    # Preserves shape
    assert_type(result, Tensor[[3, 3]])


def test_uniform_inplace():
    """Uniform distribution in-place"""
    x: Tensor[[4, 5]] = torch.randn(4, 5)
    result = x.uniform_()
    # Preserves shape
    assert_type(result, Tensor[[4, 5]])


# ==== Tensor Property Operations ====


def test_numel():
    """Number of elements"""
    x: Tensor[[3, 4, 5]] = torch.randn(3, 4, 5)
    result = torch.numel(x)
    # Returns int (symbolic multiplication of dimensions)
    assert_type(result, Literal[60])


# ==== Tier 3: torch.normal Overloads ====


def test_normal_tensor_tensor():
    """Normal with both tensor parameters"""
    mean: Tensor[[3, 4]] = torch.randn(3, 4)
    std: Tensor[[3, 4]] = torch.randn(3, 4)
    result = torch.normal(mean, std)
    assert_type(result, Tensor[[3, 4]])


def test_normal_tensor_tensor_mean_shape():
    mean: Tensor[[2, 3]] = torch.randn(2, 3)
    std: Tensor[[6]] = torch.randn(6)
    assert_type(torch.normal(mean, std), Tensor[[2, 3]])


def test_normal_tensor_scalar():
    """Normal with tensor mean, scalar std"""
    mean: Tensor[[2, 5]] = torch.randn(2, 5)
    result = torch.normal(mean, 0.5)
    assert_type(result, Tensor[[2, 5]])


def test_normal_scalar_tensor():
    """Normal with scalar mean, tensor std"""
    std: Tensor[[4, 3]] = torch.randn(4, 3)
    result = torch.normal(0.0, std)
    assert_type(result, Tensor[[4, 3]])


def test_normal_scalar_scalar_size():
    """Normal with scalar mean/std and size parameter"""
    result = torch.normal(0.0, 1.0, size=(3, 4))
    assert_type(result, Tensor[[3, 4]])


def test_normal_scalar_scalar_shape[N: IntVar](n: Int[N], plain: int):
    assert_type(torch.normal(0.0, 1.0, size=()), Tensor[[]])
    assert_type(torch.normal(0.0, 1.0, size=(n, plain)), Tensor[[N, int]])
