# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Phase 6: Specialized operations tests (FFT, Loss, Padding, Random, Properties)
from typing import assert_type, Literal

import torch
import torch.fft
import torch.nn
import torch.nn.functional
from torch import Tensor

# ==== Loss Functions ====
# Note: Loss functions approximate shape behavior (default returns scalar)


def test_mse_loss_reduced():
    """MSE loss with default reduction"""
    input: Tensor[3, 4] = torch.randn(3, 4)
    target: Tensor[3, 4] = torch.randn(3, 4)
    result = torch.nn.functional.mse_loss(input, target)
    # Default reduction='mean' returns scalar
    assert_type(result, Tensor[()])


def test_l1_loss():
    """L1 loss"""
    input: Tensor[2, 5] = torch.randn(2, 5)
    target: Tensor[2, 5] = torch.randn(2, 5)
    result = torch.nn.functional.l1_loss(input, target)
    # Default reduction returns scalar
    assert_type(result, Tensor[()])


def test_cross_entropy():
    """Cross entropy loss"""
    input: Tensor[3, 10] = torch.randn(3, 10)  # 3 samples, 10 classes
    target: Tensor[3] = torch.randn(3)
    result = torch.nn.functional.cross_entropy(input, target)
    # Returns scalar
    assert_type(result, Tensor[()])


def test_binary_cross_entropy():
    """Binary cross entropy"""
    input: Tensor[4, 5] = torch.randn(4, 5)
    target: Tensor[4, 5] = torch.randn(4, 5)
    result = torch.nn.functional.binary_cross_entropy(input, target)
    # Returns scalar
    assert_type(result, Tensor[()])


def test_kl_div():
    """KL divergence"""
    input: Tensor[2, 3] = torch.randn(2, 3)
    target: Tensor[2, 3] = torch.randn(2, 3)
    result = torch.nn.functional.kl_div(input, target)
    # Returns scalar
    assert_type(result, Tensor[()])


def test_smooth_l1_loss():
    """Smooth L1 loss"""
    input: Tensor[3, 4] = torch.randn(3, 4)
    target: Tensor[3, 4] = torch.randn(3, 4)
    result = torch.nn.functional.smooth_l1_loss(input, target)
    # Returns scalar
    assert_type(result, Tensor[()])


def test_huber_loss():
    """Huber loss"""
    input: Tensor[2, 5] = torch.randn(2, 5)
    target: Tensor[2, 5] = torch.randn(2, 5)
    result = torch.nn.functional.huber_loss(input, target)
    # Returns scalar
    assert_type(result, Tensor[()])


# ==== Padding Operations ====
# Note: Simplified implementation - pad parameter handling is complex


def test_pad_1d():
    """Pad 1D tensor"""
    x: Tensor[10] = torch.randn(10)
    # Pad operations type check but shape inference needs pad parameter
    _ = torch.nn.functional.pad(x, (2, 3))


def test_pad_2d():
    """Pad 2D tensor"""
    x: Tensor[3, 4] = torch.randn(3, 4)
    _ = torch.nn.functional.pad(x, (1, 1, 2, 2))


def test_pad_3d():
    """Pad 3D tensor"""
    x: Tensor[2, 3, 4] = torch.randn(2, 3, 4)
    _ = torch.nn.functional.pad(x, (1, 1))


# ==== FFT Operations ====


def test_fft_1d():
    """1D FFT"""
    x: Tensor[10] = torch.randn(10)
    result = torch.fft.fft(x)
    # Preserves shape
    assert_type(result, Tensor[10])


def test_ifft_1d():
    """1D inverse FFT"""
    x: Tensor[8] = torch.randn(8)
    result = torch.fft.ifft(x)
    # Preserves shape
    assert_type(result, Tensor[8])


def test_fft2_2d():
    """2D FFT"""
    x: Tensor[4, 5] = torch.randn(4, 5)
    result = torch.fft.fft2(x)
    # Preserves shape
    assert_type(result, Tensor[4, 5])


def test_fftn_3d():
    """ND FFT"""
    x: Tensor[2, 3, 4] = torch.randn(2, 3, 4)
    result = torch.fft.fftn(x)
    # Preserves shape
    assert_type(result, Tensor[2, 3, 4])


def test_rfft():
    """Real FFT (dimension changes)"""
    x: Tensor[10] = torch.randn(10)
    result = torch.fft.rfft(x)
    # Real FFT: [10] -> [6] (n//2 + 1 = 10//2 + 1 = 6)
    assert_type(result, Tensor[6])


def test_rfft_2d():
    """Real FFT on 2D tensor"""
    x: Tensor[4, 8] = torch.randn(4, 8)
    result = torch.fft.rfft(x, dim=1)
    # Real FFT along dim 1: [4, 8] -> [4, 5] (8//2 + 1 = 5)
    assert_type(result, Tensor[4, 5])


def test_irfft():
    """Inverse real FFT (dimension changes)"""
    x: Tensor[6] = torch.randn(6)
    result = torch.fft.irfft(x)
    # Inverse real FFT: [6] -> [10] (2*(n-1) = 2*(6-1) = 10)
    assert_type(result, Tensor[10])


def test_fftshift():
    """FFT shift"""
    x: Tensor[3, 4] = torch.randn(3, 4)
    result = torch.fft.fftshift(x)
    # Preserves shape
    assert_type(result, Tensor[3, 4])


def test_ifftshift():
    """Inverse FFT shift"""
    x: Tensor[5, 6] = torch.randn(5, 6)
    result = torch.fft.ifftshift(x)
    # Preserves shape
    assert_type(result, Tensor[5, 6])


# ==== Random Sampling Operations ====


def test_bernoulli():
    """Bernoulli sampling"""
    x: Tensor[3, 4] = torch.randn(3, 4)
    result = torch.bernoulli(x)
    # Preserves shape
    assert_type(result, Tensor[3, 4])


def test_bernoulli_method():
    """Bernoulli sampling as method"""
    x: Tensor[2, 5] = torch.randn(2, 5)
    result = x.bernoulli()
    # Preserves shape
    assert_type(result, Tensor[2, 5])


def test_bernoulli_inplace():
    """Bernoulli sampling in-place"""
    x: Tensor[4, 3] = torch.randn(4, 3)
    result = x.bernoulli_()
    # Preserves shape
    assert_type(result, Tensor[4, 3])


def test_multinomial_1d():
    """Multinomial sampling from 1D"""
    x: Tensor[5] = torch.randn(5)
    # Note: num_samples is positional, meta-shape may not receive it as kwarg
    _ = torch.multinomial(x, 3)


def test_multinomial_2d():
    """Multinomial sampling from 2D"""
    x: Tensor[4, 5] = torch.randn(4, 5)
    _ = torch.multinomial(x, 3)


def test_multinomial_method():
    """Multinomial as method"""
    x: Tensor[3, 10] = torch.randn(3, 10)
    _ = x.multinomial(5)


def test_normal_inplace():
    """Normal distribution sampling in-place"""
    x: Tensor[3, 4] = torch.randn(3, 4)
    result = x.normal_()
    # Preserves shape
    assert_type(result, Tensor[3, 4])


def test_poisson():
    """Poisson sampling"""
    x: Tensor[2, 3] = torch.randn(2, 3)
    result = torch.poisson(x)
    # Preserves shape
    assert_type(result, Tensor[2, 3])


def test_random_inplace():
    """Random integers in-place"""
    x: Tensor[3, 3] = torch.randn(3, 3)
    result = x.random_()
    # Preserves shape
    assert_type(result, Tensor[3, 3])


def test_uniform_inplace():
    """Uniform distribution in-place"""
    x: Tensor[4, 5] = torch.randn(4, 5)
    result = x.uniform_()
    # Preserves shape
    assert_type(result, Tensor[4, 5])


# ==== Tensor Property Operations ====


def test_numel():
    """Number of elements"""
    x: Tensor[3, 4, 5] = torch.randn(3, 4, 5)
    result = torch.numel(x)
    # Returns int (symbolic multiplication of dimensions)
    assert_type(result, Literal[60])


# ==== Tier 3: torch.normal Overloads ====


def test_normal_tensor_tensor():
    """Normal with both tensor parameters"""
    mean: Tensor[3, 4] = torch.randn(3, 4)
    std: Tensor[3, 4] = torch.randn(3, 4)
    result = torch.normal(mean, std)
    assert_type(result, Tensor[3, 4])


def test_normal_tensor_scalar():
    """Normal with tensor mean, scalar std"""
    mean: Tensor[2, 5] = torch.randn(2, 5)
    result = torch.normal(mean, 0.5)
    assert_type(result, Tensor[2, 5])


def test_normal_scalar_tensor():
    """Normal with scalar mean, tensor std"""
    std: Tensor[4, 3] = torch.randn(4, 3)
    result = torch.normal(0.0, std)
    assert_type(result, Tensor[4, 3])


def test_normal_scalar_scalar_size():
    """Normal with scalar mean/std and size parameter"""
    result = torch.normal(0.0, 1.0, size=(3, 4))
    assert_type(result, Tensor[3, 4])
