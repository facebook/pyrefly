# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Phase 3: Convolution & Pooling operations smoke tests
# Tests for CNN-critical operations: conv, conv_transpose, pooling, adaptive pooling, interpolate
from typing import assert_type

import torch
import torch.nn
import torch.nn.functional
from torch import Tensor

# ==== 1D Convolution Operations ====


def test_conv1d_basic():
    """Test 1D convolution with basic parameters"""
    input: Tensor[2, 3, 10] = torch.randn(2, 3, 10)  # (batch, in_channels, length)
    weight: Tensor[16, 3, 3] = torch.randn(
        16, 3, 3
    )  # (out_channels, in_channels, kernel)
    # Output: (batch, out_channels, length_out)
    # length_out = floor((10 + 2*0 - 1*(3-1) - 1) / 1 + 1) = floor((10 - 2 - 1) / 1 + 1) = 8
    result = torch.nn.functional.conv1d(input, weight)
    assert_type(result, Tensor[2, 16, 8])


def test_conv1d_stride_padding():
    """Test 1D convolution with stride and padding"""
    input: Tensor[1, 4, 20] = torch.randn(1, 4, 20)
    weight: Tensor[8, 4, 5] = torch.randn(8, 4, 5)
    # stride=2, padding=2
    # length_out = floor((20 + 2*2 - 1*(5-1) - 1) / 2 + 1) = floor((24 - 4 - 1) / 2 + 1) = 10
    result = torch.nn.functional.conv1d(input, weight, stride=2, padding=2)
    assert_type(result, Tensor[1, 8, 10])


# ==== 2D Convolution Operations ====


def test_conv2d_basic():
    """Test 2D convolution (most common for CNNs)"""
    input: Tensor[4, 3, 32, 32] = torch.randn(
        4, 3, 32, 32
    )  # (batch, channels, height, width)
    weight: Tensor[64, 3, 3, 3] = torch.randn(
        64, 3, 3, 3
    )  # (out_channels, in_channels, kH, kW)
    # Output: (batch, out_channels, H_out, W_out)
    # H_out = W_out = floor((32 + 2*0 - 1*(3-1) - 1) / 1 + 1) = 30
    result = torch.nn.functional.conv2d(input, weight)
    assert_type(result, Tensor[4, 64, 30, 30])


def test_conv2d_stride_padding():
    """Test 2D convolution with stride and padding"""
    input: Tensor[2, 16, 28, 28] = torch.randn(2, 16, 28, 28)
    weight: Tensor[32, 16, 5, 5] = torch.randn(32, 16, 5, 5)
    # stride=2, padding=2
    # H_out = W_out = floor((28 + 2*2 - 1*(5-1) - 1) / 2 + 1) = floor((32 - 4 - 1) / 2 + 1) = 14
    result = torch.nn.functional.conv2d(input, weight, stride=2, padding=2)
    assert_type(result, Tensor[2, 32, 14, 14])


def test_conv2d_dilation():
    """Test 2D convolution with dilation (dilated/atrous convolution)"""
    input: Tensor[1, 8, 24, 24] = torch.randn(1, 8, 24, 24)
    weight: Tensor[16, 8, 3, 3] = torch.randn(16, 8, 3, 3)
    # stride=1, padding=0, dilation=2
    # effective_kernel = 1 + dilation * (kernel - 1) = 1 + 2 * (3 - 1) = 5
    # H_out = W_out = floor((24 + 0 - 2*(3-1) - 1) / 1 + 1) = floor((24 - 4 - 1) / 1 + 1) = 20
    result = torch.nn.functional.conv2d(input, weight, stride=1, padding=0, dilation=2)
    assert_type(result, Tensor[1, 16, 20, 20])


# ==== 3D Convolution Operations ====


def test_conv3d_basic():
    """Test 3D convolution for video/volumetric data"""
    input: Tensor[1, 3, 16, 32, 32] = torch.randn(
        1, 3, 16, 32, 32
    )  # (batch, channels, depth, height, width)
    weight: Tensor[8, 3, 3, 3, 3] = torch.randn(
        8, 3, 3, 3, 3
    )  # (out_channels, in_channels, kD, kH, kW)
    # Output: (batch, out_channels, D_out, H_out, W_out)
    # D_out = H_out = W_out based on formula
    result = torch.nn.functional.conv3d(input, weight)
    assert_type(result, Tensor[1, 8, 14, 30, 30])


# ==== Transposed Convolution Operations (Deconvolution) ====


def test_conv_transpose1d():
    """Test 1D transposed convolution for upsampling"""
    input: Tensor[1, 16, 8] = torch.randn(1, 16, 8)
    weight: Tensor[16, 32, 3] = torch.randn(
        16, 32, 3
    )  # (in_channels, out_channels, kernel)
    # stride=2, padding=0
    # length_out = (8 - 1) * 2 - 2*0 + 1*(3-1) + 0 + 1 = 14 - 0 + 2 + 1 = 17
    result = torch.nn.functional.conv_transpose1d(input, weight, stride=2)
    assert_type(result, Tensor[1, 32, 17])


def test_conv_transpose2d():
    """Test 2D transposed convolution (common in GANs, autoencoders)"""
    input: Tensor[2, 64, 7, 7] = torch.randn(2, 64, 7, 7)
    weight: Tensor[64, 32, 4, 4] = torch.randn(64, 32, 4, 4)
    # stride=2, padding=1
    # H_out = W_out = (7 - 1) * 2 - 2*1 + 1*(4-1) + 0 + 1 = 12 - 2 + 3 + 1 = 14
    result = torch.nn.functional.conv_transpose2d(input, weight, stride=2, padding=1)
    assert_type(result, Tensor[2, 32, 14, 14])


def test_conv_transpose3d():
    """Test 3D transposed convolution"""
    input: Tensor[1, 32, 4, 8, 8] = torch.randn(1, 32, 4, 8, 8)
    weight: Tensor[32, 16, 4, 4, 4] = torch.randn(32, 16, 4, 4, 4)
    # stride=2, padding=1
    # D_out = H_out = W_out = (n - 1) * 2 - 2*1 + 1*(4-1) + 0 + 1
    result = torch.nn.functional.conv_transpose3d(input, weight, stride=2, padding=1)
    assert_type(result, Tensor[1, 16, 8, 16, 16])


# ==== Max Pooling Operations ====


def test_max_pool1d():
    """Test 1D max pooling"""
    input: Tensor[1, 16, 32] = torch.randn(1, 16, 32)
    # kernel_size=2, stride=2 (default = kernel_size)
    # length_out = floor((32 + 0 - 1*(2-1) - 1) / 2 + 1) = floor(30 / 2 + 1) = 16
    result = torch.nn.functional.max_pool1d(input, kernel_size=2)
    assert_type(result, Tensor[1, 16, 16])


def test_max_pool2d():
    """Test 2D max pooling (most common)"""
    input: Tensor[4, 64, 56, 56] = torch.randn(4, 64, 56, 56)
    # kernel_size=2, stride=2
    # H_out = W_out = floor((56 - 1*(2-1) - 1) / 2 + 1) = floor(54 / 2 + 1) = 28
    result = torch.nn.functional.max_pool2d(input, kernel_size=2, stride=2)
    assert_type(result, Tensor[4, 64, 28, 28])


def test_max_pool2d_padding():
    """Test 2D max pooling with padding"""
    input: Tensor[2, 32, 15, 15] = torch.randn(2, 32, 15, 15)
    # kernel_size=3, stride=2, padding=1
    # H_out = W_out = floor((15 + 2*1 - 1*(3-1) - 1) / 2 + 1) = floor((17 - 2 - 1) / 2 + 1) = 8
    result = torch.nn.functional.max_pool2d(input, kernel_size=3, stride=2, padding=1)
    assert_type(result, Tensor[2, 32, 8, 8])


def test_max_pool3d():
    """Test 3D max pooling for video data"""
    input: Tensor[1, 16, 8, 16, 16] = torch.randn(1, 16, 8, 16, 16)
    # kernel_size=2, stride=2
    result = torch.nn.functional.max_pool3d(input, kernel_size=2, stride=2)
    assert_type(result, Tensor[1, 16, 4, 8, 8])


# ==== Average Pooling Operations ====


def test_avg_pool1d():
    """Test 1D average pooling"""
    input: Tensor[1, 32, 64] = torch.randn(1, 32, 64)
    # kernel_size=4, stride=4
    result = torch.nn.functional.avg_pool1d(input, kernel_size=4, stride=4)
    assert_type(result, Tensor[1, 32, 16])


def test_avg_pool2d():
    """Test 2D average pooling"""
    input: Tensor[2, 128, 14, 14] = torch.randn(2, 128, 14, 14)
    # kernel_size=2, stride=2
    result = torch.nn.functional.avg_pool2d(input, kernel_size=2, stride=2)
    assert_type(result, Tensor[2, 128, 7, 7])


def test_avg_pool3d():
    """Test 3D average pooling"""
    input: Tensor[1, 64, 16, 32, 32] = torch.randn(1, 64, 16, 32, 32)
    # kernel_size=2, stride=2
    result = torch.nn.functional.avg_pool3d(input, kernel_size=2, stride=2)
    assert_type(result, Tensor[1, 64, 8, 16, 16])


# ==== Adaptive Max Pooling Operations ====


def test_adaptive_max_pool1d():
    """Test 1D adaptive max pooling"""
    input: Tensor[1, 16, 100] = torch.randn(1, 16, 100)
    # Output size directly specified
    result = torch.nn.functional.adaptive_max_pool1d(input, output_size=10)
    assert_type(result, Tensor[1, 16, 10])


def test_adaptive_max_pool2d():
    """Test 2D adaptive max pooling (used in classification networks)"""
    input: Tensor[4, 512, 7, 7] = torch.randn(4, 512, 7, 7)
    # Output size directly specified - common in ResNet, VGG
    result = torch.nn.functional.adaptive_max_pool2d(input, output_size=(1, 1))
    assert_type(result, Tensor[4, 512, 1, 1])


def test_adaptive_max_pool2d_variable():
    """Test 2D adaptive max pooling with variable input size"""
    input: Tensor[2, 256, 13, 17] = torch.randn(2, 256, 13, 17)
    # Adapt to fixed output size
    result = torch.nn.functional.adaptive_max_pool2d(input, output_size=(6, 6))
    assert_type(result, Tensor[2, 256, 6, 6])


def test_adaptive_max_pool3d():
    """Test 3D adaptive max pooling"""
    input: Tensor[1, 32, 10, 15, 20] = torch.randn(1, 32, 10, 15, 20)
    result = torch.nn.functional.adaptive_max_pool3d(input, output_size=(5, 5, 5))
    assert_type(result, Tensor[1, 32, 5, 5, 5])


# ==== Adaptive Average Pooling Operations ====


def test_adaptive_avg_pool1d():
    """Test 1D adaptive average pooling"""
    input: Tensor[2, 64, 50] = torch.randn(2, 64, 50)
    result = torch.nn.functional.adaptive_avg_pool1d(input, output_size=10)
    assert_type(result, Tensor[2, 64, 10])


def test_adaptive_avg_pool2d():
    """Test 2D adaptive average pooling (very common in modern architectures)"""
    input: Tensor[8, 2048, 7, 7] = torch.randn(8, 2048, 7, 7)
    # Global average pooling: 7x7 → 1x1
    result = torch.nn.functional.adaptive_avg_pool2d(input, output_size=(1, 1))
    assert_type(result, Tensor[8, 2048, 1, 1])


def test_adaptive_avg_pool2d_non_square():
    """Test 2D adaptive average pooling with non-square output"""
    input: Tensor[4, 128, 14, 21] = torch.randn(4, 128, 14, 21)
    result = torch.nn.functional.adaptive_avg_pool2d(input, output_size=(7, 7))
    assert_type(result, Tensor[4, 128, 7, 7])


def test_adaptive_avg_pool3d():
    """Test 3D adaptive average pooling"""
    input: Tensor[1, 16, 20, 40, 40] = torch.randn(1, 16, 20, 40, 40)
    result = torch.nn.functional.adaptive_avg_pool3d(input, output_size=(10, 10, 10))
    assert_type(result, Tensor[1, 16, 10, 10, 10])


# ==== Interpolation/Upsampling Operations ====


def test_interpolate_size_1d():
    """Test interpolation with size parameter (1D)"""
    input: Tensor[1, 32, 50] = torch.randn(1, 32, 50)
    # Upsample to size 100
    result = torch.nn.functional.interpolate(input, size=100)
    assert_type(result, Tensor[1, 32, 100])


def test_interpolate_size_2d():
    """Test interpolation with size parameter (2D) - common for upsampling in segmentation"""
    input: Tensor[2, 64, 16, 16] = torch.randn(2, 64, 16, 16)
    # Upsample to 32x32
    result = torch.nn.functional.interpolate(input, size=(32, 32))
    assert_type(result, Tensor[2, 64, 32, 32])


def test_interpolate_scale_factor_2d():
    """Test interpolation with scale_factor"""
    input: Tensor[4, 128, 14, 14] = torch.randn(4, 128, 14, 14)
    # 2x upsampling: 14x14 → 28x28
    result = torch.nn.functional.interpolate(input, scale_factor=2)
    assert_type(result, Tensor[4, 128, 28, 28])


def test_interpolate_size_3d():
    """Test interpolation for 3D data"""
    input: Tensor[1, 16, 8, 16, 16] = torch.randn(1, 16, 8, 16, 16)
    # Upsample to 16x32x32
    result = torch.nn.functional.interpolate(input, size=(16, 32, 32))
    assert_type(result, Tensor[1, 16, 16, 32, 32])


def test_upsample_size():
    """Test upsample (deprecated, uses interpolate internally)"""
    input: Tensor[2, 32, 10, 10] = torch.randn(2, 32, 10, 10)
    result = torch.nn.functional.upsample(input, size=(20, 20))
    assert_type(result, Tensor[2, 32, 20, 20])


def test_upsample_scale_factor():
    """Test upsample with scale_factor"""
    input: Tensor[1, 64, 7, 7] = torch.randn(1, 64, 7, 7)
    # 4x upsampling: 7x7 → 28x28
    result = torch.nn.functional.upsample(input, scale_factor=4)
    assert_type(result, Tensor[1, 64, 28, 28])


# ==== Comprehensive Tests (Multiple Operations) ====


def test_cnn_pipeline_conv_pool():
    """Test a typical CNN pipeline: conv → pool → conv → pool"""
    # Input image
    x: Tensor[1, 3, 64, 64] = torch.randn(1, 3, 64, 64)

    # First conv block
    w1: Tensor[32, 3, 3, 3] = torch.randn(32, 3, 3, 3)
    conv1 = torch.nn.functional.conv2d(x, w1, stride=1, padding=1)
    assert_type(conv1, Tensor[1, 32, 64, 64])

    # First pool
    pool1 = torch.nn.functional.max_pool2d(conv1, kernel_size=2, stride=2)
    assert_type(pool1, Tensor[1, 32, 32, 32])

    # Second conv block
    w2: Tensor[64, 32, 3, 3] = torch.randn(64, 32, 3, 3)
    conv2 = torch.nn.functional.conv2d(pool1, w2, stride=1, padding=1)
    assert_type(conv2, Tensor[1, 64, 32, 32])

    # Second pool
    pool2 = torch.nn.functional.max_pool2d(conv2, kernel_size=2, stride=2)
    assert_type(pool2, Tensor[1, 64, 16, 16])

    # Global average pooling
    gap = torch.nn.functional.adaptive_avg_pool2d(pool2, output_size=(1, 1))
    assert_type(gap, Tensor[1, 64, 1, 1])


def test_segmentation_pipeline():
    """Test encoder-decoder pipeline for segmentation: downsample → upsample"""
    # Encoder path
    x: Tensor[1, 3, 256, 256] = torch.randn(1, 3, 256, 256)

    w1: Tensor[64, 3, 3, 3] = torch.randn(64, 3, 3, 3)
    enc1 = torch.nn.functional.conv2d(x, w1, stride=2, padding=1)
    assert_type(enc1, Tensor[1, 64, 128, 128])

    w2: Tensor[128, 64, 3, 3] = torch.randn(128, 64, 3, 3)
    enc2 = torch.nn.functional.conv2d(enc1, w2, stride=2, padding=1)
    assert_type(enc2, Tensor[1, 128, 64, 64])

    # Decoder path (upsampling)
    up1 = torch.nn.functional.interpolate(enc2, size=(128, 128))
    assert_type(up1, Tensor[1, 128, 128, 128])

    w3: Tensor[128, 64, 3, 3] = torch.randn(128, 64, 3, 3)
    dec1 = torch.nn.functional.conv_transpose2d(
        enc2, w3, stride=2, padding=1, output_padding=1
    )
    assert_type(dec1, Tensor[1, 64, 128, 128])

    up2 = torch.nn.functional.interpolate(dec1, size=(256, 256))
    assert_type(up2, Tensor[1, 64, 256, 256])


# ==== Tier 3: max_pool with return_indices ====


def test_max_pool2d_return_indices():
    """Max pool with return_indices=True"""
    x: Tensor[1, 3, 32, 32] = torch.randn(1, 3, 32, 32)
    # return_indices=True returns tuple
    output, indices = torch.nn.functional.max_pool2d(
        x, kernel_size=2, return_indices=True
    )
    assert_type(output, Tensor[1, 3, 16, 16])
    assert_type(indices, Tensor[1, 3, 16, 16])


def test_max_pool2d_no_indices():
    """Max pool with return_indices=False (default)"""
    x: Tensor[1, 3, 32, 32] = torch.randn(1, 3, 32, 32)
    # return_indices=False returns single tensor
    output = torch.nn.functional.max_pool2d(x, kernel_size=2, return_indices=False)
    assert_type(output, Tensor[1, 3, 16, 16])


def test_max_pool1d_return_indices():
    """Max pool 1D with return_indices"""
    x: Tensor[2, 4, 10] = torch.randn(2, 4, 10)
    output, indices = torch.nn.functional.max_pool1d(
        x, kernel_size=2, return_indices=True
    )
    assert_type(output, Tensor[2, 4, 5])
    assert_type(indices, Tensor[2, 4, 5])
