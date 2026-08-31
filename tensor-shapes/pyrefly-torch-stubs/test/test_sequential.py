# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import TYPE_CHECKING, assert_type

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from torch import Tensor


# Test 1: Sequential with only typed-stub modules (Conv+BN+ReLU)
def test_typed_stubs_only():
    seq = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3),
        nn.BatchNorm2d(64),
        nn.ReLU(),
    )
    x: Tensor[[2, 3, 128, 128]] = torch.randn(2, 3, 128, 128)
    out = seq(x)
    assert_type(out, Tensor[[2, 64, 128, 128]])


# Test 2: Sequential with ONLY a DSL module (ReflectionPad2d)
def test_pad_only():
    seq = nn.Sequential(nn.ReflectionPad2d(1))
    x: Tensor[[2, 64, 32, 32]] = torch.randn(2, 64, 32, 32)
    out = seq(x)
    assert_type(out, Tensor[[2, 64, 34, 34]])


# Test 3: DSL module called directly (not in Sequential) for comparison
def test_pad_direct():
    pad = nn.ReflectionPad2d(1)
    x: Tensor[[2, 64, 32, 32]] = torch.randn(2, 64, 32, 32)
    out = pad(x)
    assert_type(out, Tensor[[2, 64, 34, 34]])


# Test 4: Sequential with DSL module first, then typed-stub module
def test_pad_then_conv():
    seq = nn.Sequential(
        nn.ReflectionPad2d(1),
        nn.Conv2d(64, 64, kernel_size=3, padding=0),
    )
    x: Tensor[[2, 64, 32, 32]] = torch.randn(2, 64, 32, 32)
    out = seq(x)
    assert_type(out, Tensor[[2, 64, 32, 32]])


# Test 5: Sequential with ONLY a DSL module (Upsample)
def test_upsample_only():
    seq = nn.Sequential(nn.Upsample(scale_factor=2))
    x: Tensor[[2, 64, 32, 32]] = torch.randn(2, 64, 32, 32)
    out = seq(x)
    assert_type(out, Tensor[[2, 64, 64, 64]])


def test_upsample_size_in_sequential():
    seq = nn.Sequential(nn.Upsample(size=48, mode="bilinear", align_corners=False))
    x: Tensor[[2, 64, 32, 40]] = torch.randn(2, 64, 32, 40)
    assert_type(seq(x), Tensor[[2, 64, 48, 48]])


def test_gradual_upsample_composes_in_sequential():
    # Tuple sizes and float scales are valid but gradual, so composing them must
    # widen the pipeline rather than reject it or keep a stale precise shape.
    x: Tensor[[2, 64, 32, 40]] = torch.randn(2, 64, 32, 40)
    tuple_size = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.Upsample((64, 80)))
    float_scale = nn.Sequential(nn.Upsample(scale_factor=1.5), nn.ReLU())
    assert_type(tuple_size(x), Tensor)
    assert_type(float_scale(x), Tensor)


# Test 6: Upsample called directly for comparison
def test_upsample_direct():
    up = nn.Upsample(scale_factor=2)
    x: Tensor[[2, 64, 32, 32]] = torch.randn(2, 64, 32, 32)
    out = up(x)
    assert_type(out, Tensor[[2, 64, 64, 64]])


def test_maxpool_modules_in_sequential():
    one = nn.Sequential(nn.MaxPool1d(2))
    two = nn.Sequential(nn.MaxPool2d(2, stride=2))
    three = nn.Sequential(nn.MaxPool3d(2, padding=1))
    x1: Tensor[[2, 3, 16]] = torch.randn(2, 3, 16)
    x2: Tensor[[2, 3, 16, 20]] = torch.randn(2, 3, 16, 20)
    x3: Tensor[[2, 3, 8, 10, 12]] = torch.randn(2, 3, 8, 10, 12)
    assert_type(one(x1), Tensor[[2, 3, 8]])
    assert_type(two(x2), Tensor[[2, 3, 8, 10]])
    assert_type(three(x3), Tensor[[2, 3, 5, 6, 7]])


def test_avgpool_modules_in_sequential():
    one = nn.Sequential(nn.AvgPool1d(2))
    two = nn.Sequential(nn.AvgPool2d(2, stride=2))
    three = nn.Sequential(nn.AvgPool3d(2, padding=1))
    x1: Tensor[[2, 3, 16]] = torch.randn(2, 3, 16)
    x2: Tensor[[2, 3, 16, 20]] = torch.randn(2, 3, 16, 20)
    x3: Tensor[[2, 3, 8, 10, 12]] = torch.randn(2, 3, 8, 10, 12)
    assert_type(one(x1), Tensor[[2, 3, 8]])
    assert_type(two(x2), Tensor[[2, 3, 8, 10]])
    assert_type(three(x3), Tensor[[2, 3, 5, 6, 7]])


# Test 7: Sequential with only typed-stub module (Conv2d alone)
def test_conv_only():
    seq = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1))
    x: Tensor[[2, 64, 32, 32]] = torch.randn(2, 64, 32, 32)
    out = seq(x)
    assert_type(out, Tensor[[2, 128, 32, 32]])
