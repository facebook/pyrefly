# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import assert_type, TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from torch import Tensor
    from torch_shapes import Dim


# Test 1: Sequential with only typed-stub modules (Conv+BN+ReLU)
def test_typed_stubs_only():
    seq = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3),
        nn.BatchNorm2d(64),
        nn.ReLU(),
    )
    x: Tensor[2, 3, 128, 128] = torch.randn(2, 3, 128, 128)
    out = seq(x)
    assert_type(out, Tensor[2, 64, 128, 128])


# Test 2: Sequential with ONLY a DSL module (ReflectionPad2d)
def test_pad_only():
    seq = nn.Sequential(nn.ReflectionPad2d(1))
    x: Tensor[2, 64, 32, 32] = torch.randn(2, 64, 32, 32)
    out = seq(x)
    assert_type(out, Tensor[2, 64, 34, 34])


# Test 3: DSL module called directly (not in Sequential) for comparison
def test_pad_direct():
    pad = nn.ReflectionPad2d(1)
    x: Tensor[2, 64, 32, 32] = torch.randn(2, 64, 32, 32)
    out = pad(x)
    assert_type(out, Tensor[2, 64, 34, 34])


# Test 4: Sequential with DSL module first, then typed-stub module
def test_pad_then_conv():
    seq = nn.Sequential(
        nn.ReflectionPad2d(1),
        nn.Conv2d(64, 64, kernel_size=3, padding=0),
    )
    x: Tensor[2, 64, 32, 32] = torch.randn(2, 64, 32, 32)
    out = seq(x)
    assert_type(out, Tensor[2, 64, 32, 32])


# Test 5: Sequential with ONLY a DSL module (Upsample)
def test_upsample_only():
    seq = nn.Sequential(nn.Upsample(scale_factor=2))
    x: Tensor[2, 64, 32, 32] = torch.randn(2, 64, 32, 32)
    out = seq(x)
    assert_type(out, Tensor[2, 64, 64, 64])


# Test 6: Upsample called directly for comparison
def test_upsample_direct():
    up = nn.Upsample(scale_factor=2)
    x: Tensor[2, 64, 32, 32] = torch.randn(2, 64, 32, 32)
    out = up(x)
    assert_type(out, Tensor[2, 64, 64, 64])


# Test 7: Sequential with only typed-stub module (Conv2d alone)
def test_conv_only():
    seq = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1))
    x: Tensor[2, 64, 32, 32] = torch.randn(2, 64, 32, 32)
    out = seq(x)
    assert_type(out, Tensor[2, 128, 32, 32])
