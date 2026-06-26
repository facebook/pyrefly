# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test Conv2d with tuple[int, int] kernel_size, stride, padding, dilation.

When tuple values are passed, the scalar type param (K, S, P, D) is unbound
and the spatial formula produces Unknown. Scalar inputs continue to bind
normally. This test verifies both paths work without errors.
"""

from typing import Any, assert_type, TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from torch import Tensor


def test_conv2d_scalar_kernel():
    """Scalar kernel_size binds K and produces exact spatial dims."""
    conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
    x: Tensor[1, 3, 32, 32] = torch.randn(1, 3, 32, 32)
    out = conv(x)
    assert_type(out, Tensor[1, 16, 32, 32])


def test_conv2d_tuple_kernel():
    """Tuple kernel_size leaves K unbound — channels preserved, spatial Unknown."""
    conv = nn.Conv2d(3, 16, kernel_size=(3, 5))
    x: Tensor[1, 3, 32, 32] = torch.randn(1, 3, 32, 32)
    out = conv(x)
    assert_type(out, Tensor[1, 16, Any, Any])


def test_conv2d_tuple_stride():
    """Tuple stride leaves S unbound — falls back to default S=1 in spatial formula."""
    conv = nn.Conv2d(3, 16, kernel_size=3, stride=(2, 1), padding=1)
    x: Tensor[1, 3, 64, 64] = torch.randn(1, 3, 64, 64)
    out = conv(x)
    assert_type(out, Tensor[1, 16, 64, 64])


def test_conv2d_string_padding():
    """String padding leaves P unbound — falls back to default P=0 in spatial formula."""
    conv = nn.Conv2d(3, 16, kernel_size=3, padding="same")
    x: Tensor[1, 3, 32, 32] = torch.randn(1, 3, 32, 32)
    out = conv(x)
    assert_type(out, Tensor[1, 16, 30, 30])
