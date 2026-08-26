# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch
import torch.nn as nn
from shape_extensions import assert_raises, assert_shape


def test_pixel_shuffle_shapes() -> None:
    assert_shape(nn.PixelShuffle(2)(torch.randn((2, 12, 4, 5))).shape, (2, 3, 8, 10))


def test_pixel_shuffle_rejects_invalid_shapes() -> None:
    with assert_raises(RuntimeError):
        # E: PixelShuffle requires at least 3D input
        nn.PixelShuffle(2)(torch.randn((8, 4)))
    with assert_raises(RuntimeError):
        # E: PixelShuffle upscale_factor must be positive
        nn.PixelShuffle(0)(torch.randn((2, 10, 4, 4)))
    with assert_raises(RuntimeError):
        # E: PixelShuffle input channels must be divisible
        nn.PixelShuffle(3)(torch.randn((2, 10, 4, 4)))


def test_glu_shapes() -> None:
    assert_shape(nn.GLU(1)(torch.randn((2, 6, 4))).shape, (2, 3, 4))


def test_glu_rejects_invalid_shapes() -> None:
    tensor = torch.randn((2, 5, 4))
    with assert_raises(IndexError):
        nn.GLU(3)(tensor)  # E: GLU dimension out of range
    with assert_raises(RuntimeError):
        nn.GLU(1)(tensor)  # E: GLU input dimension must be even
