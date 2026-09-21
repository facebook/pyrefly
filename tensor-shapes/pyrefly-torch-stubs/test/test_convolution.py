# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from shape_extensions import assert_shape


def test_conv2d_scalar_controls() -> None:
    tensor = torch.randn((1, 3, 32, 32))
    assert_shape(
        nn.Conv2d(3, 16, kernel_size=3, padding=1)(tensor).shape,
        (1, 16, 32, 32),
    )


def test_conv2d_tuple_kernel() -> None:
    tensor = torch.randn((1, 3, 32, 32))
    output = nn.Conv2d(3, 16, kernel_size=(3, 5))(tensor)
    # TODO: BUG: Track per-axis convolution kernel sizes.
    assert_shape(output.shape, (1, 16, Any, Any), runtime=(1, 16, 30, 28))


def test_conv2d_tuple_stride() -> None:
    tensor = torch.randn((1, 3, 64, 64))
    output = nn.Conv2d(3, 16, kernel_size=3, stride=(2, 1), padding=1)(tensor)
    # TODO: BUG: Track per-axis convolution controls instead of using scalar defaults.
    assert_shape(output.shape, (1, 16, 64, 64), runtime=(1, 16, 32, 64))


def test_conv2d_string_padding() -> None:
    tensor = torch.randn((1, 3, 32, 32))
    output = nn.Conv2d(3, 16, kernel_size=3, padding="same")(tensor)
    # TODO: BUG: Model string padding instead of using the scalar default.
    assert_shape(output.shape, (1, 16, 30, 30), runtime=(1, 16, 32, 32))
