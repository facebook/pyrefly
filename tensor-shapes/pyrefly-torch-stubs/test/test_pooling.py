# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from shape_extensions import assert_raises, assert_shape, Flag, Int, IntVar
from torch import Tensor


def test_functional_pooling() -> None:
    tensor = torch.randn((2, 3, 8, 9))
    assert_shape(F.max_pool2d(tensor, (2, 3)).shape, (2, 3, 4, 3))
    assert_shape(F.max_pool2d(tensor, (2, 3), (2, 1)).shape, (2, 3, 4, 7))
    assert_shape(F.max_pool2d(tensor, 2, None, (1, 0)).shape, (2, 3, 5, 4))
    assert_shape(
        F.max_pool2d(tensor, (2, 2), 2, 0, (2, 1)).shape,
        (2, 3, 3, 4),
    )


def test_pooling_modules() -> None:
    tensor = torch.randn((2, 3, 8, 9))
    assert_shape(nn.MaxPool2d(2)(tensor).shape, (2, 3, 4, 4))
    assert_shape(nn.AvgPool2d(2)(tensor).shape, (2, 3, 4, 4))


def test_adaptive_pooling() -> None:
    tensor = torch.randn((2, 64, 56, 56))
    assert_shape(F.adaptive_avg_pool2d(tensor, (7, 7)).shape, (2, 64, 7, 7))
    assert_shape(F.adaptive_max_pool2d(tensor, (5, 7)).shape, (2, 64, 5, 7))

    volume = torch.randn((2, 32, 12, 16, 20))
    assert_shape(
        F.adaptive_avg_pool3d(volume, (4, 7, 9)).shape,
        (2, 32, 4, 7, 9),
    )


def test_pooling_rejects_invalid_rank_and_controls() -> None:
    image = torch.randn((2, 3, 8, 8))
    assert_shape(image.shape, (2, 3, 8, 8))

    with assert_raises(RuntimeError):
        # E: pooling requires spatial rank + 1 or + 2 input
        F.max_pool2d(torch.randn((3, 8)), 2)
    with assert_raises(RuntimeError):
        # E: pooling kernel must be positive
        F.max_pool2d(image, 0)
    with assert_raises(RuntimeError):
        # E: pooling stride must be positive
        F.max_pool2d(image, 2, stride=0)
    with assert_raises(RuntimeError):
        # E: pooling padding must be nonnegative
        F.avg_pool2d(image, 2, padding=-1)
    # TODO: BUG: Accept singleton pooling tuples as repeated per-axis controls.
    singleton_kernel = F.max_pool2d(image, (2,))  # E: No matching overload
    assert tuple(singleton_kernel.shape) == (2, 3, 4, 4)


def test_pooling_rejects_nonpositive_output_extent() -> None:
    tensor = torch.randn((2, 3, 2))
    assert_shape(tensor.shape, (2, 3, 2))
    with assert_raises(RuntimeError):
        # TODO: BUG: Reject nonpositive concrete pooling output extents statically.
        F.max_pool1d(tensor, 3)


def test_adaptive_pooling_rejects_invalid_arguments() -> None:
    image = torch.randn((2, 3, 8, 8))
    assert_shape(image.shape, (2, 3, 8, 8))

    with assert_raises(TypeError):
        # E: No matching overload
        F.adaptive_avg_pool2d(image, None)
    with assert_raises(RuntimeError):
        # E: No matching overload
        F.adaptive_avg_pool2d(image, (2,))
    with assert_raises(RuntimeError):
        # E: adaptive_pool2d requires 3D or 4D input
        F.adaptive_max_pool2d(torch.randn((8, 8)), 4)


if TYPE_CHECKING:

    def check_adaptive_symbolic[B: IntVar, H: IntVar, W: IntVar, D: IntVar](
        tensor: Tensor[[B, 64, 56, 56]],
        volume: Tensor[[B, 32, 12, 16, 20]],
        height: Int[H],
        width: Int[W],
        depth: Int[D],
    ) -> None:
        assert_type(F.adaptive_avg_pool2d(tensor, (7, 7)), Tensor[[B, 64, 7, 7]])
        assert_type(
            F.adaptive_avg_pool2d(tensor, (height, width)),
            Tensor[[B, 64, H, W]],
        )
        assert_type(
            F.adaptive_max_pool2d(tensor, (height, 5)),
            Tensor[[B, 64, H, 5]],
        )
        assert_type(
            F.adaptive_avg_pool3d(volume, (depth, 7, width)),
            Tensor[[B, 32, D, 7, W]],
        )

    def check_undecidable_pool_arguments(
        tensor: Tensor[[2, 3, 8, 12]],
        unknown_arity: tuple[int, ...],
        unknown_elements: tuple[int, int],
    ) -> None:
        assert_type(F.max_pool2d(tensor, unknown_arity), Tensor)
        assert_type(F.max_pool2d(tensor, unknown_elements), Tensor)

    def check_generic_pool_arguments[
        Kernel: Flag[int],
        Stride: Flag[int],
        Padding: Flag[int],
        Dilation: Flag[int],
    ](
        tensor: Tensor[[2, 3, 8]],
        kernel: Kernel,
        stride: Stride,
        padding: Padding,
        dilation: Dilation,
    ) -> None:
        assert_type(nn.MaxPool1d(kernel)(tensor), Tensor)
        assert_type(nn.MaxPool1d(2, stride)(tensor), Tensor)
        assert_type(nn.MaxPool1d(2, 2, padding)(tensor), Tensor)
        assert_type(nn.MaxPool1d(2, 2, 0, dilation)(tensor), Tensor)

    def check_symbolic_pooling[B: IntVar, C: IntVar, H: IntVar, W: IntVar](
        tensor: Tensor[[B, C, H, W]],
    ) -> None:
        once = nn.MaxPool2d(3, stride=2, ceil_mode=True)(tensor)
        assert_type(once, Tensor[[B, C, int, int]])
        assert_type(
            nn.MaxPool2d(3, stride=2, ceil_mode=True)(once),
            Tensor[[B, C, int, int]],
        )
