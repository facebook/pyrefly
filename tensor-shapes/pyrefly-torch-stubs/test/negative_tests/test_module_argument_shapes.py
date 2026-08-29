# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from shape_extensions import Flag, IntTuple
from torch import Tensor

if TYPE_CHECKING:
    from torch._shapes import pool_shape


rank_two: Tensor[[8, 4]] = torch.randn(8, 4)
rank_one: Tensor[[8]] = torch.randn(8)
bad_channels: Tensor[[2, 10, 4, 4]] = torch.randn(2, 10, 4, 4)
rank_three: Tensor[[2, 3, 4]] = torch.randn(2, 3, 4)
rank_six: Tensor[[2, 3, 4, 4, 4, 4]] = torch.randn(2, 3, 4, 4, 4, 4)
glu_input: Tensor[[2, 5, 4]] = torch.randn(2, 5, 4)
pad_rank_two: Tensor[[4, 4]] = torch.randn(4, 4)
pad_rank_five: Tensor[[2, 3, 4, 4, 4]] = torch.randn(2, 3, 4, 4, 4)
half_pad_max = nn.MaxPool2d(2, padding=2)
half_pad_avg = nn.AvgPool2d(2, padding=2)
ceil_pad_max = nn.MaxPool1d(2, stride=2, padding=2, ceil_mode=True)
ceil_pad_avg = nn.AvgPool1d(2, stride=2, padding=2, ceil_mode=True)

nn.PixelShuffle(2)(rank_two)  # E: PixelShuffle requires at least 3D input
nn.PixelShuffle(0)(bad_channels)  # E: PixelShuffle upscale_factor must be positive
nn.PixelShuffle(3)(bad_channels)  # E: PixelShuffle input channels must be divisible
nn.GLU(3)(glu_input)  # E: GLU dimension out of range
nn.GLU(1)(glu_input)  # E: GLU input dimension must be even
nn.ReflectionPad2d(1)(pad_rank_two)  # E: 2D padding requires 3D or 4D input
nn.ReflectionPad2d(1)(pad_rank_five)  # E: 2D padding requires 3D or 4D input
nn.ReplicationPad2d(1)(pad_rank_two)  # E: 2D padding requires 3D or 4D input
nn.ReplicationPad2d(1)(pad_rank_five)  # E: 2D padding requires 3D or 4D input
nn.MaxPool1d(2)(rank_one)  # E: pooling requires spatial rank + 1 or + 2 input
nn.MaxPool1d(2)(bad_channels)  # E: pooling requires spatial rank + 1 or + 2 input
nn.MaxPool2d(2)(rank_two)  # E: pooling requires spatial rank + 1 or + 2 input
nn.MaxPool2d(2)(pad_rank_five)  # E: pooling requires spatial rank + 1 or + 2 input
nn.MaxPool3d(2)(rank_three)  # E: pooling requires spatial rank + 1 or + 2 input
nn.MaxPool3d(2)(rank_six)  # E: pooling requires spatial rank + 1 or + 2 input
nn.MaxPool2d(0)(bad_channels)  # E: pooling kernel must be positive
nn.MaxPool2d(2, stride=0)(bad_channels)  # E: pooling stride must be positive
nn.MaxPool2d(2, padding=-1)(bad_channels)  # E: pooling padding must be nonnegative
nn.MaxPool2d(2, dilation=0)(bad_channels)  # E: pooling dilation must be positive
half_pad_max(bad_channels)  # E: pooling padding must be at most half the kernel size
nn.AvgPool1d(2)(rank_one)  # E: pooling requires spatial rank + 1 or + 2 input
nn.AvgPool1d(2)(bad_channels)  # E: pooling requires spatial rank + 1 or + 2 input
nn.AvgPool2d(2)(rank_two)  # E: pooling requires spatial rank + 1 or + 2 input
nn.AvgPool2d(2)(pad_rank_five)  # E: pooling requires spatial rank + 1 or + 2 input
nn.AvgPool3d(2)(rank_three)  # E: pooling requires spatial rank + 1 or + 2 input
nn.AvgPool3d(2)(rank_six)  # E: pooling requires spatial rank + 1 or + 2 input
nn.AvgPool2d(0)(bad_channels)  # E: pooling kernel must be positive
nn.AvgPool2d(2, stride=0)(bad_channels)  # E: pooling stride must be positive
nn.AvgPool2d(2, padding=-1)(bad_channels)  # E: pooling padding must be nonnegative
half_pad_avg(bad_channels)  # E: pooling padding must be at most half the kernel size
# Padding is checked before the ceil-mode final-window correction, whose divisor
# would otherwise be zero for a window that padding alone can fill.
ceil_pad_max(rank_three)  # E: pooling padding must be at most half the kernel size
ceil_pad_avg(rank_three)  # E: pooling padding must be at most half the kernel size
# The module surfaces stay scalar-only; per-axis arguments go through `pool_shape`.
nn.MaxPool2d((2, 2))  # E: is not a valid `Flag[int]` value
nn.AvgPool2d((2, 2))  # E: is not a valid `Flag[int]` value


# A normalized argument tuple must have one entry per spatial axis. The pooling
# modules cannot spell a tuple, so the arity rules are pinned on the shared
# helper directly.
def pool2d[
    Shape: IntTuple,
    KernelSize: Flag[int | tuple[int, ...]],
    Stride: Flag[int | tuple[int, ...] | None],
    Padding: Flag[int | tuple[int, ...]],
    Dilation: Flag[int | tuple[int, ...]],
](
    input: Tensor[Shape],
    kernel_size: KernelSize,
    stride: Stride = None,
    padding: Padding = 0,
    dilation: Dilation = 1,
) -> Tensor[pool_shape(Shape, 2, KernelSize, Stride, Padding, Dilation, False)]: ...


image: Tensor[[2, 3, 8, 8]] = torch.randn(2, 3, 8, 8)
pool2d(image, (2, 2, 2))  # E: pooling kernel must match the spatial rank
pool2d(image, 2, (2, 2, 2))  # E: pooling stride must match the spatial rank
pool2d(image, 2, None, (0, 0, 0))  # E: pooling padding must match the spatial rank
pool2d(image, 2, None, 0, (1, 1, 1))  # E: pooling dilation must match the spatial rank
