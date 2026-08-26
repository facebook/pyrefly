# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from shape_extensions import assert_raises, assert_shape, Flag, Int, IntTuple, IntVar
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
    sequence = torch.randn((2, 3, 12))
    assert_shape(F.adaptive_avg_pool1d(sequence, 4).shape, (2, 3, 4))
    assert_shape(
        F.adaptive_avg_pool1d(input=sequence, output_size=4).shape,
        (2, 3, 4),
    )
    assert_shape(F.adaptive_max_pool1d(sequence, (5,)).shape, (2, 3, 5))

    tensor = torch.randn((2, 64, 56, 56))
    assert_shape(F.adaptive_avg_pool2d(tensor, (7, 7)).shape, (2, 64, 7, 7))
    assert_shape(F.adaptive_max_pool2d(tensor, (5, 7)).shape, (2, 64, 5, 7))
    assert_shape(
        F.adaptive_max_pool2d(tensor, (5, 7), return_indices=False).shape,
        (2, 64, 5, 7),
    )

    volume = torch.randn((2, 32, 12, 16, 20))
    assert_shape(
        F.adaptive_avg_pool3d(volume, (4, 7, 9)).shape,
        (2, 32, 4, 7, 9),
    )
    assert_shape(F.adaptive_avg_pool3d(volume, 4).shape, (2, 32, 4, 4, 4))
    values, indices = F.adaptive_max_pool3d(volume, (4, 7, 9), return_indices=True)
    assert_shape(values.shape, (2, 32, 4, 7, 9))
    assert_shape(indices.shape, (2, 32, 4, 7, 9))

    unbatched = torch.randn((3, 12, 15))
    assert_shape(F.adaptive_avg_pool2d(unbatched, 4).shape, (3, 4, 4))
    unbatched_volume = torch.randn((3, 8, 12, 15))
    assert_shape(
        F.adaptive_max_pool3d(unbatched_volume, (4, 5, 6)).shape,
        (3, 4, 5, 6),
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


def test_pooling_modules_reject_invalid_rank_and_controls() -> None:
    with assert_raises(RuntimeError):
        nn.MaxPool1d(2)(torch.randn((8,)))  # E: pooling requires spatial rank
    with assert_raises(RuntimeError):
        # E: pooling requires spatial rank
        nn.MaxPool2d(2)(torch.randn((2, 3, 4, 4, 4)))
    with assert_raises(RuntimeError):
        nn.MaxPool3d(2)(torch.randn((2, 3, 4)))  # E: pooling requires spatial rank
    with assert_raises(RuntimeError):
        nn.AvgPool1d(2)(torch.randn((2, 3, 4, 4)))  # E: pooling requires spatial rank
    # PyTorch indexes the missing channel dimension before checking the rank.
    with assert_raises(IndexError):
        nn.AvgPool2d(2)(torch.randn((8, 8)))  # E: pooling requires spatial rank
    with assert_raises(RuntimeError):
        # E: pooling requires spatial rank
        nn.AvgPool3d(2)(torch.randn((2, 3, 4, 4, 4, 4)))

    image = torch.randn((2, 3, 8, 8))
    with assert_raises(RuntimeError):
        nn.MaxPool2d(0)(image)  # E: pooling kernel must be positive
    with assert_raises(RuntimeError):
        nn.MaxPool2d(2, stride=0)(image)  # E: pooling stride must be positive
    with assert_raises(RuntimeError):
        nn.MaxPool2d(2, padding=-1)(image)  # E: pooling padding must be nonnegative
    with assert_raises(RuntimeError):
        nn.MaxPool2d(2, dilation=0)(image)  # E: pooling dilation must be positive
    with assert_raises(RuntimeError):
        # E: pooling padding must be at most half the kernel size
        nn.MaxPool2d(2, padding=2)(image)
    with assert_raises(RuntimeError):
        nn.AvgPool2d(0)(image)  # E: pooling kernel must be positive
    with assert_raises(RuntimeError):
        nn.AvgPool2d(2, stride=0)(image)  # E: pooling stride must be positive
    with assert_raises(RuntimeError):
        nn.AvgPool2d(2, padding=-1)(image)  # E: pooling padding must be nonnegative
    with assert_raises(RuntimeError):
        # E: pooling padding must be at most half the kernel size
        nn.AvgPool2d(2, padding=2)(image)

    # Padding is checked before the ceil-mode final-window correction, whose
    # divisor would otherwise be zero for a window that padding alone can fill.
    sequence = torch.randn((2, 3, 4))
    with assert_raises(RuntimeError):
        # E: pooling padding must be at most half the kernel size
        nn.MaxPool1d(2, stride=2, padding=2, ceil_mode=True)(sequence)
    with assert_raises(RuntimeError):
        # E: pooling padding must be at most half the kernel size
        nn.AvgPool1d(2, stride=2, padding=2, ceil_mode=True)(sequence)


def test_functional_pooling_rejects_control_tuple_rank_mismatches() -> None:
    image = torch.randn((2, 3, 8, 8))
    with assert_raises(RuntimeError):
        F.max_pool2d(image, (2, 2, 2))  # E: No matching overload
    with assert_raises(RuntimeError):
        F.max_pool2d(image, 2, (2, 2, 2))  # E: No matching overload
    with assert_raises(RuntimeError):
        # E: No matching overload
        F.max_pool2d(image, 2, None, (0, 0, 0))
    with assert_raises(RuntimeError):
        # E: No matching overload
        F.max_pool2d(image, 2, None, 0, (1, 1, 1))


def test_pooling_module_tuple_controls() -> None:
    image = torch.randn((2, 3, 8, 8))
    # TODO: BUG: Accept tuple-valued pooling module controls statically.
    max_pool = nn.MaxPool2d((2, 2))  # E: is not a valid `Flag[int]` value
    avg_pool = nn.AvgPool2d((2, 2))  # E: is not a valid `Flag[int]` value
    assert_shape(max_pool(image).shape, (2, 3, 4, 4))
    assert_shape(avg_pool(image).shape, (2, 3, 4, 4))


def test_pooling_rejects_nonpositive_output_extent() -> None:
    tensor = torch.randn((2, 3, 2))
    assert_shape(tensor.shape, (2, 3, 2))
    with assert_raises(RuntimeError):
        # E: pooling output extent must be positive
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
    from torch._shapes import pool_shape

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

    image = torch.randn((2, 3, 8, 8))
    pool2d(image, (2, 2, 2))  # E: pooling kernel must match the spatial rank
    pool2d(image, 2, (2, 2, 2))  # E: pooling stride must match the spatial rank
    pool2d(image, 2, None, (0, 0, 0))  # E: pooling padding must match the spatial rank
    # E: pooling dilation must match the spatial rank
    pool2d(image, 2, None, 0, (1, 1, 1))

    def check_adaptive_symbolic[B: IntVar, H: IntVar, W: IntVar, D: IntVar](
        sequence: Tensor[[B, 32, 12]],
        tensor: Tensor[[B, 64, 56, 56]],
        volume: Tensor[[B, 32, 12, 16, 20]],
        height: Int[H],
        width: Int[W],
        depth: Int[D],
    ) -> None:
        assert_type(F.adaptive_avg_pool1d(sequence, 4), Tensor[[B, 32, 4]])
        assert_type(F.adaptive_max_pool1d(sequence, (5,)), Tensor[[B, 32, 5]])
        assert_type(
            F.adaptive_max_pool1d(sequence, 5, return_indices=True),
            tuple[Tensor[[B, 32, 5]], Tensor[[B, 32, 5]]],
        )
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
            F.adaptive_max_pool2d(tensor, (height, 5), return_indices=True),
            tuple[Tensor[[B, 64, H, 5]], Tensor[[B, 64, H, 5]]],
        )
        assert_type(
            F.adaptive_avg_pool3d(volume, (depth, 7, width)),
            Tensor[[B, 32, D, 7, W]],
        )
        assert_type(
            F.adaptive_max_pool3d(volume, (depth, 7, width), return_indices=True),
            tuple[Tensor[[B, 32, D, 7, W]], Tensor[[B, 32, D, 7, W]]],
        )

    def check_adaptive_fallbacks(
        tensor: Tensor[[2, 3, 12, 15]],
        volume: Tensor[[2, 3, 8, 12, 15]],
        open_rank: Tensor[IntTuple],
        output_size: int,
        output_pair: tuple[int, int],
        return_indices: bool,
    ) -> None:
        assert_type(
            F.adaptive_avg_pool2d(tensor, output_size), Tensor[[2, 3, int, int]]
        )
        assert_type(
            F.adaptive_avg_pool2d(tensor, output_pair), Tensor[[2, 3, int, int]]
        )
        assert_type(F.adaptive_avg_pool2d(open_rank, (4, 5)), Tensor[IntTuple])
        assert_type(F.adaptive_avg_pool2d(tensor, (None, 5)), Tensor)
        assert_type(
            F.adaptive_max_pool2d(tensor, (4, None), return_indices=True),
            tuple[Tensor, Tensor],
        )
        assert_type(F.adaptive_avg_pool3d(volume, (None, 5, None)), Tensor)
        assert_type(F.adaptive_max_pool3d(volume, (4, None, 6)), Tensor)
        assert_type(
            F.adaptive_max_pool2d(tensor, (4, 5), return_indices=return_indices),
            Tensor | tuple[Tensor, Tensor],
        )

    def check_undecidable_pool_arguments(
        tensor: Tensor[[2, 3, 8, 12]],
        unknown_arity: tuple[int, ...],
        unknown_elements: tuple[int, int],
    ) -> None:
        assert_type(F.max_pool2d(tensor, unknown_arity), Tensor)
        assert_type(F.max_pool2d(tensor, unknown_elements), Tensor)

    def check_gradual_adaptive_output[B: IntVar](
        tensor: Tensor[[B, 64, 56, 56]], output_size: int
    ) -> None:
        assert_type(
            F.adaptive_avg_pool2d(tensor, output_size),
            Tensor[[B, 64, int, int]],
        )
        assert_type(
            F.adaptive_max_pool2d(tensor, (output_size, 7)),
            Tensor[[B, 64, int, 7]],
        )

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

    def check_unbounded_adaptive_size[Value, Text: str](
        tensor: Tensor[[2, 64, 56, 56]], value: Value, text: Text
    ) -> None:
        F.adaptive_avg_pool2d(tensor, value)  # E: No matching overload
        F.adaptive_max_pool2d(tensor, text)  # E: No matching overload
        F.adaptive_avg_pool2d(tensor, (value, value))  # E: No matching overload
        F.adaptive_avg_pool3d(tensor, (value, 7, value))  # E: No matching overload
        F.adaptive_max_pool1d(tensor, (value,))  # E: No matching overload
