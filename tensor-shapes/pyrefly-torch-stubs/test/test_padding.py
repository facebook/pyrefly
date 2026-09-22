# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from shape_extensions import assert_raises, assert_shape, IntTuple, IntVar
from torch import Tensor


def test_pad_shapes() -> None:
    vector = torch.randn(10)
    assert_shape(F.pad(vector, (2, 3)).shape, (15,))
    assert_shape(F.pad(input=vector, pad=()).shape, (10,))

    matrix = torch.randn((3, 4))
    assert_shape(F.pad(matrix, (1, 2, 3, 4)).shape, (10, 7))
    assert_shape(F.pad(matrix, (1, 1), mode="reflect").shape, (3, 6))

    tensor = torch.randn((2, 3, 5))
    assert_shape(F.pad(tensor, (1, 2, 3, 4)).shape, (2, 10, 8))
    assert_shape(F.pad(tensor, (-1, -2, 3, -1)).shape, (2, 5, 2))

    scalar = torch.randn(())
    assert_shape(F.pad(scalar, ()).shape, ())


def test_pad_rejects_invalid_shapes() -> None:
    matrix = torch.randn((2, 3))
    assert_shape(F.pad(matrix, (1, 2)).shape, (2, 6))

    with assert_raises(TypeError):
        F.pad(matrix, (1, 2.0))  # E: No matching overload

    with assert_raises(RuntimeError):
        F.pad(matrix, (1, 2, 3))  # E: pad must have an even number of entries

    with assert_raises(RuntimeError):
        # E: pad has more padding pairs than input dimensions
        F.pad(matrix, (1, 2, 3, 4, 5, 6))

    scalar = torch.randn(())
    with assert_raises(RuntimeError):
        F.pad(scalar, (1, 2))  # E: pad does not support scalar input


def test_pad_runtime_discrepancies() -> None:
    matrix = torch.randn((2, 3))

    assert_shape(
        F.pad(matrix, (1, True)).shape,
        IntTuple,
        runtime=(2, 5),
    )

    with assert_raises(RuntimeError):
        # E: pad cannot produce a negative dimension
        F.pad(torch.randn(3), (-4, 0))

    # TODO: BUG: Reject `value` for non-constant padding modes statically.
    with assert_raises(RuntimeError):
        F.pad(matrix, (1, 1), mode="reflect", value=2.5)


def test_padding_modules() -> None:
    image = torch.randn((2, 3, 4, 5))
    assert_shape(nn.ReflectionPad2d(1)(image).shape, (2, 3, 6, 7))
    assert_shape(nn.ReplicationPad2d(1)(image).shape, (2, 3, 6, 7))

    with assert_raises(NotImplementedError):
        # E: 2D padding requires 3D or 4D input
        nn.ReflectionPad2d(1)(torch.randn((4, 4)))
    with assert_raises(NotImplementedError):
        # E: 2D padding requires 3D or 4D input
        nn.ReflectionPad2d(1)(torch.randn((2, 3, 4, 4, 4)))
    with assert_raises(NotImplementedError):
        # E: 2D padding requires 3D or 4D input
        nn.ReplicationPad2d(1)(torch.randn((4, 4)))
    with assert_raises(NotImplementedError):
        # E: 2D padding requires 3D or 4D input
        nn.ReplicationPad2d(1)(torch.randn((2, 3, 4, 4, 4)))


if TYPE_CHECKING:

    def check_symbolic_pad[N: IntVar](tensor: Tensor[[N, 5]]) -> None:
        assert_type(F.pad(tensor, (1, 2, 3, 4)), Tensor[[N + 7, 8]])

    def check_gradual_pad_inputs(
        open_rank: Tensor[IntTuple],
        matrix: Tensor[[2, 3]],
        dynamic_pad: tuple[int, ...],
        dynamic_pair: tuple[int, int],
        dynamic_list: list[int],
        amount: int,
    ) -> None:
        assert_type(F.pad(open_rank, (1, 2)), Tensor[IntTuple])
        assert_type(F.pad(matrix, dynamic_pad), Tensor[IntTuple])
        assert_type(F.pad(matrix, dynamic_pair), Tensor[IntTuple])
        assert_type(F.pad(matrix, dynamic_list), Tensor[IntTuple])
        assert_type(F.pad(matrix, [1, 2]), Tensor[IntTuple])
        assert_type(F.pad(matrix, [amount, amount]), Tensor[IntTuple])
