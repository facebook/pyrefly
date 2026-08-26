# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Sequence
from typing import assert_type, TYPE_CHECKING

import torch
from shape_extensions import (
    assert_raises,
    assert_shape,
    Elements,
    Int,
    IntTuple,
    IntVar,
)
from torch import Tensor


def test_reshape_and_view_shapes() -> None:
    tensor = torch.arange(24)
    assert_shape(tensor.reshape(2, 3, 4).shape, (2, 3, 4))
    assert_shape(tensor.view((6, 4)).shape, (6, 4))
    assert_shape(torch.reshape(tensor, (4, 6)).shape, (4, 6))
    assert_shape(tensor.reshape(2, -1).shape, (2, 12))
    assert_shape(tensor.view(-1).shape, (24,))

    other = torch.ones((3, 2, 4))
    assert_shape(tensor.reshape_as(other).shape, (3, 2, 4))


def test_reshape_scalars_and_empty_tensors() -> None:
    scalar = torch.tensor(1)
    assert_shape(scalar.view(()).shape, ())
    assert_shape(scalar.reshape(-1).shape, (1,))

    empty = torch.empty((0, 3))
    assert_shape(empty.reshape(-1).shape, (0,))
    assert_shape(empty.view((3, 0)).shape, (3, 0))
    assert_shape(torch.reshape(empty, (0, 1, 3)).shape, (0, 1, 3))


def test_reshape_sequence_arguments_are_gradual() -> None:
    tensor = torch.arange(6)
    dimensions: Sequence[int] = [2, 3]
    assert_shape(tensor.reshape(dimensions).shape, IntTuple, runtime=(2, 3))
    assert_shape(tensor.view(dimensions).shape, IntTuple, runtime=(2, 3))
    assert_shape(torch.reshape(tensor, dimensions).shape, IntTuple, runtime=(2, 3))


def test_reshape_rejects_invalid_targets() -> None:
    tensor = torch.arange(6)
    assert_shape(tensor.view(2, 3).shape, (2, 3))

    with assert_raises(RuntimeError):
        tensor.view(-1, -1)  # E: can only specify one unknown dimension as -1

    with assert_raises(RuntimeError):
        tensor.view(4, -1)  # E: could not infer size for dimension -1

    with assert_raises(RuntimeError):
        tensor.reshape(-2, 3)  # E: invalid negative dimension value

    with assert_raises(RuntimeError):
        tensor.reshape(4, 2)  # E: reshape target element count does not match

    with assert_raises(RuntimeError):
        torch.reshape(tensor, (2, 2))  # E: reshape target element count does not match

    empty = torch.empty((0, 3))
    with assert_raises(RuntimeError):
        empty.reshape(0, -1)  # E: could not infer size for dimension -1


if TYPE_CHECKING:

    def check_symbolic[N: IntVar, M: IntVar](
        tensor: Tensor[[N, M]], n: Int[N], m: Int[M]
    ) -> None:
        assert_type(tensor.view(-1), Tensor[[N * M]])
        assert_type(tensor.reshape(2, n // 2, m), Tensor[[2, N // 2, M]])

    def check_symbolic_inference[N: IntVar](tensor: Tensor[[N]]) -> None:
        assert_type(tensor.view(5, -1), Tensor[[5, N // 5]])
        assert_type(tensor.reshape(1, -1, 1), Tensor[[1, N, 1]])

    def check_symbolic_factoring[A: IntVar, B: IntVar, C: IntVar](
        matrix: Tensor[[2 * A - 1, B]], tensor: Tensor[[A, B, C]]
    ) -> None:
        extent = matrix.shape[0]
        assert_type(matrix.reshape(1, extent, -1), Tensor[[1, 2 * A - 1, B]])

        leading = tensor.shape[0] * tensor.shape[1]
        assert_type(tensor.reshape(leading, -1), Tensor[[A * B, C]])

    def check_variadic[Batch: IntTuple, C: IntVar](
        tensor: Tensor[[*Elements[Batch], C]], channels: Int[C]
    ) -> None:
        assert_type(tensor.reshape(-1, channels), Tensor[[int, C]])

    def check_starred_shape_slice[B: IntVar, T: IntVar, Heads: IntVar, Head: IntVar](
        tensor: Tensor[[B, T, Heads, Head]],
    ) -> None:
        prefix = tensor.size()[:-1]
        assert_type(prefix, tuple[Int[B], Int[T], Int[Heads]])
        assert_type(
            tensor.float().reshape(*prefix, -1, 2),
            Tensor[[B, T, Heads, Head // 2, 2]],
        )

    def check_gradual(
        tensor: Tensor[IntTuple], target: tuple[int, ...], other: Tensor[IntTuple]
    ) -> None:
        assert_type(tensor.reshape((2, 3)), Tensor)
        assert_type(tensor.view(target), Tensor)
        assert_type(tensor.reshape_as(other), Tensor[IntTuple])
