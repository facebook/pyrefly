# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type, TYPE_CHECKING

import torch
from shape_extensions import assert_raises, assert_shape, Int, IntTuple, IntVar
from torch import Tensor


def test_unfold_shapes() -> None:
    assert_shape(torch.ones(8).unfold(0, 3, 1).shape, (6, 3))
    assert_shape(torch.ones((4, 6)).unfold(1, 2, 2).shape, (4, 3, 2))
    assert_shape(torch.ones((2, 5, 8)).unfold(2, 3, 1).shape, (2, 5, 6, 3))
    assert_shape(torch.ones((8, 5)).unfold(-2, 3, 2).shape, (3, 5, 3))


def test_unfold_zero_size() -> None:
    assert_shape(torch.ones(5).unfold(0, 0, 2).shape, (3, 0))
    assert_shape(torch.empty(0).unfold(0, 0, 2).shape, (1, 0))
    assert_shape(torch.tensor(1).unfold(0, 0, 2).shape, (0,))
    assert_shape(torch.tensor(1).unfold(-1, 1, 2).shape, (1,))


def test_unfold_rejects_invalid_dimensions() -> None:
    matrix = torch.ones((2, 3))
    assert_shape(matrix.unfold(0, 1, 1).shape, (2, 3, 1))

    with assert_raises(IndexError):
        matrix.unfold(-3, 1, 1)  # E: unfold dimension out of range

    scalar = torch.tensor(1)
    with assert_raises(IndexError):
        scalar.unfold(1, 0, 1)  # E: unfold dimension out of range


def test_unfold_rejects_invalid_sizes() -> None:
    matrix = torch.ones((2, 3))
    assert_shape(matrix.unfold(0, 1, 1).shape, (2, 3, 1))

    with assert_raises(RuntimeError):
        # E: unfold size must not exceed the selected dimension
        matrix.unfold(0, 3, 1)

    with assert_raises(RuntimeError):
        matrix.unfold(0, -1, 1)  # E: unfold size must be non-negative

    with assert_raises(RuntimeError):
        matrix.unfold(0, 1, 0)  # E: unfold step must be greater than zero


if TYPE_CHECKING:

    def check_symbolic[N: IntVar, M: IntVar](x: Tensor[[N]], size: Int[M]) -> None:
        assert_type(x.unfold(0, 3, 2), Tensor[[(N - 3) // 2 + 1, 3]])
        # Symbolic arguments cannot currently bind a `Flag` value.
        assert_type(x.unfold(0, size, 2), Tensor[IntTuple])

    def check_gradual(x: Tensor) -> None:
        assert_type(x.unfold(0, 3, 1), Tensor)
