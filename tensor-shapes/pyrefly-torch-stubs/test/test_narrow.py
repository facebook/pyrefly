# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

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


def test_narrow_shapes() -> None:
    tensor = torch.ones((5, 4))
    assert_shape(torch.narrow(tensor, dim=0, start=1, length=3).shape, (3, 4))
    assert_shape(tensor.narrow(dim=-1, start=0, length=2).shape, (5, 2))
    assert_shape(torch.narrow(tensor, dim=0, start=-2, length=2).shape, (2, 4))
    assert_shape(tensor.narrow(dim=0, start=5, length=0).shape, (0, 4))


def test_narrow_rejects_invalid_dimensions() -> None:
    matrix = torch.ones((2, 3))
    assert_shape(matrix.narrow(0, 0, 1).shape, (1, 3))

    with assert_raises(IndexError):
        torch.narrow(matrix, 2, 0, 1)  # E: dimension out of range

    with assert_raises(IndexError):
        matrix.narrow(-3, 0, 1)  # E: dimension out of range

    scalar = torch.tensor(1)
    with assert_raises(RuntimeError):
        scalar.narrow(0, 0, 1)  # E: dimension out of range


def test_narrow_rejects_invalid_bounds() -> None:
    vector = torch.ones(5)
    assert_shape(vector.narrow(0, 4, 1).shape, (1,))

    with assert_raises(RuntimeError):
        vector.narrow(0, 0, -1)  # E: narrow length must be non-negative

    with assert_raises(IndexError):
        torch.narrow(vector, 0, 6, 0)  # E: narrow start out of range

    with assert_raises(IndexError):
        vector.narrow(0, -6, 0)  # E: narrow start out of range

    with assert_raises(RuntimeError):
        # E: narrow start and length exceed dimension size
        vector.narrow(0, 4, 2)


if TYPE_CHECKING:

    def check_symbolic[N: IntVar, M: IntVar](x: Tensor[[N, M]]) -> None:
        assert_type(torch.narrow(x, dim=1, start=5, length=10), Tensor[[N, 10]])

    def check_symbolic_suffix[Shape: IntTuple, K: IntVar](
        x: Tensor[[*Elements[Shape], 3]], length: Int[K]
    ) -> None:
        assert_type(x.narrow(-1, 0, length), Tensor[[*Elements[Shape], K]])
        assert_type(torch.narrow(x, -1, 0, length), Tensor[[*Elements[Shape], K]])

    def check_gradual_boundaries(
        x: Tensor[[2, 3, 4]], bare: Tensor, dim: int, start: int, length: int
    ) -> None:
        assert_type(x.narrow(1, 0, length), Tensor[[2, int, 4]])
        assert_type(x.narrow(1, start, 2), Tensor[[2, 2, 4]])
        assert_type(torch.narrow(x, dim, 0, 2), Tensor[IntTuple])
        assert_type(torch.narrow(bare, 0, 0, 2), Tensor[IntTuple])

    def check_invalid_length_type[T, S: str](
        x: Tensor[[4, 32]], unconstrained: T, string: S
    ) -> None:
        # E: `T` is not assignable to upper bound `Int[int]` of type variable `Length`
        torch.narrow(x, 1, 0, unconstrained)
        # E: `S` is not assignable to upper bound `Int[int]` of type variable `Length`
        torch.narrow(x, 1, 0, string)
