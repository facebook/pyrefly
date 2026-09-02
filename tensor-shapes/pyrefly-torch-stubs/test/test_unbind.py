# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type, TYPE_CHECKING

import torch
from shape_extensions import assert_raises, assert_shape, Elements, IntTuple, IntVar
from torch import Tensor


def test_unbind_shapes() -> None:
    parts = torch.unbind(torch.ones((2, 3, 4)))
    assert len(parts) == 2
    assert_shape(parts[0].shape, (3, 4))
    assert_shape(parts[1].shape, (3, 4))

    parts = torch.ones((2, 3, 4)).unbind(1)
    assert len(parts) == 3
    assert_shape(parts[0].shape, (2, 4))
    assert_shape(parts[2].shape, (2, 4))

    parts = torch.unbind(torch.ones((2, 3, 4)), dim=-1)
    assert len(parts) == 4
    assert_shape(parts[0].shape, (2, 3))
    assert_shape(parts[3].shape, (2, 3))

    empty = torch.empty((2, 0, 4)).unbind(1)
    assert len(empty) == 0


def test_unbind_rejects_invalid_dimensions() -> None:
    matrix = torch.ones((2, 3))
    assert_shape(matrix.unbind()[0].shape, (3,))

    with assert_raises(IndexError):
        matrix.unbind(2)  # E: unbind dimension out of range

    with assert_raises(IndexError):
        torch.unbind(matrix, -3)  # E: unbind dimension out of range

    scalar = torch.tensor(1)
    assert_shape(scalar.shape, ())
    with assert_raises(IndexError):
        scalar.unbind()  # E: unbind dimension out of range


if TYPE_CHECKING:

    def check_symbolic[N: IntVar, M: IntVar](x: Tensor[[3, N, M]]) -> None:
        assert_type(torch.unbind(x), tuple[Tensor[[N, M]], ...])
        assert_type(x.unbind(-1), tuple[Tensor[[3, N]], ...])

    def check_symbolic_suffix[Shape: IntTuple](
        x: Tensor[[*Elements[Shape], 3]],
    ) -> None:
        assert_type(torch.unbind(x, -1), tuple[Tensor[Shape], ...])
        assert_type(x.unbind(-1), tuple[Tensor[Shape], ...])

    def check_gradual_boundaries(
        x: Tensor[[2, 3]],
        dim: int,
        bare: Tensor,
        open_rank: Tensor[IntTuple],
    ) -> None:
        assert_type(torch.unbind(x, dim), tuple[Tensor[IntTuple], ...])
        assert_type(x.unbind(dim), tuple[Tensor[IntTuple], ...])
        assert_type(torch.unbind(bare), tuple[Tensor[IntTuple], ...])
        assert_type(open_rank.unbind(-1), tuple[Tensor[IntTuple], ...])
