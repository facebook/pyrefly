# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type, TYPE_CHECKING

import torch
from shape_extensions import assert_raises, assert_shape, Int, IntTuple, IntVar
from torch import Tensor


def test_tile_shapes() -> None:
    tensor = torch.ones((2, 3))
    assert_shape(torch.tile(tensor, (2, 3)).shape, (4, 9))
    assert_shape(torch.tile(input=tensor, dims=(2, 3)).shape, (4, 9))
    assert_shape(tensor.tile((4,)).shape, (2, 12))
    assert_shape(torch.tile(tensor, (4, 5, 6)).shape, (4, 10, 18))
    assert_shape(tensor.tile(()).shape, (2, 3))
    assert_shape(tensor.tile((1, 0)).shape, (2, 0))

    assert_shape(torch.tile(torch.tensor(1), (2, 3)).shape, (2, 3))
    assert_shape(torch.tensor(1).tile((2, 3)).shape, (2, 3))


def test_tile_rejects_negative_repeats() -> None:
    tensor = torch.ones((2, 3))
    assert_shape(tensor.tile((1, 1)).shape, (2, 3))

    with assert_raises(RuntimeError):
        torch.tile(tensor, (1, -1))  # E: repeat dimensions must be non-negative

    with assert_raises(RuntimeError):
        torch.tile(tensor, (-1,))  # E: repeat dimensions must be non-negative


if TYPE_CHECKING:

    def check_symbolic[N: IntVar](x: Tensor[[N, 3]], n: Int[N]) -> None:
        assert_type(x.tile((2, 1)), Tensor[[2 * N, 3]])
        assert_type(x.tile((2, n)), Tensor[[2 * N, 3 * N]])

    def check_gradual(
        open_input: Tensor[IntTuple],
        concrete: Tensor[[2, 3]],
        open_repeats: tuple[int, ...],
    ) -> None:
        assert_type(open_input.tile((2, 3)), Tensor[IntTuple])
        assert_type(torch.tile(concrete, open_repeats), Tensor[IntTuple])

    def check_invalid_argument_types(x: Tensor[[2, 3]]) -> None:
        torch.tile(x, [2, 3])  # E: is not assignable to upper bound `IntTuple`
        x.tile((2, 3.0))  # E: is not assignable to parameter `dims`
