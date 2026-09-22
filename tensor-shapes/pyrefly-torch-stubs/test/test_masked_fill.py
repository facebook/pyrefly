# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type, TYPE_CHECKING

import torch
from shape_extensions import assert_raises, assert_shape, IntTuple, IntVar
from torch import Tensor


def test_masked_fill_shapes() -> None:
    tensor = torch.zeros((2, 3))
    mask = torch.tensor([[True], [False]])
    assert_shape(torch.masked_fill(tensor, mask, 1.0).shape, (2, 3))
    assert_shape(tensor.masked_fill(mask, -1.0).shape, (2, 3))
    assert_shape(tensor.masked_fill_(mask, 2.0).shape, (2, 3))


def test_masked_fill_can_expand_the_input() -> None:
    tensor = torch.zeros((2, 3))
    mask = torch.ones((1, 2, 3), dtype=torch.bool)

    assert_shape(torch.masked_fill(tensor, mask, 1.0).shape, (1, 2, 3))
    assert_shape(tensor.masked_fill(mask, 1.0).shape, (1, 2, 3))


def test_masked_fill_rejects_invalid_broadcasts() -> None:
    tensor = torch.zeros((2, 3))
    assert_shape(tensor.shape, (2, 3))

    incompatible = torch.ones((4,), dtype=torch.bool)
    with assert_raises(RuntimeError):
        # E: Cannot broadcast dimension
        torch.masked_fill(tensor, incompatible, 1.0)

    expanding = torch.ones((1, 2, 3), dtype=torch.bool)
    # TODO: BUG: In-place masked_fill cannot expand the input shape.
    with assert_raises(RuntimeError):
        tensor.masked_fill_(expanding, 1.0)


if TYPE_CHECKING:

    def check_symbolic[B: IntVar, Heads: IntVar, T: IntVar](
        tensor: Tensor[[B, Heads, T, T]], mask: Tensor[[1, 1, T, T]]
    ) -> None:
        assert_type(tensor.masked_fill(mask, float("-inf")), Tensor[[B, Heads, T, T]])
        assert_type(torch.masked_fill(tensor, mask, 0.0), Tensor[[B, Heads, T, T]])

    def check_gradual_mask(tensor: Tensor[[2, 3]], mask: Tensor[IntTuple]) -> None:
        assert_type(tensor.masked_fill(mask, 0.0), Tensor[IntTuple])
