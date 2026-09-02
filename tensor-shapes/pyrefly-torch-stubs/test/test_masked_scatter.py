# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type, TYPE_CHECKING

import torch
from shape_extensions import assert_raises, assert_shape, IntTuple, IntVar
from torch import Tensor


def test_masked_scatter_shapes() -> None:
    tensor = torch.zeros((2, 3))
    mask = torch.tensor([[True], [False]])
    source = torch.arange(3, dtype=torch.float32)
    assert_shape(torch.masked_scatter(tensor, mask, source).shape, (2, 3))
    assert_shape(tensor.masked_scatter(mask, source).shape, (2, 3))
    assert_shape(tensor.masked_scatter_(mask, source).shape, (2, 3))


def test_masked_scatter_can_expand_the_input() -> None:
    tensor = torch.zeros((2, 3))
    mask = torch.ones((1, 2, 3), dtype=torch.bool)
    source = torch.arange(6, dtype=torch.float32)

    assert_shape(torch.masked_scatter(tensor, mask, source).shape, (1, 2, 3))
    assert_shape(tensor.masked_scatter(mask, source).shape, (1, 2, 3))


def test_masked_scatter_runtime_errors() -> None:
    tensor = torch.zeros((2, 3))
    assert_shape(tensor.shape, (2, 3))

    incompatible = torch.ones((4,), dtype=torch.bool)
    with assert_raises(RuntimeError):
        # E: Cannot broadcast dimension
        torch.masked_scatter(tensor, incompatible, torch.ones(8))

    expanding = torch.ones((1, 2, 3), dtype=torch.bool)
    # TODO: BUG: In-place masked_scatter cannot expand the input shape.
    with assert_raises(RuntimeError):
        tensor.masked_scatter_(expanding, torch.ones(6))

    # The mask values, rather than its shape alone, determine the required source size.
    with assert_raises(RuntimeError):
        tensor.masked_scatter(torch.ones((2, 3), dtype=torch.bool), torch.ones(2))


if TYPE_CHECKING:

    def check_symbolic[N: IntVar, M: IntVar](
        tensor: Tensor[[N, M]], mask: Tensor[[N, M]], source: Tensor[[N * M]]
    ) -> None:
        assert_type(tensor.masked_scatter(mask, source), Tensor[[N, M]])
        assert_type(torch.masked_scatter(tensor, mask, source), Tensor[[N, M]])

    def check_gradual_mask(
        tensor: Tensor[[2, 3]], mask: Tensor[IntTuple], source: Tensor[[6]]
    ) -> None:
        assert_type(tensor.masked_scatter(mask, source), Tensor[IntTuple])
