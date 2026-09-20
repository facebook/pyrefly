# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, assert_type, TYPE_CHECKING

import torch
from shape_extensions import assert_raises, assert_shape, IntVar
from torch import Tensor


def test_masked_select_has_data_dependent_length() -> None:
    tensor = torch.arange(6).reshape(2, 3)
    mask = torch.tensor([[True], [False]])
    assert_shape(torch.masked_select(tensor, mask).shape, (Any,), runtime=(3,))
    assert_shape(tensor.masked_select(mask).shape, (Any,), runtime=(3,))

    empty_mask = torch.zeros((2, 3), dtype=torch.bool)
    assert_shape(tensor.masked_select(empty_mask).shape, (Any,), runtime=(0,))


def test_masked_select_rejects_incompatible_masks() -> None:
    tensor = torch.zeros((2, 3))
    assert_shape(
        tensor.masked_select(torch.ones((2, 1), dtype=torch.bool)).shape,
        (Any,),
        runtime=(6,),
    )

    # TODO: BUG: Reject masks that cannot broadcast to the input.
    with assert_raises(RuntimeError):
        torch.masked_select(tensor, torch.ones(4, dtype=torch.bool))


if TYPE_CHECKING:

    def check_symbolic[N: IntVar, M: IntVar](
        tensor: Tensor[[N, M]], mask: Tensor[[N, M]]
    ) -> None:
        assert_type(torch.masked_select(tensor, mask), Tensor[[Any]])
        assert_type(tensor.masked_select(mask), Tensor[[Any]])
