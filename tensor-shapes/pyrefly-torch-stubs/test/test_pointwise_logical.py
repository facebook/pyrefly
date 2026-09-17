# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type, TYPE_CHECKING

import torch
from shape_extensions import assert_raises, assert_shape, IntVar
from torch import Tensor


def test_logical_function_shapes() -> None:
    left = torch.ones((2, 1))
    right = torch.ones((1, 3))

    assert_shape(torch.logical_and(left, right).shape, (2, 3))
    assert_shape(torch.logical_not(left).shape, (2, 1))


def test_logical_method_shapes() -> None:
    left = torch.ones((2, 1))
    right = torch.ones((1, 3))

    assert_shape(left.logical_or(right).shape, (2, 3))
    assert_shape(left.logical_not().shape, (2, 1))


def test_logical_rejects_incompatible_shapes() -> None:
    left = torch.ones((2, 3))
    right = torch.ones((4, 5))
    assert_shape(torch.logical_and(left, torch.ones((2, 3))).shape, (2, 3))

    with assert_raises(RuntimeError):
        # E: Cannot broadcast dimension
        torch.logical_and(left, right)
    with assert_raises(RuntimeError):
        # E: Cannot broadcast dimension
        left.logical_and(right)


if TYPE_CHECKING:

    def check_symbolic_logical[N: IntVar, M: IntVar](
        left: Tensor[[N, 1]], right: Tensor[[1, M]]
    ) -> None:
        assert_type(torch.logical_and(left, right), Tensor[[N, M]])
        assert_type(torch.logical_or(left, right), Tensor[[N, M]])
        assert_type(left.logical_and(right), Tensor[[N, M]])
        assert_type(left.logical_or(right), Tensor[[N, M]])
