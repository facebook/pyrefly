# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type, TYPE_CHECKING

import torch
from shape_extensions import assert_raises, assert_shape, IntVar
from torch import Tensor


def test_dot_shape() -> None:
    left = torch.ones(5)
    right = torch.ones(5)
    assert_shape(torch.dot(left, right).shape, ())
    assert_shape(left.dot(right).shape, ())


def test_dot_rejects_invalid_inputs() -> None:
    assert_shape(torch.ones(2).dot(torch.ones(2)).shape, ())

    with assert_raises(RuntimeError):
        # E: Shape dimension mismatch: expected Int[2], got Int[3]
        torch.dot(torch.ones(2), torch.ones(3))

    with assert_raises(RuntimeError):
        # E: Tensor rank mismatch: expected 1 dimensions, got 2 dimensions
        torch.dot(torch.ones((2, 3)), torch.ones(3))


if TYPE_CHECKING:

    def check_symbolic[N: IntVar](left: Tensor[[N]], right: Tensor[[N]]) -> None:
        assert_type(torch.dot(left, right), Tensor[[]])
        assert_type(left.dot(right), Tensor[[]])
