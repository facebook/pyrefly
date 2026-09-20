# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type, TYPE_CHECKING

import torch
from shape_extensions import assert_raises, assert_shape, Int, IntVar
from torch import Tensor


def test_broadcast_to_shapes() -> None:
    assert_shape(torch.broadcast_to(torch.ones(3), (2, 3)).shape, (2, 3))
    assert_shape(torch.broadcast_to(torch.ones((1, 3)), (2, 3)).shape, (2, 3))
    assert_shape(torch.broadcast_to(torch.ones((2, 1)), (2, 4)).shape, (2, 4))
    assert_shape(torch.broadcast_to(torch.ones((2, 3)), (-1, 3)).shape, (2, 3))
    assert_shape(torch.broadcast_to(torch.ones(()), (2, 3)).shape, (2, 3))


def test_broadcast_to_rejects_invalid_targets() -> None:
    tensor = torch.ones((2, 3))
    assert_shape(torch.broadcast_to(tensor, (4, 2, 3)).shape, (4, 2, 3))

    with assert_raises(RuntimeError):
        # E: expand cannot resize a non-singleton dimension
        torch.broadcast_to(tensor, (4, 5))

    with assert_raises(RuntimeError):
        # E: expand target rank cannot be smaller than input rank
        torch.broadcast_to(tensor, (3,))

    with assert_raises(RuntimeError):
        # E: expand target dimension cannot be less than -1
        torch.broadcast_to(tensor, (-2, 3))

    with assert_raises(RuntimeError):
        # E: expand cannot use -1 for a new leading dimension
        torch.broadcast_to(tensor, (-1, 2, 3))


if TYPE_CHECKING:

    def check_symbolic[N: IntVar](x: Tensor[[N, 1]], n: Int[N], plain: int) -> None:
        assert_type(torch.broadcast_to(x, (n, 5)), Tensor[[N, 5]])
        assert_type(torch.broadcast_to(x, (3, 5)), Tensor[[3, 5]])
        assert_type(torch.broadcast_to(x, (n, plain)), Tensor[[N, int]])
