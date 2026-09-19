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
    # TODO: BUG: Resolve -1 target dimensions from the input shape.
    assert_shape(
        torch.broadcast_to(torch.ones((2, 3)), (-1, 3)).shape,
        (-1, 3),
        runtime=(2, 3),
    )
    assert_shape(torch.broadcast_to(torch.ones(()), (2, 3)).shape, (2, 3))


def test_broadcast_to_rejects_invalid_targets() -> None:
    tensor = torch.ones((2, 3))
    assert_shape(torch.broadcast_to(tensor, (4, 2, 3)).shape, (4, 2, 3))

    with assert_raises(RuntimeError):
        # TODO: BUG: Reject incompatible broadcast targets statically.
        torch.broadcast_to(tensor, (4, 5))

    with assert_raises(RuntimeError):
        # TODO: BUG: Reject targets with fewer dimensions statically.
        torch.broadcast_to(tensor, (3,))

    with assert_raises(RuntimeError):
        # TODO: BUG: Reject target dimensions below -1 statically.
        torch.broadcast_to(tensor, (-2, 3))

    with assert_raises(RuntimeError):
        # TODO: BUG: Reject -1 for a new leading dimension statically.
        torch.broadcast_to(tensor, (-1, 2, 3))


if TYPE_CHECKING:

    def check_symbolic[N: IntVar](x: Tensor[[N, 1]], n: Int[N], plain: int) -> None:
        assert_type(torch.broadcast_to(x, (n, 5)), Tensor[[N, 5]])
        assert_type(torch.broadcast_to(x, (3, 5)), Tensor[[3, 5]])
        assert_type(torch.broadcast_to(x, (n, plain)), Tensor[[N, int]])
