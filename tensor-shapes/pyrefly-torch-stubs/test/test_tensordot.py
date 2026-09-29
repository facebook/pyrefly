# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type, TYPE_CHECKING

import torch
from shape_extensions import assert_raises, assert_shape, IntVar
from torch import Tensor


def test_tensordot_shapes() -> None:
    left = torch.randn((2, 3, 4))
    right = torch.randn((4, 6))
    assert_shape(torch.tensordot(left, right, dims=1).shape, (2, 3, 6))
    assert_shape(
        torch.tensordot(torch.randn((2, 3)), torch.randn((2, 3)), dims=2).shape, ()
    )


def test_tensordot_rejects_invalid_dimensions() -> None:
    left = torch.randn((2, 3))
    right = torch.randn((4, 5))
    assert_shape(left.shape, (2, 3))

    with assert_raises(RuntimeError):
        torch.tensordot(left, right, dims=-1)  # E: dims must be non-negative
    with assert_raises(RuntimeError):
        torch.tensordot(left, right, dims=3)  # E: dims exceeds input rank
    with assert_raises(RuntimeError):
        # E: contracted dimensions must match
        torch.tensordot(left, right, dims=1)


if TYPE_CHECKING:

    def check_symbolic_tensordot[N: IntVar, M: IntVar, K: IntVar](
        left: Tensor[[N, M, K]], right: Tensor[[K, 6]]
    ) -> None:
        assert_type(torch.tensordot(left, right, dims=1), Tensor[[N, M, 6]])
