# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type, TYPE_CHECKING

import torch
from shape_extensions import assert_raises, assert_shape, IntTuple, IntVar
from torch import Tensor


def test_permute_shapes() -> None:
    tensor = torch.ones((2, 3, 4))
    assert_shape(torch.permute(tensor, (2, 0, 1)).shape, (4, 2, 3))
    assert_shape(tensor.permute(2, 0, 1).shape, (4, 2, 3))
    assert_shape(tensor.permute((2, 0, 1)).shape, (4, 2, 3))
    assert_shape(tensor.permute(0, -1, 1).shape, (2, 4, 3))
    assert_shape(tensor.permute(0, 1, 2).shape, (2, 3, 4))


def test_permute_scalar() -> None:
    scalar = torch.tensor(1)
    assert_shape(torch.permute(scalar, ()).shape, ())
    assert_shape(scalar.permute(()).shape, ())


def test_permute_rejects_invalid_dimensions() -> None:
    tensor = torch.ones((2, 3, 4))
    assert_shape(tensor.permute(2, 0, 1).shape, (4, 2, 3))

    with assert_raises(RuntimeError):
        # E: Cannot evaluate type-level shape DSL call: permute dimensions must match the input rank
        tensor.permute(0, 1)

    with assert_raises(IndexError):
        # E: Cannot evaluate type-level shape DSL call: permute dimension out of range
        tensor.permute(0, 1, 3)

    with assert_raises(RuntimeError):
        # E: Cannot evaluate type-level shape DSL call: permute dimensions must be unique
        torch.permute(tensor, (0, -1, 2))


if TYPE_CHECKING:

    def check_missing_dims(scalar: Tensor[[]]) -> None:
        # TODO: BUG: Runtime requires `dims`, even for a scalar tensor.
        assert_type(scalar.permute(), Tensor[[]])

    def check_symbolic[N: IntVar, M: IntVar, K: IntVar](
        x: Tensor[[N, M, K]], dims: tuple[int, int, int]
    ) -> None:
        assert_type(x.permute(2, 0, 1), Tensor[[K, N, M]])
        assert_type(torch.permute(x, (-1, 0, 1)), Tensor[[K, N, M]])
        assert_type(x.permute(dims), Tensor[IntTuple])
