# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type, TYPE_CHECKING

import torch
from shape_extensions import assert_raises, assert_shape, IntTuple, IntVar
from torch import Tensor


def test_where_broadcasts_shapes() -> None:
    condition = torch.tensor([[True], [False], [True]])
    columns = torch.ones((1, 4))
    matrix = torch.zeros((3, 4))
    assert_shape(torch.where(condition, columns, matrix).shape, (3, 4))
    assert_shape(torch.where(condition, matrix, 0.0).shape, (3, 4))
    assert_shape(torch.where(condition, 1.0, matrix).shape, (3, 4))

    assert_shape(torch.where(condition, 1.0, 0.0).shape, (3, 1))


def test_where_indices() -> None:
    condition = torch.tensor([[True, False, True], [False, True, False]])
    rows, columns = torch.where(condition)
    assert_shape(rows.shape, IntTuple, runtime=(3,))
    assert_shape(columns.shape, IntTuple, runtime=(3,))


def test_where_rejects_incompatible_shapes() -> None:
    condition = torch.ones((2, 3), dtype=torch.bool)
    assert_shape(torch.where(condition, torch.ones((2, 3)), 0.0).shape, (2, 3))

    with assert_raises(RuntimeError):
        # E: Cannot broadcast dimension
        torch.where(condition, torch.ones((4, 3)), torch.zeros((2, 3)))


def test_where_rejects_invalid_keyword_forms() -> None:
    condition = torch.ones((2, 3), dtype=torch.bool)
    tensor = torch.ones((2, 3))
    assert_shape(torch.where(condition, tensor, tensor).shape, (2, 3))

    with assert_raises(TypeError):
        # A scalar value parameter is named `self` at runtime, not `input`.
        # E: No matching overload
        torch.where(condition, input=0.0, other=tensor)

    with assert_raises(TypeError):
        torch.where(condition, tensor, 0.0, out=tensor)  # E: not assignable


if TYPE_CHECKING:

    def check_symbolic_where[N: IntVar, M: IntVar](
        condition: Tensor[[N, 1]],
        row: Tensor[[1, M]],
        matrix: Tensor[[N, M]],
        dynamic: Tensor,
    ) -> None:
        assert_type(torch.where(condition, row, matrix), Tensor[[N, M]])
        assert_type(torch.where(condition, matrix, 0.0), Tensor[[N, M]])
        assert_type(torch.where(condition, 1.0, matrix), Tensor[[N, M]])
        assert_type(torch.where(condition, dynamic, matrix), Tensor[IntTuple])
