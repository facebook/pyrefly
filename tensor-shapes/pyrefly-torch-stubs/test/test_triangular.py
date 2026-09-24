# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type, TYPE_CHECKING

import torch
from shape_extensions import assert_shape, IntTuple
from torch import Tensor


def test_triangular_shapes() -> None:
    matrix = torch.randn((3, 4))
    assert_shape(torch.tril(matrix).shape, (3, 4))
    assert_shape(torch.triu(matrix, diagonal=1).shape, (3, 4))
    assert_shape(matrix.tril(diagonal=-1).shape, (3, 4))
    assert_shape(matrix.triu().shape, (3, 4))


def test_triangular_index_shapes() -> None:
    assert_shape(torch.tril_indices(3, 3).shape, (2, int), runtime=(2, 6))
    assert_shape(torch.triu_indices(4, 5, offset=1).shape, (2, int), runtime=(2, 10))


if TYPE_CHECKING:

    def check_symbolic_triangular[Shape: IntTuple](tensor: Tensor[Shape]) -> None:
        assert_type(torch.tril(tensor), Tensor[Shape])
        assert_type(tensor.triu(), Tensor[Shape])

    def check_runtime_triangular_indices(rows: int, columns: int, offset: int) -> None:
        assert_type(torch.tril_indices(rows, columns, offset), Tensor[[2, int]])
        assert_type(torch.triu_indices(rows, columns, offset), Tensor[[2, int]])
