# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type, TYPE_CHECKING

import torch
from shape_extensions import assert_raises, assert_shape, IntTuple, IntVar
from torch import Tensor


def test_scatter_shapes() -> None:
    tensor = torch.zeros((3, 4))
    column_indices = torch.tensor([[0, 1], [2, 3], [1, 0]])
    columns = torch.ones((3, 2))
    assert_shape(torch.scatter(tensor, 1, column_indices, columns).shape, (3, 4))
    assert_shape(tensor.scatter(-1, column_indices, columns).shape, (3, 4))

    row_indices = torch.tensor([[0, 1, 2, 0], [2, 0, 1, 2]])
    rows = torch.ones((2, 4))
    assert_shape(tensor.scatter(0, row_indices, rows).shape, (3, 4))


def test_scatter_scalar_shapes() -> None:
    scalar = torch.tensor(0.0)
    scalar_index = torch.tensor(0)
    vector_index = torch.tensor([0], dtype=torch.int64)
    scalar_source = torch.tensor(1.0)
    vector_source = torch.ones(1)

    assert_shape(scalar.scatter(0, scalar_index, scalar_source).shape, ())
    assert_shape(torch.scatter(scalar, -1, vector_index, vector_source).shape, ())
    assert_shape(scalar.scatter(0, scalar_index, vector_source).shape, ())
    assert_shape(scalar.scatter(0, vector_index, scalar_source).shape, ())
    assert_shape(
        scalar.scatter(0, torch.empty(0, dtype=torch.int64), scalar_source).shape,
        (),
    )


def test_scatter_scalar_rejects_invalid_inputs() -> None:
    scalar = torch.tensor(0.0)
    scalar_index = torch.tensor(0)
    scalar_source = torch.tensor(1.0)
    assert_shape(scalar.scatter(0, scalar_index, scalar_source).shape, ())

    with assert_raises(IndexError):
        scalar.scatter(1, scalar_index, scalar_source)  # E: dimension out of range

    with assert_raises(RuntimeError):
        # E: scatter index rank must match input rank
        scalar.scatter(0, torch.zeros((1, 1), dtype=torch.int64), torch.ones((1, 1)))

    with assert_raises(RuntimeError):
        # E: scatter source rank must match index rank
        scalar.scatter(0, scalar_index, torch.ones((1, 1)))

    with assert_raises(RuntimeError):
        # E: scatter index shape exceeds source shape
        scalar.scatter(0, torch.zeros(2, dtype=torch.int64), scalar_source)


def test_scatter_empty_index_skips_shape_validation() -> None:
    scalar = torch.tensor(0.0)
    empty_matrix = torch.empty((0, 2), dtype=torch.int64)
    assert_shape(scalar.scatter(0, empty_matrix, torch.tensor(1.0)).shape, ())

    matrix = torch.zeros((2, 3))
    assert_shape(matrix.scatter(0, empty_matrix, torch.tensor(1.0)).shape, (2, 3))


def test_scatter_rejects_invalid_dimensions_and_shapes() -> None:
    tensor = torch.zeros((3, 4))
    indices = torch.zeros((3, 2), dtype=torch.int64)
    source = torch.ones((3, 2))
    assert_shape(tensor.scatter(1, indices, source).shape, (3, 4))

    with assert_raises(IndexError):
        tensor.scatter(2, indices, source)  # E: scatter dimension out of range

    with assert_raises(RuntimeError):
        # E: scatter index rank must match input rank
        torch.scatter(tensor, 1, torch.zeros(2, dtype=torch.int64), source)

    with assert_raises(IndexError):
        tensor.scatter(  # E: scatter source rank must match index rank
            1, indices, torch.ones(6)
        )

    with assert_raises(RuntimeError):
        tensor.scatter(  # E: scatter index shape exceeds input shape
            1, torch.zeros((4, 2), dtype=torch.int64), torch.ones((4, 2))
        )

    with assert_raises(RuntimeError):
        tensor.scatter(  # E: scatter index shape exceeds source shape
            1, torch.zeros((3, 3), dtype=torch.int64), torch.ones((3, 2))
        )


if TYPE_CHECKING:

    def check_symbolic[N: IntVar, M: IntVar, K: IntVar](
        tensor: Tensor[[N, M]], indices: Tensor[[N, K]], source: Tensor[[N, K]]
    ) -> None:
        assert_type(torch.scatter(tensor, 1, indices, source), Tensor[[N, M]])
        assert_type(tensor.scatter(1, indices, source), Tensor[[N, M]])

    def check_gradual_inputs(
        tensor: Tensor[[2, 3]], indices: Tensor[IntTuple], source: Tensor[IntTuple]
    ) -> None:
        assert_type(tensor.scatter(0, indices, source), Tensor[[2, 3]])
