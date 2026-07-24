# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import numpy as np
from shape_extensions import assert_shape


def test_row_vector_broadcasts_over_matrix() -> None:
    matrix = np.ones((3, 4))
    row = np.full(4, 2.0)

    assert_shape(matrix + row, (3, 4))
    assert_shape(row + matrix, (3, 4))
    assert_shape(matrix * row, (3, 4))
    assert_shape(row * matrix, (3, 4))


def test_singleton_column_broadcasts_over_matrix() -> None:
    matrix = np.ones((3, 4))
    column = np.full((3, 1), 2.0)

    assert_shape(matrix + column, (3, 4))
    assert_shape(column + matrix, (3, 4))
    assert_shape(matrix * column, (3, 4))
    assert_shape(column * matrix, (3, 4))


def test_column_broadcasts_with_row_vector() -> None:
    column = np.ones((3, 1))
    row = np.full(4, 2.0)

    assert_shape(column + row, (3, 4))
    assert_shape(row + column, (3, 4))
    assert_shape(column - row, (3, 4))
    assert_shape(row - column, (3, 4))
    assert_shape(column * row, (3, 4))
    assert_shape(row * column, (3, 4))


def test_scalar_left_and_division_broadcasting() -> None:
    a = np.ones((3, 4))

    assert_shape(1.0 + a, (3, 4))
    assert_shape(2.0 * a, (3, 4))
    assert_shape(a / 2.0, (3, 4))
    assert_shape(1.0 / (a + 1.0), (3, 4))


def test_subtraction_rejects_incompatible_broadcast() -> None:
    a = np.ones((3, 4))
    b = np.ones(5)

    assert_shape(a - np.ones((3, 4)), (3, 4))
    try:
        # E: Cannot evaluate type-level shape DSL call
        a - b
    except ValueError:
        pass
    else:
        raise AssertionError("expected NumPy to reject incompatible shapes")
