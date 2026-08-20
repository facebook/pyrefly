# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import numpy as np
from shape_extensions import assert_shape


def test_matrix_arithmetic() -> None:
    a = np.ones((3, 4))
    b = np.full((3, 4), 2.0)

    assert_shape(a + b, (3, 4))
    assert_shape(b - a, (3, 4))
    assert_shape(a * b, (3, 4))
    assert_shape(b**2, (3, 4))


def test_column_subtraction_preserves_shape() -> None:
    outcomes = np.ones((3, 1))
    probabilities = np.full((3, 1), 0.25)

    assert_shape(outcomes - probabilities, (3, 1))
    assert_shape(1.0 - probabilities, (3, 1))


def test_scalar_rhs_arithmetic() -> None:
    a = np.full(4, 2.0)
    b = np.ones((3, 4))
    c = np.full((3, 4), 2.0)

    assert_shape(a * 2.0, (4,))
    assert_shape(a + 1.0, (4,))
    assert_shape(a - 1.0, (4,))
    assert_shape(a**2, (4,))
    assert_shape(b + 1.0, (3, 4))
    assert_shape(1.0 - b, (3, 4))
    assert_shape(c * 2.0, (3, 4))


def test_unary_arithmetic() -> None:
    a = np.full(5, -1.0)

    assert_shape(np.abs(a), (5,))
    assert_shape(np.negative(a), (5,))
    assert_shape(-a, (5,))
    assert_shape(+a, (5,))
    assert_shape(-np.ones((3, 4)), (3, 4))
