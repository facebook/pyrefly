# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, assert_type, cast

import numpy as np
from shape_extensions import assert_shape, IntTuple


def test_matrix_arithmetic() -> None:
    a = np.ones((3, 4))
    b = np.full((3, 4), 2.0)

    assert_shape((a + b).shape, (3, 4))
    assert_shape((b - a).shape, (3, 4))
    assert_shape((a * b).shape, (3, 4))
    assert_shape((b**2).shape, (3, 4))


def test_column_subtraction_preserves_shape() -> None:
    outcomes = np.ones((3, 1))
    probabilities = np.full((3, 1), 0.25)

    assert_shape((outcomes - probabilities).shape, (3, 1))
    assert_shape((1.0 - probabilities).shape, (3, 1))


def test_scalar_rhs_arithmetic() -> None:
    a = np.full(4, 2.0)
    b = np.ones((3, 4))
    c = np.full((3, 4), 2.0)

    assert_shape((a * 2.0).shape, (4,))
    assert_shape((a + 1.0).shape, (4,))
    assert_shape((a - 1.0).shape, (4,))
    assert_shape((a**2).shape, (4,))
    assert_shape((b + 1.0).shape, (3, 4))
    assert_shape((1.0 - b).shape, (3, 4))
    assert_shape((c * 2.0).shape, (3, 4))


def test_unary_arithmetic() -> None:
    a = np.full(5, -1.0)

    assert_shape(np.abs(a).shape, (5,))
    assert_shape(np.negative(a).shape, (5,))
    assert_shape((-a).shape, (5,))
    assert_shape((+a).shape, (5,))
    assert_shape((-np.ones((3, 4))).shape, (3, 4))


def test_scalar_defaults_preserve_shape_and_dtype() -> None:
    a = np.full((2, 3), 2.0, dtype=np.float64)

    assert_type(a + 1.0, np.ndarray[[2, 3], np.dtype[np.float64]])
    assert_type(1.0 + a, np.ndarray[[2, 3], np.dtype[np.float64]])
    assert_type(a - 1.0, np.ndarray[[2, 3], np.dtype[np.float64]])
    assert_type(1.0 - a, np.ndarray[[2, 3], np.dtype[np.float64]])
    assert_type(a * 2.0, np.ndarray[[2, 3], np.dtype[np.float64]])
    assert_type(2.0 * a, np.ndarray[[2, 3], np.dtype[np.float64]])
    assert_type(a / 2.0, np.ndarray[[2, 3], np.dtype[np.float64]])
    assert_type(2.0 / a, np.ndarray[[2, 3], np.dtype[np.float64]])
    assert_type(a // 2.0, np.ndarray[[2, 3], np.dtype[np.float64]])
    assert_type(2.0 // a, np.ndarray[[2, 3], np.dtype[np.float64]])
    assert_type(a % 2.0, np.ndarray[[2, 3], np.dtype[np.float64]])
    assert_type(2.0 % a, np.ndarray[[2, 3], np.dtype[np.float64]])
    assert_type(a**2.0, np.ndarray[[2, 3], np.dtype[np.float64]])
    assert_type(2.0**a, np.ndarray[[2, 3], np.dtype[np.float64]])

    assert_type(a < 2.0, np.ndarray[[2, 3], np.dtype[np.bool_]])
    assert_type(a <= 2.0, np.ndarray[[2, 3], np.dtype[np.bool_]])
    assert_type(a > 2.0, np.ndarray[[2, 3], np.dtype[np.bool_]])
    assert_type(a >= 2.0, np.ndarray[[2, 3], np.dtype[np.bool_]])
    assert_type(a == 2.0, np.ndarray[[2, 3], np.dtype[np.bool_]])
    assert_type(a != 2.0, np.ndarray[[2, 3], np.dtype[np.bool_]])

    integers = np.ones((2, 3), dtype=np.int64)
    assert_type(integers & 1, np.ndarray[[2, 3], Any])
    assert_type(1 & integers, np.ndarray[[2, 3], Any])
    assert_type(integers | 1, np.ndarray[[2, 3], Any])
    assert_type(1 | integers, np.ndarray[[2, 3], Any])
    assert_type(integers ^ 1, np.ndarray[[2, 3], Any])
    assert_type(1 ^ integers, np.ndarray[[2, 3], Any])

    scalar = np.array(2.0)
    assert_type(scalar + 1.0, np.ndarray[[]])
    assert_shape((scalar + 1.0).shape, ())
    assert_shape((a + scalar).shape, (2, 3))


def test_scalar_defaults_preserve_unknown_shapes() -> None:
    unknown = cast("np.ndarray", np.ones((2, 3), dtype=np.int64))
    concrete = np.ones((2, 3))

    assert_type(unknown + 1.0, np.ndarray)
    assert_type(1.0 + unknown, np.ndarray)
    assert_type(unknown + concrete, np.ndarray)
    assert_type(concrete + unknown, np.ndarray[IntTuple, np.dtype[np.float64]])
    assert_shape((unknown + 1.0).shape, IntTuple, runtime=(2, 3))
    assert_type(unknown > 1.0, Any)
    assert_type(unknown & 1, np.ndarray)
