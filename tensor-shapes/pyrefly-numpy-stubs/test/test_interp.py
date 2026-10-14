# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type, TYPE_CHECKING

import numpy as np
from shape_extensions import assert_shape, IntTuple


def test_interp_scalar() -> None:
    """Regression for https://github.com/facebook/pyrefly/issues/5231."""
    weighted_quantiles = np.array([0.0, 0.5, 1.0])
    filtered_values = np.array([10.0, 20.0, 30.0])
    sorter = np.array([0, 1, 2])
    result: np.float64 = np.interp(0.5, weighted_quantiles, filtered_values[sorter])
    median = float(result)
    assert_type(median, float)
    assert median == 20.0
    assert_shape(np.asarray(median).shape, ())


def test_interp_array() -> None:
    result = np.interp(np.ones((2, 3)), [0.0, 2.0], [1.0, 5.0])
    assert_type(result, np.ndarray[[2, 3]])
    assert_shape(result.shape, (2, 3))
    assert result.dtype == np.float64
    assert result[0, 0] == 3.0


def test_interp_nested_sequence() -> None:
    result = np.interp([[0.0, 0.5], [1.0, 2.0]], [0.0, 1.0], [2.0, 4.0], -1.0, 9.0)
    assert_type(result, np.ndarray[[2, 2]])
    assert_shape(result.shape, (2, 2))
    assert result[1, 1] == 9.0


def test_interp_zero_dimensional() -> None:
    result: np.float64 = np.interp(np.array(0.5), [0.0, 1.0], [2.0, 4.0])
    assert float(result) == 3.0
    assert_shape(np.asarray(result).shape, ())


def test_interp_complex() -> None:
    values = np.ones(2, dtype=np.complex128)
    result = np.interp(np.ones((2, 3)), [0.0, 2.0], values)
    assert_type(result, np.ndarray[[2, 3]])
    assert_shape(result.shape, (2, 3))
    assert result.dtype == np.complex128
    scalar: np.complex128 = np.interp(0.5, [0.0, 2.0], values)
    assert scalar == 1.0 + 0.0j
    assert_shape(np.asarray(scalar).shape, ())


def test_interp_period() -> None:
    result = np.interp([0.0, 180.0, 360.0], [0.0, 180.0], [0.0, 1.0], period=360.0)
    assert_shape(result.shape, (3,))
    assert result[2] == 0.0


if TYPE_CHECKING:

    def check_interp_symbolic[Shape: IntTuple](x: np.ndarray[Shape]) -> None:
        result = np.interp(x, [0.0, 1.0], [2.0, 4.0])
        assert_type(result, np.ndarray[Shape])
