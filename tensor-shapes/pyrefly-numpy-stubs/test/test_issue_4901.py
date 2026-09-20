# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
from typing import Any, assert_type, TYPE_CHECKING

import numpy as np
from shape_extensions import assert_shape


def test_statistics_and_array_like_inputs() -> None:
    weights = np.array([[1.0, 2.0], [3.0, 4.0]])
    column = weights[:, 0]
    assert_shape(weights.shape, (2, 2), runtime=(2, 2))
    assert np.percentile(column, 50) == np.median(column)
    assert np.average(column, weights=np.ones_like(column)) == 2.0
    assert np.cumsum(column).tolist() == [1.0, 4.0]
    assert np.linspace(0, 1, 3).tolist() == [0.0, 0.5, 1.0]
    assert np.log(0.9) < 0
    assert np.maximum(0, [-1, 1]).tolist() == [0, 1]
    assert np.clip([0, 1, 2], 0, 1).tolist() == [0, 1, 1]
    assert np.min([1.0, 2.0]) == 1.0
    assert np.max([1.0, 2.0]) == 2.0
    assert np.isinf(np.inf)
    assert np.e > 2


def test_iteration_and_assignment() -> None:
    values = np.zeros(3, dtype=np.int32)
    values[0] = 2
    values[1:] = [3, 4]
    values[2] += 1
    assert list(values) == [2, 3, 5]
    assert set(values) == {2, 3, 5}
    assert list(zip(values, values)) == [(2, 2), (3, 3), (5, 5)]
    assert_shape(values.shape, (3,))

    matrix = np.zeros((2, 3))
    matrix[np.arange(2), np.zeros(2, dtype=np.intp)] = 1
    for row in matrix:
        assert_type(row, np.ndarray[[3], np.dtype[np.float64]])
        assert_shape(row.shape, (3,))
        assert row.tolist() == [1, 0, 0]


def test_comparisons_and_boolean_masks() -> None:
    values = np.ones((2, 3))
    row = np.zeros(3)
    greater = values > row
    assert_type(greater, np.ndarray[[2, 3], np.dtype[np.bool_]])
    assert_shape(greater.shape, (2, 3))
    assert greater.all()
    assert_shape((values >= 1).shape, (2, 3))
    assert_shape((values < 2).shape, (2, 3))
    assert_shape((0 < values).shape, (2, 3))
    assert_shape((values <= row).shape, (2, 3))
    assert_shape((values == row).shape, (2, 3))
    assert_shape((values != row).shape, (2, 3))
    equal = np.equal(values, row)
    assert_shape(equal.shape, (2, 3))
    assert not equal.any()
    valid = ~np.isnan(values)
    assert_shape(valid.shape, (2, 3))
    values[valid & greater] = 2
    assert (values == 2).all()
    assert_shape((valid | greater).shape, (2, 3))
    assert_shape((valid ^ greater).shape, (2, 3))
    assert_shape((True & valid).shape, (2, 3))
    assert_shape((False | valid).shape, (2, 3))
    assert_shape((True ^ valid).shape, (2, 3))
    assert_shape((valid & np.ones(3, dtype=np.bool_)).shape, (2, 3))


def test_scalar_conversion_after_reduction() -> None:
    values = np.ones(3)
    total = np.sum(values)
    assert_shape(total.shape, ())
    assert_type(float(total), float)
    assert_type(int(total), int)
    assert_type(complex(total), complex)
    assert float(total) == 3.0
    assert int(total) == 3
    assert complex(total) == 3 + 0j
    assert math.ceil(total) == 3
    assert math.floor(total) == 3
    assert not math.isnan(total)
    assert total > 0
    assert_type(total == 3, Any)
    assert_type(total != 0, Any)
    assert total == 3
    assert total != 0
    assert max(1.0, total) == 3.0
    assert min(total, 4.0) == 3.0
    assert float(values[0]) == 1.0
    assert math.isnan(np.array(np.nan))
    assert values.min().item() == 1.0
    assert_type(values.tolist(), Any)


def test_invalid_operations_remain_errors() -> None:
    values = np.ones((2, 3))
    assert_shape(values.shape, (2, 3))
    try:
        values > np.ones(4)  # E: Cannot broadcast dimension
    except ValueError:
        pass
    else:
        raise AssertionError("expected NumPy to reject incompatible shapes")

    try:
        values["invalid"] = 0  # E: Cannot set item
    except IndexError:
        pass
    else:
        raise AssertionError("expected NumPy to reject a string index")

    try:
        np.min(values, axis=3)  # E: axis out of bounds
    except ValueError:
        pass
    else:
        raise AssertionError("expected NumPy to reject an invalid axis")

    try:
        values.__float__()  # E: not assignable
    except TypeError:
        pass
    else:
        raise AssertionError("expected NumPy to require a scalar array")

    if TYPE_CHECKING:
        # Unknown shapes retain support for NumPy scalar conversion and iteration.
        def check_unknown_shape(array: np.ndarray) -> None:
            math.isnan(array)
            list(array)
