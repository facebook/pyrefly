# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, cast, Literal

import numpy as np
from shape_extensions import assert_shape


def make_array(shape: Any, value: Any = 1.0) -> Any:
    return np.full(shape, value)


def test_exponential_and_log_preserve_shape() -> None:
    a = np.full(4, 2.0)
    b = np.ones((3, 4))

    assert_shape(np.exp(a), (4,))
    assert_shape(np.log(a), (4,))
    assert_shape(np.log2(a), (4,))
    assert_shape(np.log10(a), (4,))
    assert_shape(np.sqrt(b), (3, 4))
    assert_shape(np.power(b, 2), (3, 4))


def test_binary_ufuncs_preserve_matrix_shape() -> None:
    a = np.ones((3, 4))
    b = np.full((3, 4), 2.0)

    assert_shape(np.minimum(a, b), (3, 4))
    assert_shape(np.maximum(a, b), (3, 4))
    assert_shape(np.arctan2(a, b), (3, 4))


def test_extrema_broadcast_row_vector_over_matrix() -> None:
    matrix = np.ones((3, 4))
    row = np.full(4, 2.0)

    assert_shape(np.minimum(matrix, row), (3, 4))
    assert_shape(np.minimum(row, matrix), (3, 4))
    assert_shape(np.maximum(matrix, row), (3, 4))
    assert_shape(np.maximum(row, matrix), (3, 4))


def test_binary_ufuncs_broadcast_higher_rank_arrays() -> None:
    a = cast(
        "np.ndarray[tuple[Literal[2], Literal[3], Literal[1], Literal[5]]]",
        make_array((2, 3, 1, 5)),
    )
    b = cast(
        "np.ndarray[tuple[Literal[1], Literal[3], Literal[4], Literal[1]]]",
        make_array((1, 3, 4, 1), 2.0),
    )

    assert_shape(np.minimum(a, b), (2, 3, 4, 5))
    assert_shape(np.maximum(a, b), (2, 3, 4, 5))
    assert_shape(np.arctan2(a, b), (2, 3, 4, 5))


def test_binary_ufuncs_reject_incompatible_broadcast() -> None:
    a = np.ones((3, 4))
    b = np.ones(5)

    assert_shape(np.minimum(a, np.ones((3, 4))), (3, 4))
    try:
        assert_shape(
            np.minimum(a, b),  # E: operands could not be broadcast together
            (3, 4),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("expected NumPy to reject incompatible shapes")


def test_trig_preserves_shape() -> None:
    angles = np.ones((2, 3))

    assert_shape(np.sin(angles), (2, 3))
    assert_shape(np.cos(angles), (2, 3))
    assert_shape(np.tan(angles), (2, 3))
    assert_shape(np.arcsin(np.full((2, 3), 0.5)), (2, 3))


def test_rounding_preserves_shape() -> None:
    a = np.full(5, -1.7)

    assert_shape(np.floor(a), (5,))
    assert_shape(np.ceil(a), (5,))
    assert_shape(np.round(a), (5,))
    assert_shape(np.trunc(a), (5,))
    assert_shape(np.clip(a, -1.0, 2.0), (5,))
