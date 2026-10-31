# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, assert_type, TYPE_CHECKING

import numpy as np
from shape_extensions import assert_raises, assert_shape, IntTuple


def _assert_immutable_scalar(value: Any) -> None:
    assert isinstance(value, np.float64)
    assert not isinstance(value, np.ndarray)
    assert_shape(value.shape, ())
    if not TYPE_CHECKING:
        with assert_raises(TypeError):
            value[()] = 0.0


def test_matrix_full_integer_indexing_is_a_scalar() -> None:
    matrix: np.ndarray[[4, 4], np.dtype[np.float64]] = np.ones((4, 4), dtype=np.float64)
    value = matrix[1, 0]
    negative = matrix[-1, -1]
    # Fixed-rank overloads also match a gradual rank, so their safe result is Any.
    assert_type(value, Any)
    assert_type(negative, Any)
    assert_shape(value.shape, IntTuple, runtime=())
    _assert_immutable_scalar(value)
    _assert_immutable_scalar(negative)
    assert value == negative == 1.0


def test_vector_integer_tuple_indexing_is_a_scalar() -> None:
    vector = np.ones(4, dtype=np.float64)
    value = vector[(1,)]
    assert_type(value, Any)
    assert_shape(value.shape, IntTuple, runtime=())
    _assert_immutable_scalar(value)
    assert value == 1.0


def test_empty_tuple_and_ellipsis_have_different_result_kinds() -> None:
    scalar_array = np.ones((), dtype=np.float64)
    extracted = scalar_array[()]
    retained = scalar_array[...]
    assert_type(extracted, Any)
    assert_type(retained, np.ndarray[[], np.dtype[np.float64]])
    assert_shape(extracted.shape, IntTuple, runtime=())
    assert_shape(retained.shape, ())
    _assert_immutable_scalar(extracted)
    assert isinstance(retained, np.ndarray)
    retained[...] = 2.0
    assert scalar_array.item() == 2.0
    assert extracted == 1.0


def test_slices_newaxis_and_partial_indices_remain_arrays() -> None:
    matrix = np.ones((4, 4), dtype=np.float64)
    sliced = matrix[1:2, 0]
    expanded = matrix[1, 0, None]
    assert_type(sliced, np.ndarray[[1], np.dtype[np.float64]])
    assert_type(expanded, np.ndarray[[1], np.dtype[np.float64]])
    assert_shape(sliced.shape, (1,))
    assert_shape(expanded.shape, (1,))
    assert isinstance(sliced, np.ndarray)
    assert isinstance(expanded, np.ndarray)
    sliced[0] = 2.0
    assert matrix[1, 0] == 2.0
    expanded[0] = 3.0
    assert matrix[1, 0] == 3.0
    higher = np.ones((4, 4, 4), dtype=np.float64)
    assert_type(higher[1, 0], np.ndarray[[4], np.dtype[np.float64]])
    assert_shape(higher[1, 0].shape, (4,))


def _index_unknown_rank(
    value: np.ndarray[IntTuple, np.dtype[np.float64]],
) -> Any:
    if TYPE_CHECKING:
        assert_type(value[0], Any)
        assert_type(value[(0,)], Any)
        assert_type(value[(0, 0)], Any)
        assert_type(value[()], Any)
        assert_type(value[...], np.ndarray[IntTuple, np.dtype[np.float64]])
    return value[0]


def test_unknown_rank_integer_indexing_can_return_scalar_or_array() -> None:
    scalar = _index_unknown_rank(np.ones(4, dtype=np.float64))
    row = _index_unknown_rank(np.ones((4, 4), dtype=np.float64))
    _assert_immutable_scalar(scalar)
    assert isinstance(row, np.ndarray)
    assert_shape(row.shape, IntTuple, runtime=(4,))


def test_invalid_index_rank_remains_an_error() -> None:
    matrix = np.ones((4, 4), dtype=np.float64)
    vector = np.ones(4, dtype=np.float64)
    scalar_array = np.ones((), dtype=np.float64)
    with assert_raises(IndexError):
        matrix[1, 0, 0]  # E: Too many indices
    with assert_raises(IndexError):
        vector[(0, 0)]  # E: Too many indices
    with assert_raises(IndexError):
        scalar_array[(0,)]  # E: Cannot index scalar
    if not TYPE_CHECKING:
        with assert_raises(IndexError):
            matrix[4, 0]
        with assert_raises(IndexError):
            matrix[-5, 0]
