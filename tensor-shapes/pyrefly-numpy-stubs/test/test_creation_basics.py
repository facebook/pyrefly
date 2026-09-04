# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, assert_type

import numpy as np
from shape_extensions import assert_shape, IntVar

GRADUAL_SHAPE_RUNTIME_TESTS = {
    "test_diag_dtype_and_broad_offset",
    "test_diag_matrix_runtime_shape",
}


def test_zeros_1d_int_shape() -> None:
    assert_shape(np.zeros(5), (5,))


def test_ones_1d_int_shape() -> None:
    assert_shape(np.ones(4), (4,))


def test_full_1d_int_shape() -> None:
    assert_shape(np.full(3, 7.0), (3,))


def test_empty_1d_int_shape() -> None:
    assert_shape(np.empty(6), (6,))


def test_zeros_tuple_shape() -> None:
    assert_shape(np.zeros((3, 4)), (3, 4))


def test_ones_tuple_shape() -> None:
    assert_shape(np.ones((2, 5)), (2, 5))


def test_full_tuple_shape() -> None:
    assert_shape(np.full((3, 3), -1.0), (3, 3))


def test_empty_tuple_shape() -> None:
    assert_shape(np.empty((6,)), (6,))


def test_eye_square_shape() -> None:
    assert_shape(np.eye(4), (4, 4))


def test_identity_square_shape() -> None:
    assert_shape(np.identity(5), (5, 5))


def test_diag_vector_default_and_offsets() -> None:
    diagonal = np.full(5, 2.0)
    off_diagonal = np.full(4, -1.0)

    assert_shape(np.diag(diagonal), (5, 5))
    assert_shape(np.diag(off_diagonal, 1), (5, 5))
    assert_shape(np.diag(off_diagonal, k=-2), (6, 6))


def check_diag_symbolic[N: IntVar, DType](
    vector: np.ndarray[[N], DType], k: int
) -> None:
    assert_type(np.diag(vector), np.ndarray[[N, N], DType])
    assert_type(np.diag(vector, k=-2), np.ndarray[[N + 2, N + 2], DType])
    assert_type(np.diag(vector, k), np.ndarray[[int, int], DType])


def check_diag_gradual_dtype(vector: np.ndarray[[4], Any]) -> None:
    assert_type(np.diag(vector, k=1), np.ndarray[[5, 5], Any])


def test_diag_dtype_and_broad_offset() -> None:
    vector = np.full(4, 1.0, dtype=np.float32)
    k: int = 2
    result = np.diag(vector, k)

    assert_type(result, np.ndarray[[int, int], np.dtype[np.float32]])
    assert_type(result.dtype, np.dtype[np.float32])
    assert result.shape == (6, 6)


def check_diag_general_rank[DType](matrix: np.ndarray[[2, 3], DType]) -> None:
    assert_type(np.diag(matrix), np.ndarray[[int], DType])


def check_diag_unknown_rank[DType](array: np.ndarray[Any, DType]) -> None:
    assert_type(np.diag(array), np.ndarray[Any, DType])


# The gradual fallback also admits ranks NumPy itself rejects; that is the price of keeping
# the dtype for unknown-rank inputs.
def check_diag_rank_zero_falls_back[DType](array: np.ndarray[[], DType]) -> None:
    assert_type(np.diag(array), np.ndarray[Any, DType])


def check_diag_rank_three_falls_back[DType](
    array: np.ndarray[[2, 3, 4], DType],
) -> None:
    assert_type(np.diag(array), np.ndarray[Any, DType])


def test_diag_matrix_runtime_shape() -> None:
    result = np.diag(np.ones((2, 3)))
    assert_type(result, np.ndarray[[int], np.dtype[np.float64]])
    assert result.shape == (2,)
