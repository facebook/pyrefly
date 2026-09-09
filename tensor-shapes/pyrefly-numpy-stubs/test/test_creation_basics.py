# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, assert_type, TYPE_CHECKING

import numpy as np
from shape_extensions import assert_shape, IntTuple, IntVar

GRADUAL_SHAPE_RUNTIME_TESTS = {
    "test_diag_dtype_and_broad_offset",
    "test_diag_matrix_runtime_shape",
}


def check_array_and_asarray_list_literal_types() -> None:
    assert_type(np.array(1), np.ndarray[[], Any])
    assert_type(np.array(None), np.ndarray[[], Any])
    assert_type(np.array([1, 2, 3]), np.ndarray[[3], Any])
    assert_type(np.array([[None], [None]]), np.ndarray[[2, 1], Any])
    assert_type(np.array([[1, 2], [3, 4]]), np.ndarray[[2, 2], Any])
    assert_type(np.asarray([[], []]), np.ndarray[[2, 0], Any])
    assert_type(np.asarray(["a", "b"]), np.ndarray[[2], Any])
    assert_type(np.array([1, 2], dtype=np.float32), np.ndarray[[2], Any])


def test_array_and_asarray_list_literals() -> None:
    assert_shape(np.array([1, 2, 3]).shape, (3,))
    assert_shape(np.array([[1, 2], [3, 4]]).shape, (2, 2))
    assert_shape(np.array([[], []]).shape, (2, 0))
    assert_shape(np.asarray([1, 2, 3]).shape, (3,))
    assert_shape(np.asarray([[1, 2], [3, 4]]).shape, (2, 2))
    assert_shape(np.asarray([[], []]).shape, (2, 0))


def check_array_compatibility[DType](
    array: np.ndarray[[2, 3], DType], raw: list[int], dynamic: Any
) -> None:
    assert_type(np.array(array), np.ndarray[[2, 3], DType])
    assert_type(np.asarray(array), np.ndarray[[2, 3], DType])
    assert_type(np.array(array, dtype=np.float32), np.ndarray[[2, 3], Any])
    assert_type(np.asarray(array, dtype=np.float32), np.ndarray[[2, 3], Any])
    assert_type(np.array(raw), np.ndarray[IntTuple, Any])
    assert_type(np.asarray(dynamic), np.ndarray[Any, Any])
    assert_type(np.array([1, 2], like=dynamic), Any)
    assert_type(np.asarray([1, 2], like=dynamic), Any)
    assert_type(np.array([1, 2], like=object()), Any)
    assert_type(np.asarray([1, 2], like=object()), Any)
    assert_type(np.array([1, 2], like=None), np.ndarray[[2], Any])
    assert_type(np.asarray([1, 2], like=None), np.ndarray[[2], Any])
    assert_type(np.array([1, 2], ndmin=0), np.ndarray[[2], Any])
    assert_type(np.array([1, 2], ndmin=2), np.ndarray[IntTuple, Any])


if TYPE_CHECKING:
    assert_type(np.array([[1], [2, 3]]), np.ndarray[IntTuple, Any])
    assert_type(np.asarray([1, [2]]), np.ndarray[IntTuple, Any])


def test_zeros_1d_int_shape() -> None:
    assert_shape(np.zeros(5).shape, (5,))


def test_ones_1d_int_shape() -> None:
    assert_shape(np.ones(4).shape, (4,))


def test_full_1d_int_shape() -> None:
    assert_shape(np.full(3, 7.0).shape, (3,))


def test_empty_1d_int_shape() -> None:
    assert_shape(np.empty(6).shape, (6,))


def test_zeros_tuple_shape() -> None:
    assert_shape(np.zeros((3, 4)).shape, (3, 4))


def test_ones_tuple_shape() -> None:
    assert_shape(np.ones((2, 5)).shape, (2, 5))


def test_full_tuple_shape() -> None:
    assert_shape(np.full((3, 3), -1.0).shape, (3, 3))


def test_empty_tuple_shape() -> None:
    assert_shape(np.empty((6,)).shape, (6,))


def test_eye_square_shape() -> None:
    assert_shape(np.eye(4).shape, (4, 4))


def test_identity_square_shape() -> None:
    assert_shape(np.identity(5).shape, (5, 5))


def test_diag_vector_default_and_offsets() -> None:
    diagonal = np.full(5, 2.0)
    off_diagonal = np.full(4, -1.0)

    assert_shape(np.diag(diagonal).shape, (5, 5))
    assert_shape(np.diag(off_diagonal, 1).shape, (5, 5))
    assert_shape(np.diag(off_diagonal, k=-2).shape, (6, 6))


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
    assert_type(np.diag(array), np.ndarray[IntTuple, DType])


# The gradual fallback also admits ranks NumPy itself rejects; that is the price of keeping
# the dtype for unknown-rank inputs.
def check_diag_rank_zero_falls_back[DType](array: np.ndarray[[], DType]) -> None:
    assert_type(np.diag(array), np.ndarray[IntTuple, DType])


def check_diag_rank_three_falls_back[DType](
    array: np.ndarray[[2, 3, 4], DType],
) -> None:
    assert_type(np.diag(array), np.ndarray[IntTuple, DType])


def test_diag_matrix_runtime_shape() -> None:
    result = np.diag(np.ones((2, 3)))
    assert_type(result, np.ndarray[[int], np.dtype[np.float64]])
    assert result.shape == (2,)


def test_stack_axis0() -> None:
    x = np.zeros(3)
    stacked = np.stack([x, x])
    assert_type(stacked, np.ndarray[[2, 3], Any])
    assert_shape(stacked.shape, (2, 3))


def test_stack_axis1() -> None:
    x = np.zeros((2, 3))
    stacked = np.stack([x, x, x], axis=1)
    assert_type(stacked, np.ndarray[[2, 3, 3], Any])
    assert_shape(stacked.shape, (2, 3, 3))


def test_stack_negative_axis() -> None:
    x = np.zeros(4)
    stacked = np.stack([x, x], axis=-1)
    assert_type(stacked, np.ndarray[[4, 2], Any])
    assert_shape(stacked.shape, (4, 2))


def test_stack_rejects_mismatched_shapes() -> None:
    assert_shape(np.stack([np.zeros(2)]).shape, (1, 2))
    try:
        np.stack([np.zeros(2), np.zeros(3)])  # E: same shape
    except ValueError:
        pass
    else:
        raise AssertionError("expected NumPy to reject mismatched shapes")


def test_stack_rejects_out_of_range_axis() -> None:
    assert_shape(np.stack([np.zeros(2)]).shape, (1, 2))
    try:
        np.stack([np.zeros(2)], axis=2)  # E: axis out of range
    except ValueError:
        pass
    else:
        raise AssertionError("expected NumPy to reject an out-of-range axis")


def check_stack_symbolic[N: IntVar](x: np.ndarray[[N], Any]) -> None:
    assert_type(np.stack([x, x, x]), np.ndarray[[3, N], Any])


class Axis:
    def __index__(self) -> int:
        return 0


def test_stack_array_like_fallback() -> None:
    stacked = np.stack([[1, 2], [3, 4]], axis=Axis(), dtype=np.float64, casting="safe")
    assert_type(stacked, np.ndarray)
    assert_shape(stacked.shape, (2, 2))


def test_stack_out_fallback() -> None:
    out = np.empty((2, 2))
    stacked = np.stack([[1, 2], [3, 4]], out=out, casting="unsafe")
    assert_type(stacked, np.ndarray[[2, 2], np.dtype[np.float64]])
    assert_shape(stacked.shape, (2, 2))
    assert stacked is out
