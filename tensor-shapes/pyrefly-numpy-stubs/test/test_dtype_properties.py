# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, assert_type

import numpy as np
from shape_extensions import assert_shape


def test_zeros_default_dtype() -> None:
    x = np.zeros(5)

    assert_shape(x, (5,))
    assert_type(x.dtype, np.dtype[np.float64])
    assert x.dtype == np.dtype(np.float64)


def test_zeros_explicit_scalar_dtype() -> None:
    x = np.zeros((2, 3), dtype=np.int32)

    assert_shape(x, (2, 3))
    assert_type(x.dtype, np.dtype[np.int32])
    assert x.dtype == np.dtype(np.int32)


def test_zeros_builtin_type_dtype_falls_back_to_unknown() -> None:
    x = np.zeros(5, dtype=int)

    assert_shape(x, (5,))
    assert_type(x.dtype, Any)
    assert x.dtype == np.dtype(int)


def test_zeros_explicit_dtype_object() -> None:
    x = np.zeros((2, 3), dtype=np.dtype(np.float32))

    assert_shape(x, (2, 3))
    assert_type(x.dtype, np.dtype[np.float32])
    assert x.dtype == np.dtype(np.float32)


def test_zeros_builtin_dtype_object_falls_back_to_unknown() -> None:
    d = np.dtype(int)
    x = np.zeros((2,), dtype=d)

    assert_type(d, np.dtype[Any])
    assert_shape(x, (2,))
    assert_type(x.dtype, Any)
    assert x.dtype == np.dtype(int)


def test_ones_default_dtype() -> None:
    x = np.ones((2, 3))

    assert_shape(x, (2, 3))
    assert_type(x.dtype, np.dtype[np.float64])
    assert x.dtype == np.dtype(np.float64)


def test_ones_explicit_scalar_dtype() -> None:
    x = np.ones(5, dtype=np.bool_)

    assert_shape(x, (5,))
    assert_type(x.dtype, np.dtype[np.bool_])
    assert x.dtype == np.dtype(np.bool_)


def test_ones_tuple_shape_explicit_scalar_dtype() -> None:
    x = np.ones((5,), dtype=np.int64)

    assert_shape(x, (5,))
    assert_type(x.dtype, np.dtype[np.int64])
    assert x.dtype == np.dtype(np.int64)


def test_ones_explicit_dtype_object() -> None:
    x = np.ones(5, dtype=np.dtype(np.bool_))

    assert_shape(x, (5,))
    assert_type(x.dtype, np.dtype[np.bool_])
    assert x.dtype == np.dtype(np.bool_)


def test_ones_tuple_shape_explicit_dtype_object() -> None:
    x = np.ones((4,), dtype=np.dtype(np.float64))

    assert_shape(x, (4,))
    assert_type(x.dtype, np.dtype[np.float64])
    assert x.dtype == np.dtype(np.float64)


def test_empty_default_dtype() -> None:
    x = np.empty((4,))

    assert_shape(x, (4,))
    assert_type(x.dtype, np.dtype[np.float64])
    assert x.dtype == np.dtype(np.float64)


def test_full_omitted_dtype_uses_fill_value_dtype() -> None:
    x = np.full((2, 3), 7)

    assert_shape(x, (2, 3))
    assert_type(x.dtype, Any)
    assert x.dtype == np.dtype(int)


def test_empty_explicit_scalar_dtype() -> None:
    x = np.empty((4,), dtype=np.float32)

    assert_shape(x, (4,))
    assert_type(x.dtype, np.dtype[np.float32])
    assert x.dtype == np.dtype(np.float32)


def test_empty_matrix_explicit_scalar_dtype() -> None:
    x = np.empty((2, 3), dtype=np.bool_)

    assert_shape(x, (2, 3))
    assert_type(x.dtype, np.dtype[np.bool_])
    assert x.dtype == np.dtype(np.bool_)


def test_empty_explicit_dtype_object() -> None:
    x = np.empty((4,), dtype=np.dtype(np.int32))

    assert_shape(x, (4,))
    assert_type(x.dtype, np.dtype[np.int32])
    assert x.dtype == np.dtype(np.int32)


def test_empty_matrix_explicit_dtype_object() -> None:
    x = np.empty((2, 3), dtype=np.dtype(np.int64))

    assert_shape(x, (2, 3))
    assert_type(x.dtype, np.dtype[np.int64])
    assert x.dtype == np.dtype(np.int64)


def test_full_explicit_scalar_dtype() -> None:
    x = np.full((3, 4), 7, dtype=np.int64)

    assert_shape(x, (3, 4))
    assert_type(x.dtype, np.dtype[np.int64])
    assert x.dtype == np.dtype(np.int64)


def test_full_scalar_shape_explicit_scalar_dtype() -> None:
    x = np.full(3, 7, dtype=np.float32)

    assert_shape(x, (3,))
    assert_type(x.dtype, np.dtype[np.float32])
    assert x.dtype == np.dtype(np.float32)


def test_full_explicit_dtype_object() -> None:
    x = np.full((3, 4), 7, dtype=np.dtype(np.float32))

    assert_shape(x, (3, 4))
    assert_type(x.dtype, np.dtype[np.float32])
    assert x.dtype == np.dtype(np.float32)


def test_full_scalar_shape_explicit_dtype_object() -> None:
    x = np.full(3, 7, dtype=np.dtype(np.float32))

    assert_shape(x, (3,))
    assert_type(x.dtype, np.dtype[np.float32])
    assert x.dtype == np.dtype(np.float32)


def test_eye_default_dtype() -> None:
    x = np.eye(3)

    assert_shape(x, (3, 3))
    assert_type(x.dtype, np.dtype[np.float64])
    assert x.dtype == np.dtype(np.float64)


def test_eye_explicit_scalar_dtype() -> None:
    x = np.eye(4, dtype=np.int32)

    assert_shape(x, (4, 4))
    assert_type(x.dtype, np.dtype[np.int32])
    assert x.dtype == np.dtype(np.int32)


def test_eye_explicit_dtype_object() -> None:
    x = np.eye(3, dtype=np.dtype(np.float32))

    assert_shape(x, (3, 3))
    assert_type(x.dtype, np.dtype[np.float32])
    assert x.dtype == np.dtype(np.float32)


def test_eye_positional_scalar_dtype() -> None:
    x = np.eye(4, None, 0, np.int64)

    assert_shape(x, (4, 4))
    assert_type(x.dtype, np.dtype[np.int64])
    assert x.dtype == np.dtype(np.int64)


def test_eye_positional_dtype_object() -> None:
    x = np.eye(3, None, 0, np.dtype(np.bool_))

    assert_shape(x, (3, 3))
    assert_type(x.dtype, np.dtype[np.bool_])
    assert x.dtype == np.dtype(np.bool_)


def test_identity_default_dtype() -> None:
    x = np.identity(4)

    assert_shape(x, (4, 4))
    assert_type(x.dtype, np.dtype[np.float64])
    assert x.dtype == np.dtype(np.float64)


def test_identity_explicit_scalar_dtype() -> None:
    x = np.identity(5, dtype=np.bool_)

    assert_shape(x, (5, 5))
    assert_type(x.dtype, np.dtype[np.bool_])
    assert x.dtype == np.dtype(np.bool_)


def test_identity_explicit_dtype_object() -> None:
    x = np.identity(5, dtype=np.dtype(np.int64))

    assert_shape(x, (5, 5))
    assert_type(x.dtype, np.dtype[np.int64])
    assert x.dtype == np.dtype(np.int64)
