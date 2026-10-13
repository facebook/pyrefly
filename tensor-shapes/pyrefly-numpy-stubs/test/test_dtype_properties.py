# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, assert_type

import numpy as np
from shape_extensions import assert_shape


def check_number_is_not_gradual() -> None:
    assert_type(np.number, type[np.number])


def check_integer_is_not_gradual() -> None:
    assert_type(np.integer, type[np.integer])


def check_floating_hierarchy_is_not_gradual() -> None:
    assert_type(np.inexact, type[np.inexact])
    assert_type(np.floating, type[np.floating])


def test_signed_scalar_hierarchy() -> None:
    assert_shape(np.zeros((2,), dtype=np.int8).shape, (2,))
    assert_type(np.signedinteger, type[np.signedinteger])
    assert issubclass(np.int8, np.signedinteger)
    assert issubclass(np.int32, np.integer)
    assert issubclass(np.intp, np.number)
    assert np.byte is np.int8
    assert np.intc is np.int32
    assert np.short is np.int16


def test_unsigned_scalar_hierarchy() -> None:
    assert_shape(np.zeros((2,), dtype=np.uint16).shape, (2,))
    assert_type(np.unsignedinteger, type[np.unsignedinteger])
    assert issubclass(np.uint8, np.unsignedinteger)
    assert issubclass(np.uintp, np.integer)
    assert np.ubyte is np.uint8
    assert np.uintc is np.uint32
    assert np.ushort is np.uint16


def test_floating_and_complex_scalar_hierarchy() -> None:
    assert_shape(np.ones((2,), dtype=np.complex64).shape, (2,))
    assert_type(np.complexfloating, type[np.complexfloating])
    assert issubclass(np.float16, np.floating)
    assert issubclass(np.float64, np.inexact)
    assert issubclass(np.complex128, np.complexfloating)
    assert issubclass(np.complex64, np.number)
    assert np.half is np.float16
    assert np.single is np.float32
    assert np.cdouble is np.complex128


def test_flexible_and_datetime_scalar_hierarchy() -> None:
    assert_shape(np.zeros((2,), dtype=np.object_).shape, (2,))
    assert_type(np.flexible, type[np.flexible])
    assert issubclass(np.bytes_, np.character)
    assert issubclass(np.str_, np.flexible)
    assert issubclass(np.void, np.generic)
    assert issubclass(np.datetime64, np.generic)
    assert issubclass(np.timedelta64, np.signedinteger)


def test_bool_alias_and_broadcast_iterator() -> None:
    left = np.ones((2, 1))
    right = np.ones((1, 3))
    result = np.broadcast(left, right)

    assert_shape(left.shape, (2, 1))
    assert_type(np.bool, type[np.bool_])
    assert np.bool is np.bool_
    assert_type(result.shape, tuple[int, ...])
    assert result.shape == (2, 3)
    assert result.size == 6
    assert len(next(result)) == 2
    assert_type(result.reset(), None)


def test_ones_supports_other_ranks() -> None:
    assert_shape(np.ones(()).shape, ())
    assert_shape(np.ones((2, 3, 4)).shape, (2, 3, 4))


def test_zeros_default_dtype() -> None:
    x = np.zeros(5)

    assert_shape(x.shape, (5,))
    assert_type(x.dtype, np.dtype[np.float64])
    assert x.dtype == np.dtype(np.float64)


def test_zeros_explicit_scalar_dtype() -> None:
    x = np.zeros((2, 3), dtype=np.int32)

    assert_shape(x.shape, (2, 3))
    assert_type(x.dtype, np.dtype[np.int32])
    assert x.dtype == np.dtype(np.int32)


def test_zeros_builtin_type_dtype_falls_back_to_unknown() -> None:
    x = np.zeros(5, dtype=int)

    assert_shape(x.shape, (5,))
    assert_type(x.dtype, Any)
    assert x.dtype == np.dtype(int)


def test_zeros_explicit_dtype_object() -> None:
    x = np.zeros((2, 3), dtype=np.dtype(np.float32))

    assert_shape(x.shape, (2, 3))
    assert_type(x.dtype, np.dtype[np.float32])
    assert x.dtype == np.dtype(np.float32)


def test_zeros_builtin_dtype_object_falls_back_to_unknown() -> None:
    d = np.dtype(int)
    x = np.zeros((2,), dtype=d)

    assert_type(d, np.dtype[Any])
    assert_shape(x.shape, (2,))
    assert_type(x.dtype, Any)
    assert x.dtype == np.dtype(int)


def test_ones_default_dtype() -> None:
    x = np.ones((2, 3))

    assert_shape(x.shape, (2, 3))
    assert_type(x.dtype, np.dtype[np.float64])
    assert x.dtype == np.dtype(np.float64)


def test_ones_explicit_scalar_dtype() -> None:
    x = np.ones(5, dtype=np.bool_)

    assert_shape(x.shape, (5,))
    assert_type(x.dtype, np.dtype[np.bool_])
    assert x.dtype == np.dtype(np.bool_)


def test_ones_tuple_shape_explicit_scalar_dtype() -> None:
    x = np.ones((5,), dtype=np.int64)

    assert_shape(x.shape, (5,))
    assert_type(x.dtype, np.dtype[np.int64])
    assert x.dtype == np.dtype(np.int64)


def test_ones_explicit_dtype_object() -> None:
    x = np.ones(5, dtype=np.dtype(np.bool_))

    assert_shape(x.shape, (5,))
    assert_type(x.dtype, np.dtype[np.bool_])
    assert x.dtype == np.dtype(np.bool_)


def test_ones_tuple_shape_explicit_dtype_object() -> None:
    x = np.ones((4,), dtype=np.dtype(np.float64))

    assert_shape(x.shape, (4,))
    assert_type(x.dtype, np.dtype[np.float64])
    assert x.dtype == np.dtype(np.float64)


def test_empty_default_dtype() -> None:
    x = np.empty((4,))

    assert_shape(x.shape, (4,))
    assert_type(x.dtype, np.dtype[np.float64])
    assert x.dtype == np.dtype(np.float64)


def test_full_omitted_dtype_uses_fill_value_dtype() -> None:
    x = np.full((2, 3), 7)

    assert_shape(x.shape, (2, 3))
    assert_type(x.dtype, Any)
    assert x.dtype == np.dtype(int)


def test_empty_explicit_scalar_dtype() -> None:
    x = np.empty((4,), dtype=np.float32)

    assert_shape(x.shape, (4,))
    assert_type(x.dtype, np.dtype[np.float32])
    assert x.dtype == np.dtype(np.float32)


def test_empty_matrix_explicit_scalar_dtype() -> None:
    x = np.empty((2, 3), dtype=np.bool_)

    assert_shape(x.shape, (2, 3))
    assert_type(x.dtype, np.dtype[np.bool_])
    assert x.dtype == np.dtype(np.bool_)


def test_empty_explicit_dtype_object() -> None:
    x = np.empty((4,), dtype=np.dtype(np.int32))

    assert_shape(x.shape, (4,))
    assert_type(x.dtype, np.dtype[np.int32])
    assert x.dtype == np.dtype(np.int32)


def test_empty_matrix_explicit_dtype_object() -> None:
    x = np.empty((2, 3), dtype=np.dtype(np.int64))

    assert_shape(x.shape, (2, 3))
    assert_type(x.dtype, np.dtype[np.int64])
    assert x.dtype == np.dtype(np.int64)


def test_full_explicit_scalar_dtype() -> None:
    x = np.full((3, 4), 7, dtype=np.int64)

    assert_shape(x.shape, (3, 4))
    assert_type(x.dtype, np.dtype[np.int64])
    assert x.dtype == np.dtype(np.int64)


def test_full_scalar_shape_explicit_scalar_dtype() -> None:
    x = np.full(3, 7, dtype=np.float32)

    assert_shape(x.shape, (3,))
    assert_type(x.dtype, np.dtype[np.float32])
    assert x.dtype == np.dtype(np.float32)


def test_full_explicit_dtype_object() -> None:
    x = np.full((3, 4), 7, dtype=np.dtype(np.float32))

    assert_shape(x.shape, (3, 4))
    assert_type(x.dtype, np.dtype[np.float32])
    assert x.dtype == np.dtype(np.float32)


def test_full_scalar_shape_explicit_dtype_object() -> None:
    x = np.full(3, 7, dtype=np.dtype(np.float32))

    assert_shape(x.shape, (3,))
    assert_type(x.dtype, np.dtype[np.float32])
    assert x.dtype == np.dtype(np.float32)


def test_eye_default_dtype() -> None:
    x = np.eye(3)

    assert_shape(x.shape, (3, 3))
    assert_type(x.dtype, np.dtype[np.float64])
    assert x.dtype == np.dtype(np.float64)


def test_eye_explicit_scalar_dtype() -> None:
    x = np.eye(4, dtype=np.int32)

    assert_shape(x.shape, (4, 4))
    assert_type(x.dtype, np.dtype[np.int32])
    assert x.dtype == np.dtype(np.int32)


def test_eye_explicit_dtype_object() -> None:
    x = np.eye(3, dtype=np.dtype(np.float32))

    assert_shape(x.shape, (3, 3))
    assert_type(x.dtype, np.dtype[np.float32])
    assert x.dtype == np.dtype(np.float32)


def test_eye_positional_scalar_dtype() -> None:
    x = np.eye(4, None, 0, np.int64)

    assert_shape(x.shape, (4, 4))
    assert_type(x.dtype, np.dtype[np.int64])
    assert x.dtype == np.dtype(np.int64)


def test_eye_positional_dtype_object() -> None:
    x = np.eye(3, None, 0, np.dtype(np.bool_))

    assert_shape(x.shape, (3, 3))
    assert_type(x.dtype, np.dtype[np.bool_])
    assert x.dtype == np.dtype(np.bool_)


def test_identity_default_dtype() -> None:
    x = np.identity(4)

    assert_shape(x.shape, (4, 4))
    assert_type(x.dtype, np.dtype[np.float64])
    assert x.dtype == np.dtype(np.float64)


def test_identity_explicit_scalar_dtype() -> None:
    x = np.identity(5, dtype=np.bool_)

    assert_shape(x.shape, (5, 5))
    assert_type(x.dtype, np.dtype[np.bool_])
    assert x.dtype == np.dtype(np.bool_)


def test_identity_explicit_dtype_object() -> None:
    x = np.identity(5, dtype=np.dtype(np.int64))

    assert_shape(x.shape, (5, 5))
    assert_type(x.dtype, np.dtype[np.int64])
    assert x.dtype == np.dtype(np.int64)
