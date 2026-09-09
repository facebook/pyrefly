# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, assert_type, Literal, TYPE_CHECKING

import numpy as np
from shape_extensions import assert_shape


def test_ndarray_properties_and_shape_preserving_methods() -> None:
    a = np.ones((2, 3))

    assert_type(a.ndim, int)
    assert_type(a.size, int)
    assert_type(a.itemsize, int)
    assert_type(a.nbytes, int)
    assert_type(a.strides, tuple[int, ...])
    assert_type(a.data, memoryview)
    assert_type(a.device, Literal["cpu"])
    assert_type(a.base, Any)
    assert_shape(a.__array__().shape, (2, 3))
    assert_shape(a.__array__(dtype=np.float32, copy=True).shape, (2, 3))
    assert_type(a.__buffer__(0), memoryview)
    assert not hasattr(a, "__round__")
    try:
        round(a)  # E: requires attribute `__round__`
    except TypeError:
        pass
    else:
        raise AssertionError("round(ndarray) unexpectedly succeeded")
    assert_shape(a.real.shape, (2, 3))
    assert_shape(a.imag.shape, (2, 3))
    converted = a.astype(np.int32)
    assert_shape(converted.shape, (2, 3))
    assert_type(converted.dtype, np.dtype[np.int32])
    assert_shape(a.byteswap().shape, (2, 3))
    assert_shape(a.clip(0.0, 1.0).shape, (2, 3))
    assert_shape(a.conjugate().shape, (2, 3))
    assert_shape(a.copy().shape, (2, 3))
    assert_shape(a.round().shape, (2, 3))
    assert_shape(a.getfield(np.float64).shape, (2, 3))
    assert_shape(a.to_device("cpu").shape, (2, 3))
    assert_type(a.item(0), Any)
    assert_type(a.dumps(), bytes)
    assert_type(a.tobytes(), bytes)

    if TYPE_CHECKING:
        assert_type(a.__array_interface__, Any)
        a.__array_interface__ = {}  # E: read-only property
        a.base = None  # E: read-only property
        a.data = memoryview(b"")  # E: read-only property
        a.device = "cpu"  # E: read-only property
        a.itemsize = 1  # E: read-only property
        a.nbytes = 1  # E: read-only property
        a.ndim = 1  # E: read-only property
        a.size = 1  # E: read-only property
        a.real = np.zeros((2, 3))
        a.imag = np.zeros((2, 3))


def test_ndarray_reduction_methods() -> None:
    a = np.ones((2, 3))

    assert_shape(a.all(axis=0).shape, (3,))
    assert_shape(a.any(axis=1, keepdims=True).shape, (2, 1))
    assert_shape(a.argmax(axis=0).shape, (3,))
    assert_shape(a.argmin(axis=1).shape, (2,))
    assert_shape(a.prod(axis=0).shape, (3,))
    assert_shape(a.std(axis=1).shape, (2,))
    assert_shape(a.var(keepdims=True).shape, (1, 1))
    assert_shape(a.cumsum(axis=0).shape, (2, 3))
    assert_type(a.cumprod(), np.ndarray[[int]])


def test_ndarray_ordering_and_flattening_methods() -> None:
    a = np.ones((2, 3))

    assert_shape(a.argsort().shape, (2, 3))
    assert_shape(a.argpartition(1).shape, (2, 3))
    assert_type(a.argsort(None), np.ndarray[[int], np.dtype[np.intp]])
    assert_type(a.argpartition(1, None), np.ndarray[[int], np.dtype[np.intp]])
    assert_type(a.flatten(), np.ndarray[[int], np.dtype[np.float64]])
    assert_type(a.ravel(), np.ndarray[[int], np.dtype[np.float64]])
    assert_shape(a.shape, (2, 3))


def test_ndarray_mutating_methods() -> None:
    a = np.ones((2, 3))

    assert_type(a.fill(2.0), None)
    assert_type(a.partition(1), None)
    assert_type(a.put([0], [3.0]), None)
    assert_type(a.resize, Any)
    assert_type(a.setfield(1.0, np.float64), None)
    assert_type(a.setflags(write=True), None)
    assert_type(a.sort(), None)
    assert_shape(a.shape, (2, 3))
