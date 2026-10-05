# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, assert_type, Literal, TYPE_CHECKING

import numpy as np
from shape_extensions import assert_shape, IntTuple, IntVar


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
    assert_type(a.flags.f_contiguous, bool)
    assert_type(a.flags["W"], bool)

    if TYPE_CHECKING:
        assert_type(a.__array_interface__, Any)
        a.__array__(copy="yes")  # E: Argument `Literal['yes']` is not assignable
        a.__array_interface__ = {}  # E: read-only property
        assert_type(a.tofile(object()), None)
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


def test_ndarray_matrix_transpose_and_swapaxes() -> None:
    array = np.ones((2, 3, 4))

    assert_shape(array.mT.shape, (2, 4, 3))
    assert_shape(array.swapaxes(0, -1).shape, (4, 3, 2))
    assert_shape(array.swapaxes(1, 1).shape, (2, 3, 4))
    assert_type(array.mT.dtype, np.dtype[np.float64])

    if TYPE_CHECKING:
        np.ones((3,)).mT  # E: swapaxes axis out of bounds
        array.swapaxes(0, 3)  # E: swapaxes axis out of bounds


def test_ndarray_transpose() -> None:
    array = np.ones((2, 3, 4))

    assert_shape(array.transpose().shape, (4, 3, 2))
    assert_shape(array.transpose(None).shape, (4, 3, 2))
    assert_shape(array.transpose((2, 0, 1)).shape, (4, 2, 3))
    assert_shape(array.transpose(1, 2, 0).shape, (3, 4, 2))
    assert_shape(array.transpose((-1, 0, 1)).shape, (4, 2, 3))

    if TYPE_CHECKING:
        array.transpose((0, 1))  # E: transpose axes must match the array rank
        array.transpose(0, 0, 1)  # E: transpose axes must be unique
        array.transpose((0, 1, 3))  # E: transpose axis out of bounds
        array.transpose(())  # E: transpose axes must match the array rank


def test_ndarray_nonzero_indices() -> None:
    indices = np.ones((2, 3)).nonzero()

    assert_type(
        indices,
        tuple[
            np.ndarray[[int], np.dtype[np.intp]], np.ndarray[[int], np.dtype[np.intp]]
        ],
    )
    assert_shape(indices[0].shape, (int,), runtime=(6,))
    assert_shape(indices[1].shape, (int,), runtime=(6,))

    if TYPE_CHECKING:
        np.ones(()).nonzero()  # E: nonzero requires at least one dimension


def test_ndarray_flat_iterator_and_ctypes() -> None:
    array = np.ones((2, 3))

    assert_type(array.ctypes.data, int)
    assert isinstance(array.flat, np.flatiter)
    assert_shape(array.flat.base.shape, (2, 3))
    assert array.flat[0] == 1.0

    if TYPE_CHECKING:
        assert_type(array.flat, np.flatiter[np.ndarray[[2, 3], np.dtype[np.float64]]])
        array.ctypes = None  # E: read-only property
        array.flat = None  # E: read-only property


def test_ndarray_squeeze() -> None:
    array = np.ones((2, 1, 3, 1))

    assert_shape(array.squeeze().shape, (2, 3))
    assert_shape(array.squeeze(1).shape, (2, 3, 1))
    assert_shape(array.squeeze(axis=1).shape, (2, 3, 1))
    assert_shape(array.squeeze((1, -1)).shape, (2, 3))
    assert_shape(np.ones(()).squeeze(0).shape, ())
    assert_shape(np.ones(()).squeeze(axis=-1).shape, ())
    assert_shape(np.ones((1,)).squeeze().shape, ())
    assert_shape(array.squeeze(()).shape, (2, 1, 3, 1))

    if TYPE_CHECKING:
        array.squeeze(4)  # E: squeeze axis out of bounds
        array.squeeze(2)  # E: squeeze axis must have length 1
        array.squeeze((1, 1))  # E: squeeze axes must be unique
        np.ones((1,)).squeeze(-2)  # E: squeeze axis out of bounds

        def check_squeeze[Batch: IntTuple, N: IntVar](
            value: np.ndarray[[*Batch, 1, N]],
        ) -> None:
            assert_type(value.squeeze(-2), np.ndarray[[*Batch, N]])


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
