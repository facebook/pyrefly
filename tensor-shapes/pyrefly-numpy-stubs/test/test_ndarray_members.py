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


def test_ndarray_resize() -> None:
    array = np.ones((2, 3))

    assert_shape(array.shape, (2, 3))

    if TYPE_CHECKING:
        array.resize()  # E: not_usable_with_shape_types
        array.resize((3, 2))  # E: not_usable_with_shape_types


def test_ndarray_view() -> None:
    array = np.ones((2, 3))

    assert_shape(array.view().shape, (2, 3))
    assert_type(array.view(np.int32).dtype, np.dtype[np.int32])
    assert_type(array.view(np.dtype(np.int64)).dtype, np.dtype[np.int64])
    assert array.view(np.uint8).shape == (2, 24)

    if TYPE_CHECKING:
        array.view(42)  # E: No matching overload


def test_ndarray_compress_and_repeat() -> None:
    array = np.arange(6).reshape((2, 3))

    assert_shape(array.compress([True, False, True]).shape, (int,), runtime=(2,))
    assert_shape(array.compress([True, False], axis=0).shape, (int, 3), runtime=(1, 3))
    assert_shape(array.compress([True, False], axis=-1).shape, (2, int), runtime=(2, 1))
    assert_shape(array.compress([], axis=1).shape, (2, int), runtime=(2, 0))
    assert_shape(np.array(7).compress([True], axis=0).shape, (int,), runtime=(1,))
    assert_shape(np.array(7).compress([True], axis=-1).shape, (int,), runtime=(1,))
    assert_type(array.compress([True, False], axis=0).dtype, np.dtype[np.intp])
    row_out = np.empty((1, 3), dtype=np.intp)
    flat_out = np.empty((2,), dtype=np.intp)
    assert_type(
        array.compress([True, False], 0, row_out),
        np.ndarray[[1, 3], np.dtype[np.intp]],
    )
    assert_type(
        array.compress([True, False, True], out=flat_out),
        np.ndarray[[2], np.dtype[np.intp]],
    )

    def dynamic_condition(condition: list[int]) -> None:
        assert_shape(array.compress(condition, axis=1).shape, (2, int))
        assert_shape(array.compress(condition).shape, (int,))

    assert_shape(array.repeat(2).shape, (12,))
    assert_type(array.repeat(2, axis=1).dtype, np.dtype[np.intp])
    assert array.compress([True, False], axis=0).shape == (1, 3)
    assert_shape(array.repeat(2, axis=1).shape, (2, 6))
    assert_shape(array.repeat(3, axis=-2).shape, (6, 3))
    assert_shape(array.repeat([1, 2, 3], axis=1).shape, (2, 6))
    assert_shape(array.repeat((1, 2, 3), axis=-1).shape, (2, 6))
    assert_shape(array.repeat([2], axis=1).shape, (2, 6))
    assert_shape(array.repeat([1, 2, 3, 4, 5, 6]).shape, (21,))
    assert_shape(np.array(7).repeat(2, axis=0).shape, (2,))

    def dynamic_counts(counts: list[int]) -> None:
        assert_shape(array.repeat(counts, axis=1).shape, (2, int))
        assert_shape(array.repeat(counts).shape, (int,))
        assert_shape(array.repeat(np.array(counts), axis=1).shape, (2, int))

    if TYPE_CHECKING:
        array.repeat("twice")  # E: No matching overload
        array.repeat(2, axis=2)  # E: axis is out of bounds
        array.repeat(-1)  # E: repeats may not contain negative values
        array.repeat([1, 2], axis=1)  # E: repeats must match the selected axis
        array.repeat([1, -1, 1], axis=1)  # E: repeats may not contain negative values
        array.compress([1], axis="first")  # E: No matching overload
        array.compress([1], axis=2)  # E: axis out of bounds


def test_ndarray_choose_and_take() -> None:
    indices = np.array([[0, 1], [1, 0]])
    choices = (np.ones((2, 2)), np.zeros((2, 2)))
    selected = indices.choose(choices)

    assert_type(selected, np.ndarray[IntTuple])
    assert selected.shape == (2, 2)
    out = np.empty((2, 2))
    assert_type(
        indices.choose(choices, out=out), np.ndarray[[2, 2], np.dtype[np.float64]]
    )
    assert_shape(indices.take(np.array([[0, 1]])).shape, (1, 2))
    assert_shape(indices.take([0, 1]).shape, (int,), runtime=(2,))
    assert_type(indices.take(0), np.generic)
    assert_type(
        np.ones((2, 2), dtype=np.int32).take(0, axis=1).dtype, np.dtype[np.int32]
    )
    assert_shape(indices.take([0, 1], axis=0).shape, (int, 2), runtime=(2, 2))
    assert_shape(indices.take(np.array([[0, 1]]), axis=-1).shape, (2, 1, 2))
    assert_shape(indices.take(0, axis=1).shape, (2,))
    assert_type(
        indices.take([0, 1], axis=0, out=out), np.ndarray[[2, 2], np.dtype[np.float64]]
    )

    if TYPE_CHECKING:
        indices.choose(choices, mode="invalid")  # E: No matching overload
        indices.take([0], mode="invalid")  # E: No matching overload
        indices.take([0], axis=2)  # E: axis out of bounds


def test_ndarray_diagonal() -> None:
    array = np.ones((2, 3, 4))

    assert_shape(array.diagonal().shape, (4, 2))
    assert_shape(array.diagonal(1, 1, 2).shape, (2, 3))
    assert_shape(array.diagonal(-1, 0, -1).shape, (3, 1))
    assert_shape(array.diagonal(3, 1, 2).shape, (2, 1))
    assert_shape(array.diagonal(5, 1, 2).shape, (2, 0))
    assert_shape(array.diagonal(1, 2, 1).shape, (2, 2))
    assert_type(array.diagonal().dtype, np.dtype[np.float64])

    if TYPE_CHECKING:
        np.ones((3,)).diagonal()  # E: diagonal requires at least two dimensions
        array.diagonal(axis1=0, axis2=0)  # E: diagonal axes must be distinct
        array.diagonal(axis2=3)  # E: diagonal axis out of bounds


def test_ndarray_trace() -> None:
    array = np.ones((2, 3, 4))

    assert_shape(array.trace().shape, (4,))
    assert_shape(array.trace(axis1=1, axis2=-1).shape, (2,))
    assert_type(np.ones((2, 3)).trace(), np.generic)
    assert_type(array.trace(out=np.zeros((4,))), np.ndarray[[4], np.dtype[np.float64]])

    if TYPE_CHECKING:
        array.trace(axis1=0, axis2=0)  # E: diagonal axes must be distinct
        np.ones((3,)).trace()  # E: diagonal requires at least two dimensions


def test_ndarray_dot() -> None:
    vector = np.ones((3,))
    matrix = np.ones((2, 3))
    right = np.ones((3, 4))

    assert_type(vector.dot(vector), np.generic)
    assert_shape(vector.dot(right).shape, (4,))
    assert_shape(matrix.dot(vector).shape, (2,))
    assert_shape(matrix.dot(right).shape, (2, 4))
    assert_shape(np.ones((2, 5, 3)).dot(np.ones((6, 3, 4))).shape, (2, 5, 6, 4))
    assert_shape(matrix.dot(2).shape, (2, 3))
    assert_shape(vector.dot(np.ones(())).shape, (3,))

    if TYPE_CHECKING:
        matrix.dot(np.ones((2, 4)))  # E: dot contraction dimensions must agree
        vector.dot(np.ones((2,)))  # E: dot contraction dimensions must agree


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


def test_ndarray_searchsorted() -> None:
    sorted_array = np.array([1, 3, 5])

    assert_type(sorted_array.searchsorted(3), np.intp)
    assert_shape(sorted_array.searchsorted(np.ones((2, 3))).shape, (2, 3))
    assert_shape(sorted_array.searchsorted([[0, 2], [5, 6]]).shape, (2, 2))
    assert sorted_array.searchsorted(3, side="right") == 2

    if TYPE_CHECKING:
        np.ones((2, 3)).searchsorted(2)  # E: No matching overload
        sorted_array.searchsorted(3, side="middle")  # E: No matching overload


def test_ndarray_reshape() -> None:
    array = np.ones((2, 3))

    assert_shape(array.reshape((3, 2)).shape, (3, 2))
    assert_shape(array.reshape([3, 2]).shape, (3, 2))
    assert_shape(array.reshape(3, 2).shape, (3, 2))
    assert_shape(array.reshape(6).shape, (6,))
    assert_shape(array.reshape((-1, 2)).shape, (3, 2))
    assert_shape(array.reshape(3, -1).shape, (3, 2))
    assert_shape(array.reshape(None).shape, (2, 3))
    assert_shape(array.reshape((3, 2), order="F", copy=True).shape, (3, 2))

    if TYPE_CHECKING:
        array.reshape()  # E: reshape expects at least one dimension
        array.reshape((5, 2))  # E: reshape target element count
        array.reshape((-1, -1))  # E: reshape allows only one inferred dimension
        array.reshape((-2, 3))  # E: reshape dimensions must be at least -1
        array.reshape((0, -1))  # E: reshape cannot infer a dimension


def test_ndarray_mutating_methods() -> None:
    a = np.ones((2, 3))

    assert_type(a.fill(2.0), None)
    assert_type(a.partition(1), None)
    assert_type(a.put([0], [3.0]), None)
    assert_type(a.setfield(1.0, np.float64), None)
    assert_type(a.setflags(write=True), None)
    assert_type(a.sort(), None)
    assert_shape(a.shape, (2, 3))
