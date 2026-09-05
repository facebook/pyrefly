# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, assert_type, cast, Literal, TYPE_CHECKING

import numpy as np
from shape_extensions import assert_shape, Int, IntTuple, IntVar


def make_array(shape: Any) -> Any:
    return np.ones(shape)


def test_reduce_matrix_no_axis() -> None:
    a = np.ones((3, 4))

    assert_shape(np.sum(a).shape, ())
    assert_shape(np.mean(a).shape, ())


def test_reduce_higher_rank_no_axis() -> None:
    a = cast(
        "np.ndarray[tuple[Literal[2], Literal[3], Literal[4]]]",
        make_array((2, 3, 4)),
    )

    assert_shape(np.sum(a).shape, ())
    assert_shape(np.mean(a, keepdims=True).shape, (1, 1, 1))
    assert_shape(np.max(a, keepdims=True).shape, (1, 1, 1))


def test_reduce_scalar_and_empty_axis_tuple() -> None:
    scalar = cast("np.ndarray[tuple[()]]", make_array(()))
    matrix = np.ones((3, 4))

    assert_shape(np.sum(scalar).shape, ())
    assert_shape(np.mean(scalar, keepdims=True).shape, ())
    assert_shape(np.sum(matrix, axis=()).shape, (3, 4))
    assert_shape(np.sum(matrix, axis=(), keepdims=True).shape, (3, 4))

    try:
        # E: axis out of bounds
        np.sum(scalar, axis=(0,))
    except ValueError:
        pass
    else:
        raise AssertionError("expected NumPy to reject an axis for a scalar")

    if TYPE_CHECKING:
        # The shared static rule rejects integer axes for rank-zero arrays.
        np.sum(scalar, axis=0)  # E: axis out of bounds

    try:
        # `Flag` keeps bool and int separate, matching NumPy's rejection of boolean axes.
        np.sum(matrix, axis=True)  # E: not a valid `Flag[int | tuple[int, ...] | None]`
    except TypeError:
        pass
    else:
        raise AssertionError("expected NumPy to reject a boolean axis")


def test_reduce_matrix_axis_zero() -> None:
    a = np.ones((3, 4))

    assert_shape(np.sum(a, axis=0).shape, (4,))
    assert_shape(np.mean(a, axis=0).shape, (4,))
    assert_shape(np.min(a, axis=0).shape, (4,))
    assert_shape(np.max(a, axis=0).shape, (4,))


def test_mean_method_axis_zero_broadcasts_over_matrix() -> None:
    a = np.ones((3, 4))
    column_means = a.mean(axis=0)

    assert_shape(column_means.shape, (4,))
    assert_shape((a - column_means).shape, (3, 4))


def test_method_reductions_for_cross_entropy() -> None:
    logits = np.ones((5, 3))
    shifted = logits - logits.max(axis=1, keepdims=True)
    normalizers = np.exp(shifted).sum(axis=1, keepdims=True)
    row_losses = np.ones(5)
    loss = row_losses.mean()

    assert_shape(logits.max(axis=1, keepdims=True).shape, (5, 1))
    assert_shape(shifted.shape, (5, 3))
    assert_shape(normalizers.shape, (5, 1))
    assert_shape(loss.shape, ())


def test_method_reductions_share_free_function_shapes() -> None:
    a = cast(
        "np.ndarray[tuple[Literal[2], Literal[3], Literal[4]]]",
        make_array((2, 3, 4)),
    )

    assert_shape(a.sum(), ())
    assert_shape(a.mean(axis=-1), (2, 3))
    assert_shape(a.min(axis=1), (2, 4))
    assert_shape(a.max(axis=(0, 2)), (3,))
    assert_shape(a.sum(axis=(1, 2), keepdims=True), (2, 1, 1))
    assert_shape(a.mean(keepdims=True), (1, 1, 1))


def test_reduce_matrix_axis_one() -> None:
    a = np.ones((3, 4))

    assert_shape(np.sum(a, axis=1).shape, (3,))
    assert_shape(np.mean(a, axis=1).shape, (3,))
    assert_shape(np.min(a, axis=1).shape, (3,))
    assert_shape(np.max(a, axis=1).shape, (3,))


def test_reduce_higher_rank_axis() -> None:
    a = cast(
        "np.ndarray[tuple[Literal[2], Literal[3], Literal[4]]]",
        make_array((2, 3, 4)),
    )

    assert_shape(np.sum(a, axis=1).shape, (2, 4))
    assert_shape(np.mean(a, axis=-1).shape, (2, 3))
    assert_shape(np.sum(a, axis=(0, 2)).shape, (3,))


def test_method_sum_3d_axis_one_for_nbody() -> None:
    forces = cast(
        "np.ndarray[tuple[Literal[5], Literal[5], Literal[3]]]",
        make_array((5, 5, 3)),
    )

    assert_shape(forces.sum(axis=1).shape, (5, 3))


def test_argmin_matrix_axis() -> None:
    a = np.ones((3, 4))
    labels = np.argmin(a, axis=-1)

    assert_shape(np.argmin(a, axis=0).shape, (4,))
    assert_shape(np.argmin(a, axis=1).shape, (3,))
    assert_shape(np.argmin(a, axis=-2).shape, (4,))
    assert_shape(labels.shape, (3,))
    assert_type(labels.dtype, np.dtype[np.intp])
    assert labels.dtype == np.dtype(np.intp)


def test_expand_dims_matrix_negative_axis() -> None:
    a = np.ones((3, 4))

    assert_shape(np.expand_dims(a, axis=-3).shape, (1, 3, 4))
    assert_shape(np.expand_dims(a, axis=-2).shape, (3, 1, 4))
    assert_shape(np.expand_dims(a, axis=-1).shape, (3, 4, 1))


def test_nearest_centroid_assignment_shapes() -> None:
    points = np.ones((5, 3))
    centroids = np.ones((4, 3))

    point_vectors = np.expand_dims(points, axis=-2)
    centroid_vectors = np.expand_dims(centroids, axis=-3)
    deltas = point_vectors - centroid_vectors
    squared_distances = np.sum(deltas**2, axis=-1)
    labels = np.argmin(squared_distances, axis=-1)

    assert_shape(point_vectors.shape, (5, 1, 3))
    assert_shape(centroid_vectors.shape, (1, 4, 3))
    assert_shape(deltas.shape, (5, 4, 3))
    assert_shape(squared_distances.shape, (5, 4))
    assert_shape(labels.shape, (5,))


def test_reduce_rejects_invalid_axes() -> None:
    a = np.ones((3, 4))

    assert_shape(np.sum(a, axis=0).shape, (4,))
    try:
        # E: axis out of bounds
        np.sum(a, axis=3)
    except ValueError:
        pass
    else:
        raise AssertionError("expected NumPy to reject out-of-bounds axis")

    try:
        # E: duplicate axis
        np.sum(a, axis=(0, 0))
    except ValueError:
        pass
    else:
        raise AssertionError("expected NumPy to reject duplicate axes")

    try:
        # E: duplicate axis
        np.sum(a, axis=(0, -2))
    except ValueError:
        pass
    else:
        raise AssertionError("expected NumPy to reject normalized duplicate axes")


def test_reduce_matrix_axis_zero_keepdims() -> None:
    a = np.ones((3, 4))

    assert_shape(np.sum(a, axis=0, keepdims=True).shape, (1, 4))
    assert_shape(np.mean(a, axis=0, keepdims=True).shape, (1, 4))
    assert_shape(np.min(a, axis=0, keepdims=True).shape, (1, 4))
    assert_shape(np.max(a, axis=0, keepdims=True).shape, (1, 4))


def test_reduce_matrix_axis_one_keepdims() -> None:
    a = np.ones((3, 4))

    assert_shape(np.sum(a, axis=1, keepdims=True).shape, (3, 1))
    assert_shape(np.mean(a, axis=1, keepdims=True).shape, (3, 1))
    assert_shape(np.min(a, axis=1, keepdims=True).shape, (3, 1))
    assert_shape(np.max(a, axis=1, keepdims=True).shape, (3, 1))


def test_reduce_higher_rank_keepdims() -> None:
    a = cast(
        "np.ndarray[tuple[Literal[2], Literal[3], Literal[4]]]",
        make_array((2, 3, 4)),
    )

    assert_shape(np.sum(a, axis=(1, 2), keepdims=True).shape, (2, 1, 1))
    assert_shape(np.min(a, axis=0, keepdims=True).shape, (1, 3, 4))


def test_keepdims_reduction_broadcasts_over_matrix() -> None:
    a = np.ones((3, 4))
    row_totals = np.sum(a, axis=1, keepdims=True)

    assert_shape((a / row_totals).shape, (3, 4))


def check_generic_reduction_flags[N: IntVar](
    a: np.ndarray[[N, 3], np.dtype[np.float32]],
    gradual: np.ndarray[[int, 3], np.dtype[np.float32]],
    axis: int,
    axes: tuple[int, ...],
    keepdims: bool,
) -> None:
    assert_type(np.sum(a, axis=1), np.ndarray[[N], Any])
    assert_type(np.mean(a, axis=1), np.ndarray[[N], Any])
    assert_type(np.min(a, axis=1), np.ndarray[[N], Any])
    assert_type(np.max(a, axis=1), np.ndarray[[N], Any])
    assert_type(np.sum(gradual, axis=1), np.ndarray[[int], Any])
    assert_type(np.sum(a, axis=axis), np.ndarray[IntTuple, Any])
    assert_type(np.sum(a, axis=axes), np.ndarray[IntTuple, Any])
    assert_type(np.sum(a, axis=1, keepdims=keepdims), np.ndarray[IntTuple, Any])
    assert_type(a.sum(), np.ndarray[[], np.dtype[np.float32]])
    assert_type(a.mean(axis=1), np.ndarray[[N], np.dtype[np.float32]])
    assert_type(a.min(axis=1), np.ndarray[[N], np.dtype[np.float32]])
    assert_type(a.max(axis=0, keepdims=True), np.ndarray[[1, 3], np.dtype[np.float32]])
    assert_type(a.sum(axis=axis), np.ndarray[IntTuple, np.dtype[np.float32]])
    assert_type(
        a.mean(axis=axes, keepdims=keepdims),
        np.ndarray[IntTuple, np.dtype[np.float32]],
    )
    if TYPE_CHECKING:
        assert_type(
            np.sum(a, axis=999999999999999999999999),
            np.ndarray[IntTuple, Any],
        )
        assert_type(
            np.sum(a, axis=-999999999999999999999999),
            np.ndarray[IntTuple, Any],
        )


def check_union_reduction_flags(
    a: np.ndarray[[2, 3]],
    axis: Literal[0, 1],
    keepdims: Literal[True, False],
) -> None:
    assert_type(np.sum(a, axis=axis), np.ndarray[IntTuple, Any])
    assert_type(np.sum(a, axis=0, keepdims=keepdims), np.ndarray[IntTuple, Any])


def check_symbolic_axis_becomes_gradual[N: IntVar](
    a: np.ndarray[[N, 3]], axis: Int[N]
) -> None:
    assert_type(np.sum(a, axis=axis), np.ndarray[IntTuple, Any])
