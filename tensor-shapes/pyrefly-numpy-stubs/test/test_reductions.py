# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, assert_type, cast, Literal, TYPE_CHECKING

import numpy as np
from shape_extensions import assert_raises, assert_shape, Int, IntTuple, IntVar


def make_array(shape: Any) -> Any:
    return np.ones(shape)


def test_reduce_matrix_no_axis() -> None:
    a = np.ones((3, 4))

    assert_shape(np.sum(a).shape, ())
    assert_shape(np.mean(a).shape, ())


def test_extrema_initial_value() -> None:
    a = np.zeros((0, 3))

    assert_shape(a.max(axis=0, initial=0.0).shape, (3,))
    assert_shape(a.min(axis=0, initial=0.0).shape, (3,))
    assert_shape(np.max(a, axis=0, initial=0.0).shape, (3,))
    assert_shape(np.min(a, axis=0, initial=0.0).shape, (3,))
    assert np.array_equal(np.max(a, axis=0, initial=2.0), np.full(3, 2.0))


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
    assert_shape(np.sum(scalar, axis=0).shape, ())
    assert_shape(np.sum(scalar, axis=-1).shape, ())
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

    try:
        # `Flag` keeps bool and int separate, matching NumPy's rejection of boolean axes.
        np.sum(matrix, axis=True)  # E: not a valid `Flag[int | tuple[int, ...] | None]`
    except TypeError:
        pass
    else:
        raise AssertionError("expected NumPy to reject a boolean axis")


if TYPE_CHECKING:

    def check_index_reductions_unknown_rank(
        a: np.ndarray, b: np.ndarray[IntTuple, Any], keepdims: bool
    ) -> None:
        assert_type(a.argmax(axis=0), np.ndarray[IntTuple, np.dtype[np.intp]])
        assert_type(a.argmax(keepdims=True), np.ndarray[IntTuple, np.dtype[np.intp]])
        assert_type(
            a.argmax(axis=0, keepdims=keepdims),
            np.ndarray[IntTuple, np.dtype[np.intp]],
        )
        assert_type(b.argmin(axis=0), np.ndarray[IntTuple, np.dtype[np.intp]])
        assert_type(b.argmin(keepdims=True), np.ndarray[IntTuple, np.dtype[np.intp]])
        assert_type(np.argmax(a, axis=0), np.ndarray[IntTuple, np.dtype[np.intp]])
        assert_type(
            np.argmax(a, keepdims=True), np.ndarray[IntTuple, np.dtype[np.intp]]
        )
        assert_type(
            np.argmax(a, axis=0, keepdims=keepdims),
            np.ndarray[IntTuple, np.dtype[np.intp]],
        )
        assert_type(np.argmin(b, axis=0), np.ndarray[IntTuple, np.dtype[np.intp]])

    def check_symbolic_reductions[D: IntVar, S: IntTuple](
        leading: np.ndarray[[D, *S], Any], trailing: np.ndarray[[*S, D], Any]
    ) -> None:
        assert_type(leading.sum(axis=0), np.ndarray[[*S], Any])
        assert_type(leading.sum(axis=0, keepdims=True), np.ndarray[[1, *S], Any])
        assert_type(trailing.mean(axis=-1), np.ndarray[[*S], Any])


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

    assert_shape(a.sum().shape, ())
    assert_shape(a.mean(axis=-1).shape, (2, 3))
    assert_shape(a.min(axis=1).shape, (2, 4))
    assert_shape(a.max(axis=(0, 2)).shape, (3,))
    assert_shape(a.sum(axis=(1, 2), keepdims=True).shape, (2, 1, 1))
    assert_type(a.max(axis=(0, 2)), np.ndarray[[3], Any])
    assert_type(a.sum(axis=(1, 2), keepdims=True), np.ndarray[[2, 1, 1], Any])
    assert_shape(a.mean(keepdims=True).shape, (1, 1, 1))


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
    assert_type(np.sum(a, axis=(0, 2)), np.ndarray[[3], Any])


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


def test_argmin_3d_axis() -> None:
    result = np.argmin(np.zeros((2, 3, 4)), axis=1)
    assert_type(result, np.ndarray[[2, 4], np.dtype[np.intp]])
    assert_shape(result.shape, (2, 4))


def test_argmin_no_axis_and_keepdims() -> None:
    flat = np.argmin(np.ones((2, 3)))
    assert_type(flat, np.intp)
    assert_shape(flat.shape, ())
    kept = np.argmin(np.ones((2, 3)), axis=0, keepdims=True)
    assert_type(kept, np.ndarray[[1, 3], np.dtype[np.intp]])
    assert_shape(kept.shape, (1, 3))


def test_argmin_argmax_reject_tuple_axis() -> None:
    a = np.ones((3, 4))

    with assert_raises(TypeError):
        np.argmin(a, axis=(0, 1))  # E: No matching overload
    with assert_raises(TypeError):
        a.argmin(axis=(0, 1))  # E: No matching overload
    with assert_raises(TypeError):
        a.argmax(axis=(0, 1))  # E: No matching overload


def test_expand_dims_matrix_negative_axis() -> None:
    a = np.ones((3, 4))

    assert_shape(np.expand_dims(a, axis=-3).shape, (1, 3, 4))
    assert_shape(np.expand_dims(a, axis=-2).shape, (3, 1, 4))
    assert_shape(np.expand_dims(a, axis=-1).shape, (3, 4, 1))


def test_expand_dims_vector() -> None:
    a = np.zeros(4)

    assert_shape(np.expand_dims(a, axis=0).shape, (1, 4))
    assert_shape(np.expand_dims(a, axis=1).shape, (4, 1))
    assert_shape(np.expand_dims(a, axis=-1).shape, (4, 1))


def test_expand_dims_3d() -> None:
    result = np.expand_dims(np.zeros((2, 3, 4)), axis=1)
    assert_type(result, np.ndarray[[2, 1, 3, 4], np.dtype[np.float64]])
    assert_shape(result.shape, (2, 1, 3, 4))
    assert_shape(np.expand_dims(np.zeros((2, 3, 4)), axis=-1).shape, (2, 3, 4, 1))


def test_expand_dims_rejects_out_of_bounds_axis() -> None:
    assert_shape(np.expand_dims(np.zeros((2, 2)), axis=2).shape, (2, 2, 1))
    try:
        np.expand_dims(np.zeros((2, 2)), axis=3)  # E: axis out of bounds
    except ValueError:
        pass
    else:
        raise AssertionError("expected NumPy to reject an out-of-bounds axis")


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
    assert_type(np.sum(a, axis=(1, 2), keepdims=True), np.ndarray[[2, 1, 1], Any])
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


def test_argmax_axes_and_dtype() -> None:
    a = np.arange(24).reshape(2, 3, 4)
    result = np.argmax(a, axis=1)
    assert_type(result, np.ndarray[[2, 4], np.dtype[np.intp]])
    assert_shape(result.shape, (2, 4))
    assert_shape(np.argmax(a, axis=-1).shape, (2, 3))
    assert_shape(np.argmax(a, axis=-3).shape, (3, 4))
    assert result.dtype == np.dtype(np.intp)
    assert np.array_equal(result, np.full((2, 4), 2))


def test_argmax_default_and_explicit_none() -> None:
    a = np.arange(6).reshape(2, 3)
    omitted = np.argmax(a)
    explicit = np.argmax(a, axis=None, keepdims=False)
    assert_type(omitted, np.intp)
    assert_type(explicit, np.intp)
    assert_shape(omitted.shape, ())
    assert_shape(explicit.shape, ())
    assert isinstance(omitted, np.intp)
    assert omitted == explicit == 5


def test_argmax_keepdims() -> None:
    a = np.ones((2, 3, 4))
    flattened = np.argmax(a, keepdims=True)
    columns = np.argmax(a, axis=-2, keepdims=True)
    assert_type(flattened, np.ndarray[[1, 1, 1], np.dtype[np.intp]])
    assert_type(columns, np.ndarray[[2, 1, 4], np.dtype[np.intp]])
    assert_shape(flattened.shape, (1, 1, 1))
    assert_shape(columns.shape, (2, 1, 4))


def test_argmax_methods_and_sequence_input() -> None:
    a = np.ones((2, 3))
    assert_type(a.argmax(axis=1), np.ndarray[[2], np.dtype[np.intp]])
    assert_type(a.argmin(axis=0), np.ndarray[[3], np.dtype[np.intp]])
    assert_shape(a.argmax(axis=1).shape, (2,))
    assert_shape(a.argmin(axis=0).shape, (3,))
    assert np.argmax([[1, 3, 2], [4, 2, 1]], axis=1).shape == (2,)
    assert np.argmin([[1, 3, 2], [4, 2, 1]], axis=1).shape == (2,)


def test_argmax_invalid_axes() -> None:
    a = np.ones((2, 3))
    with assert_raises(TypeError):
        np.argmax(a, axis=(0, 1))  # E: No matching overload
    with assert_raises(TypeError):
        np.argmax(a, axis=True)  # E: No matching overload
    with assert_raises(ValueError):
        np.argmax(a, axis=2)  # E: axis out of bounds
    with assert_raises(ValueError):
        np.argmax(a, axis=-3)  # E: axis out of bounds


def test_index_reductions_out_keyword() -> None:
    a = np.arange(6).reshape(2, 3)
    out = np.zeros(2, dtype=np.int32)
    largest = np.argmax(a, axis=1, out=out)
    assert_type(largest, np.ndarray[[2], np.dtype[np.int32]])
    assert largest is out
    assert_shape(largest.shape, (2,))
    assert np.array_equal(out, [2, 2])
    smallest = np.argmin(a, axis=1, out=out)
    assert_type(smallest, np.ndarray[[2], np.dtype[np.int32]])
    assert smallest is out
    assert_shape(smallest.shape, (2,))
    assert np.array_equal(out, [0, 0])


def test_index_reductions_out_positional_and_none() -> None:
    a = np.arange(6).reshape(2, 3)
    out = np.zeros((2, 1), dtype=np.int32)
    result = np.argmax(a, 1, out, keepdims=True)
    assert_type(result, np.ndarray[[2, 1], np.dtype[np.int32]])
    assert result is out
    assert_shape(result.shape, (2, 1))
    result = np.argmin(a, 1, out, keepdims=True)
    assert_type(result, np.ndarray[[2, 1], np.dtype[np.int32]])
    assert result is out
    assert_shape(result.shape, (2, 1))
    explicit_none = np.argmax(a, axis=1, out=None)
    assert_type(explicit_none, np.ndarray[[2], np.dtype[np.intp]])
    assert_shape(explicit_none.shape, (2,))
    positional_none = np.argmin(a, 1, None)
    assert_type(positional_none, np.ndarray[[2], np.dtype[np.intp]])
    assert_shape(positional_none.shape, (2,))
    scalar_out = np.zeros((), dtype=np.intp)
    assert np.argmax(a, out=scalar_out) is scalar_out
    assert_shape(scalar_out.shape, ())
    assert np.argmin([3, 1, 2], out=scalar_out) is scalar_out
    list_result = np.argmax([3, 1, 2], None, scalar_out)
    assert_type(list_result, np.ndarray[[], np.dtype[np.intp]])
    assert list_result is scalar_out


class ArrayConvertible:
    def __array__(self, dtype: object = None) -> np.ndarray:
        return np.ones((2, 3))


def test_index_reductions_scalar_and_array_convertible_inputs() -> None:
    scalar = np.argmax(7)
    assert_type(scalar, np.intp)
    assert scalar == 0
    scalar = np.argmin(np.float64(7))
    assert_type(scalar, np.intp)
    assert scalar == 0
    converted = np.argmax(ArrayConvertible(), axis=1)  # E: No matching overload
    assert converted.shape == (2,)
    assert_shape(np.ones((2, 3)).shape, (2, 3))


def test_index_reductions_vector_and_scalar_results() -> None:
    vector = np.arange(3)
    scalar = np.ones(())
    positive = np.argmax(vector, axis=0)
    negative = np.argmin(vector, axis=-1)
    zero_rank = np.argmax(scalar, axis=0)
    # Explicit-axis scalar results still need rank-sensitive result modeling.
    assert_type(positive, np.ndarray[[], np.dtype[np.intp]])
    assert_type(negative, np.ndarray[[], np.dtype[np.intp]])
    assert_type(zero_rank, np.ndarray[[], np.dtype[np.intp]])
    assert_shape(positive.shape, ())
    assert_shape(negative.shape, ())
    assert_shape(zero_rank.shape, ())
    zero_rank_kept = np.argmin(scalar, axis=-1, keepdims=True)
    assert_type(zero_rank_kept, np.ndarray[[], np.dtype[np.intp]])
    assert isinstance(zero_rank_kept, np.intp)
    assert_shape(zero_rank_kept.shape, ())
    assert isinstance(positive, np.intp)
    assert isinstance(negative, np.intp)
    assert isinstance(zero_rank, np.intp)
    assert positive == 2
    assert negative == zero_rank == 0
    method_default = vector.argmax()
    method_axis = vector.argmin(axis=0)
    assert_type(method_default, np.intp)
    assert_type(method_axis, np.ndarray[[], np.dtype[np.intp]])
    assert isinstance(method_default, np.intp)
    assert isinstance(method_axis, np.intp)
    assert_shape(method_default.shape, ())
    assert_shape(method_axis.shape, ())
    kept = np.argmax(vector, axis=0, keepdims=True)
    assert_type(kept, np.ndarray[[1], np.dtype[np.intp]])
    assert_shape(kept.shape, (1,))
    scalar_out = np.zeros((), dtype=np.int32)
    result = np.argmax(vector, 0, scalar_out)
    assert_type(result, np.ndarray[[], np.dtype[np.int32]])
    assert result is scalar_out
    assert_shape(result.shape, ())
    with assert_raises(ValueError):
        np.argmax(vector, axis=1)  # E: axis out of bounds


if TYPE_CHECKING:

    def check_argmax_gradual_and_symbolic[N: IntVar](
        a: np.ndarray[[N, 3], Any], axis: int, keepdims: bool
    ) -> None:
        assert_type(np.argmax(a, axis=1), np.ndarray[[N], np.dtype[np.intp]])
        assert_type(np.argmax(a, axis=axis), np.ndarray[IntTuple, np.dtype[np.intp]])
        assert_type(
            np.argmax(a, axis=1, keepdims=keepdims),
            np.ndarray[IntTuple, np.dtype[np.intp]],
        )
