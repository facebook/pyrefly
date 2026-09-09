# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type, TYPE_CHECKING

import jax.numpy as jnp
from jax import Array
from shape_extensions import assert_shape, IntTuple


class IndexScalar:
    def __index__(self) -> int:
        return 0


class ArrayLikeIndex:
    @property
    def shape(self) -> tuple[int, ...]:
        return (2,)

    @property
    def dtype(self) -> object:
        return int


def test_basic_indexing() -> None:
    x = jnp.ones((2, 3, 4))

    assert_shape(x[0].shape, (3, 4))
    assert_shape(x[:, 1:].shape, (2, 2, 4))
    assert_shape(x[..., 0].shape, (2, 3))
    assert_shape(x[None, ...].shape, (1, 2, 3, 4))


def test_integer_tuple_indexing() -> None:
    x = jnp.ones((2, 4, 5))

    assert_shape(x[:, (0, 2, 3), :].shape, (2, 3, 5))


def test_indexing_scalar_is_rejected() -> None:
    x = jnp.ones(())
    assert_shape(x.shape, ())
    try:
        x[0]  # E: Cannot index scalar tensor (rank 0)
    except IndexError:
        pass
    else:
        raise AssertionError("expected JAX to reject indexing a scalar")


def test_list_indexing_is_statically_accepted_for_compatibility() -> None:
    x = jnp.ones((2, 3, 4))
    assert_shape(x.shape, (2, 3, 4))

    if TYPE_CHECKING:
        # TODO(stroxler): Preserve the length of list-literal indices in annotations.
        assert_type(x[[0, 1]], Array[IntTuple[int, 3, 4]])
    else:
        try:
            x[[0, 1]]
        except TypeError:
            pass
        else:
            raise AssertionError("expected JAX to reject list indexing at runtime")


def test_gradual_and_fallback_indexing() -> None:
    x = jnp.ones((2, 3, 4))

    array_index = jnp.arange(2)
    if TYPE_CHECKING:
        assert_type(x[array_index], Array[IntTuple])
        assert_type(x[:, array_index], Array[IntTuple])
        assert_type(x[True], Array[IntTuple])
        assert_type(x[:, array_index, (0, 1)], Array[IntTuple])
        assert_type(x[IndexScalar()], Array[IntTuple])
        assert_type(x[ArrayLikeIndex()], Array[IntTuple])
        assert_type(x[[[0, 1], [1, 0]]], Array[IntTuple])
    else:
        assert_shape(x[array_index].shape, (2, 3, 4))
        assert_shape(x[:, array_index].shape, (2, 2, 4))
        assert_shape(x[True].shape, (1, 2, 3, 4))
        assert_shape(x[:, array_index, (0, 1)].shape, (2, 2))
        for index in (IndexScalar(), ArrayLikeIndex(), [[0, 1], [1, 0]]):
            try:
                x[index]
            except (IndexError, TypeError):
                pass
            else:
                raise AssertionError(f"expected JAX to reject {index!r}")


def test_invalid_index() -> None:
    x = jnp.ones((2, 3, 4))
    assert_shape(x.shape, (2, 3, 4))

    try:
        x[0, 0, 0, 0]  # E: Too many indices
    except IndexError:
        pass
    else:
        raise AssertionError("expected JAX to reject too many indices")

    try:
        x[  # E: Cannot index into
            "bad"
        ]
    except (IndexError, TypeError):
        pass
    else:
        raise AssertionError("expected JAX to reject a string index")

    try:
        x[..., ...]  # E: an index may contain at most one ellipsis
    except IndexError:
        pass
    else:
        raise AssertionError("expected JAX to reject multiple ellipses")


def test_take() -> None:
    x = jnp.ones((2, 3, 4))
    idx = jnp.array([[0, 1], [2, 0]])

    # Multi-dimensional indices along axis
    assert_shape(jnp.take(x, idx, axis=1).shape, (2, 2, 2, 4))
    assert_shape(x.take(idx, axis=1).shape, (2, 2, 2, 4))

    # Multi-dimensional indices without axis (flattens input)
    assert_shape(jnp.take(x, idx, axis=None).shape, (2, 2))
    assert_shape(x.take(idx, axis=None).shape, (2, 2))
    assert_shape(jnp.take(x, idx).shape, (2, 2))
    assert_shape(x.take(idx).shape, (2, 2))

    # Scalar index along axis
    assert_shape(jnp.take(x, 1, axis=1).shape, (2, 4))
    assert_shape(x.take(1, axis=1).shape, (2, 4))

    # Scalar index without axis
    assert_shape(jnp.take(x, 1, axis=None).shape, ())
    assert_shape(x.take(1, axis=None).shape, ())
    assert_shape(jnp.take(x, 1).shape, ())
    assert_shape(x.take(1).shape, ())


def test_take_rejects_out_of_bounds_axis() -> None:
    x = jnp.ones((2, 3))
    assert_shape(jnp.take(x, 0, axis=1).shape, (2,))
    try:
        # E: Cannot evaluate type-level shape DSL call: axis out of bounds
        jnp.take(x, 0, axis=2)
    except ValueError:
        pass
    else:
        raise AssertionError("expected JAX to reject an out-of-bounds axis")


def test_take_along_axis() -> None:
    x = jnp.ones((2, 3, 4))
    idx = jnp.zeros((2, 1, 4), dtype=int)

    assert_shape(jnp.take_along_axis(x, idx, axis=1).shape, (2, 1, 4))
    assert_shape(jnp.take_along_axis(x, idx, axis=-2).shape, (2, 1, 4))

    idx_1d = jnp.zeros(5, dtype=int)
    assert_shape(jnp.take_along_axis(x, idx_1d, axis=None).shape, (5,))


def test_take_along_axis_rejects_mismatched_rank() -> None:
    x = jnp.ones((2, 3, 4))
    assert_shape(
        jnp.take_along_axis(x, jnp.zeros((2, 1, 4), dtype=int), axis=1).shape,
        (2, 1, 4),
    )
    try:
        # E: Cannot evaluate type-level shape DSL call: indices and arr must have the same number of dimensions
        jnp.take_along_axis(x, jnp.zeros((2, 1), dtype=int), axis=1)
    except ValueError:
        pass
    else:
        raise AssertionError("expected JAX to reject mismatched index rank")


def test_compress() -> None:
    x = jnp.ones((2, 3, 4))
    cond = jnp.array([True, False, True])

    assert_shape(jnp.compress(cond, x, axis=1, size=2).shape, (2, 2, 4))
    assert_shape(x.compress(cond, axis=1, size=2).shape, (2, 2, 4))

    cond_flat = jnp.array([True, False] * 12)
    assert_shape(jnp.compress(cond_flat, x, size=5).shape, (5,))
    assert_shape(x.compress(cond_flat, size=5).shape, (5,))


def test_compress_rejects_out_of_bounds_axis() -> None:
    x = jnp.ones((2, 3))
    cond = jnp.array([True, False])
    assert_shape(jnp.compress(cond, x, axis=0, size=1).shape, (1, 3))
    try:
        # E: Cannot evaluate type-level shape DSL call: axis out of bounds
        jnp.compress(cond, x, axis=5, size=1)
    except ValueError:
        pass
    else:
        raise AssertionError("expected JAX to reject an out-of-bounds axis")


def test_extract() -> None:
    arr = jnp.arange(6)
    cond = jnp.array([True, False, True, False, True, False])
    assert_shape(jnp.extract(cond, arr, size=3).shape, (3,))


def test_fill_diagonal() -> None:
    a = jnp.zeros((3, 4))
    assert_shape(jnp.fill_diagonal(a, 1, inplace=False).shape, (3, 4))

    b = jnp.zeros((3, 3, 3))
    assert_shape(jnp.fill_diagonal(b, 1, inplace=False).shape, (3, 3, 3))


def test_fill_diagonal_rejects_1d() -> None:
    assert_shape(jnp.fill_diagonal(jnp.zeros((2, 2)), 1, inplace=False).shape, (2, 2))
    a = jnp.zeros(3)
    try:
        # E: Cannot evaluate type-level shape DSL call: array must be at least 2-d
        jnp.fill_diagonal(a, 1, inplace=False)
    except ValueError:
        pass
    else:
        raise AssertionError("expected JAX to reject 1-D array")


def test_place() -> None:
    x = jnp.zeros(5)
    mask = jnp.array([True, False, True, False, False])
    assert_shape(jnp.place(x, mask, 9, inplace=False).shape, (5,))


def test_put() -> None:
    x = jnp.zeros(5)
    ind = jnp.array([0, 2])
    assert_shape(jnp.put(x, ind, 7, inplace=False).shape, (5,))


def test_put_along_axis() -> None:
    y = jnp.zeros((2, 3))
    ind = jnp.array([[0], [1]])
    assert_shape(jnp.put_along_axis(y, ind, 4, axis=1, inplace=False).shape, (2, 3))


def test_diag_indices() -> None:
    idx2 = jnp.diag_indices(4, ndim=2)
    assert len(idx2) == 2
    assert_shape(idx2[0].shape, (4,))
    assert_shape(idx2[1].shape, (4,))

    idx3 = jnp.diag_indices(5, ndim=3)
    assert len(idx3) == 3
    assert_shape(idx3[0].shape, (5,))
    assert_shape(idx3[1].shape, (5,))
    assert_shape(idx3[2].shape, (5,))


def test_diag_indices_from() -> None:
    arr = jnp.zeros((4, 4))
    idx = jnp.diag_indices_from(arr)
    assert len(idx) == 2
    assert_shape(idx[0].shape, (4,))
    assert_shape(idx[1].shape, (4,))


def test_diag_indices_from_rejects_non_square() -> None:
    arr_sq = jnp.zeros((3, 3))
    assert_shape(jnp.diag_indices_from(arr_sq)[0].shape, (3,))
    arr = jnp.zeros((3, 4))
    try:
        # E: Cannot evaluate type-level shape DSL call: All dimensions of input must be of equal length
        jnp.diag_indices_from(arr)
    except ValueError:
        pass
    else:
        raise AssertionError("expected JAX to reject non-square input")


def test_mask_indices() -> None:
    r, c = jnp.mask_indices(4, jnp.triu, 1, size=6)
    assert_shape(r.shape, (6,))
    assert_shape(c.shape, (6,))


def test_tril_and_triu_indices() -> None:
    r, c = jnp.tril_indices(4)
    assert_shape(r.shape, (10,))
    assert_shape(c.shape, (10,))

    a = jnp.zeros((4, 4))
    r2, c2 = jnp.tril_indices_from(a)
    assert_shape(r2.shape, (10,))
    assert_shape(c2.shape, (10,))

    r3, c3 = jnp.triu_indices(4)
    assert_shape(r3.shape, (10,))
    assert_shape(c3.shape, (10,))

    r4, c4 = jnp.triu_indices_from(a)
    assert_shape(r4.shape, (10,))
    assert_shape(c4.shape, (10,))


def test_unravel_index() -> None:
    idx = jnp.array([22, 41, 37])
    res = jnp.unravel_index(idx, (6, 7, 8))
    assert len(res) == 3
    assert_shape(res[0].shape, (3,))
    assert_shape(res[1].shape, (3,))
    assert_shape(res[2].shape, (3,))

    res_scalar = jnp.unravel_index(5, (6, 7))
    assert len(res_scalar) == 2
    assert_shape(res_scalar[0].shape, ())
    assert_shape(res_scalar[1].shape, ())


def test_ravel_multi_index() -> None:
    arrs = (jnp.array([3, 6, 6]), jnp.array([4, 5, 1]))
    res = jnp.ravel_multi_index(arrs, (7, 6))
    assert_shape(res.shape, (3,))


def test_ix() -> None:
    a = jnp.zeros(2)
    b = jnp.zeros(3)
    c = jnp.zeros(4)
    ix = jnp.ix_(a, b, c)
    assert len(ix) == 3
    assert_shape(ix[0].shape, (2, 1, 1))
    assert_shape(ix[1].shape, (1, 3, 1))
    assert_shape(ix[2].shape, (1, 1, 4))


def test_ix_rejects_non_1d() -> None:
    assert_shape(jnp.ix_(jnp.zeros(2))[0].shape, (2,))
    try:
        # E: Cannot evaluate type-level shape DSL call: Arguments to jax.numpy.ix_ must be 1-dimensional
        jnp.ix_(jnp.zeros((2, 2)))
    except ValueError:
        pass
    else:
        raise AssertionError("expected JAX to reject non-1D input")


def test_delete_and_insert() -> None:
    a = jnp.array([1, 2, 3, 4, 5])
    assert_shape(jnp.delete(a, 1).shape, (4,))
    assert_shape(jnp.insert(a, 1, 99).shape, (6,))


def test_trim_zeros() -> None:
    a = jnp.array([0, 0, 1, 2, 0])
    assert_shape(jnp.trim_zeros(a).shape, (2,))
