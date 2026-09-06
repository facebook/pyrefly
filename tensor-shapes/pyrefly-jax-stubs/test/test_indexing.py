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
