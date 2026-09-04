# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import jax.numpy as jnp
from shape_extensions import assert_shape


def test_sort() -> None:
    x = jnp.ones((2, 3, 4))

    assert_shape(jnp.sort(x), (2, 3, 4))
    assert_shape(jnp.sort(x, axis=0), (2, 3, 4))
    assert_shape(jnp.sort(x, axis=1), (2, 3, 4))
    assert_shape(jnp.sort(x, axis=-1), (2, 3, 4))
    assert_shape(jnp.sort(x, axis=-2), (2, 3, 4))
    assert_shape(jnp.sort(x, axis=None), (24,))

    # Method
    assert_shape(x.sort(), (2, 3, 4))
    assert_shape(x.sort(axis=0), (2, 3, 4))
    assert_shape(x.sort(axis=None), (24,))


def test_sort_rejects_out_of_bounds_axis() -> None:
    x = jnp.ones((2, 3))
    assert_shape(jnp.sort(x, axis=1), (2, 3))
    try:
        # E: Cannot evaluate type-level shape DSL call: axis out of bounds
        jnp.sort(x, axis=2)
    except ValueError:
        pass
    else:
        raise AssertionError("expected JAX to reject an out-of-bounds axis")


def test_argsort() -> None:
    x = jnp.ones((2, 3, 4))

    assert_shape(jnp.argsort(x), (2, 3, 4))
    assert_shape(jnp.argsort(x, axis=0), (2, 3, 4))
    assert_shape(jnp.argsort(x, axis=1), (2, 3, 4))
    assert_shape(jnp.argsort(x, axis=-1), (2, 3, 4))
    assert_shape(jnp.argsort(x, axis=None), (24,))

    # Method
    assert_shape(x.argsort(), (2, 3, 4))
    assert_shape(x.argsort(axis=0), (2, 3, 4))
    assert_shape(x.argsort(axis=None), (24,))


def test_argsort_rejects_out_of_bounds_axis() -> None:
    x = jnp.ones((2, 3))
    assert_shape(jnp.argsort(x, axis=1), (2, 3))
    try:
        # E: Cannot evaluate type-level shape DSL call: axis out of bounds
        jnp.argsort(x, axis=2)
    except ValueError:
        pass
    else:
        raise AssertionError("expected JAX to reject an out-of-bounds axis")


def test_sort_complex() -> None:
    x = jnp.ones((2, 3), dtype=jnp.complex64)
    assert_shape(jnp.sort_complex(x), (2, 3))


def test_partition() -> None:
    x = jnp.ones((3, 5))

    assert_shape(jnp.partition(x, 2), (3, 5))
    assert_shape(jnp.partition(x, 1, axis=0), (3, 5))
    assert_shape(jnp.partition(x, 1, axis=-1), (3, 5))


def test_partition_rejects_out_of_bounds_axis() -> None:
    x = jnp.ones((2, 3))
    assert_shape(jnp.partition(x, 1, axis=1), (2, 3))
    try:
        # E: Cannot evaluate type-level shape DSL call: axis out of bounds
        jnp.partition(x, 1, axis=2)
    except ValueError:
        pass
    else:
        raise AssertionError("expected JAX to reject an out-of-bounds axis")


def test_argpartition() -> None:
    x = jnp.ones((3, 5))

    assert_shape(jnp.argpartition(x, 2), (3, 5))
    assert_shape(jnp.argpartition(x, 1, axis=0), (3, 5))
    assert_shape(jnp.argpartition(x, 1, axis=-1), (3, 5))

    # Method
    assert_shape(x.argpartition(2), (3, 5))
    assert_shape(x.argpartition(1, axis=0), (3, 5))


def test_lexsort() -> None:
    a = jnp.ones(4)
    b = jnp.ones(4)
    assert_shape(jnp.lexsort((b, a)), (4,))

    a2 = jnp.zeros((3, 4))
    b2 = jnp.zeros((3, 4))
    assert_shape(jnp.lexsort((b2, a2)), (3, 4))


def test_top_k() -> None:
    x = jnp.ones((2, 3, 6))

    values, indices = jnp.top_k(x, 4)
    assert_shape(values, (2, 3, 4))
    assert_shape(indices, (2, 3, 4))

    values_ax0, indices_ax0 = jnp.top_k(x, 1, axis=0)
    assert_shape(values_ax0, (1, 3, 6))
    assert_shape(indices_ax0, (1, 3, 6))

    values_ax1, indices_ax1 = jnp.top_k(x, 2, axis=1)
    assert_shape(values_ax1, (2, 2, 6))
    assert_shape(indices_ax1, (2, 2, 6))


def test_top_k_rejects_out_of_bounds_axis() -> None:
    x = jnp.ones((2, 3))
    v, idx = jnp.top_k(x, 1, axis=1)
    assert_shape(v, (2, 1))
    try:
        # E: Cannot evaluate type-level shape DSL call: axis out of bounds
        jnp.top_k(x, 1, axis=5)
    except ValueError:
        pass
    else:
        raise AssertionError("expected JAX to reject an out-of-bounds axis")


def test_searchsorted() -> None:
    a = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

    assert_shape(jnp.searchsorted(a, 2.5), ())
    assert_shape(jnp.searchsorted(a, jnp.ones(3)), (3,))
    assert_shape(jnp.searchsorted(a, jnp.zeros((2, 3))), (2, 3))

    # Method
    assert_shape(a.searchsorted(2.5), ())
    assert_shape(a.searchsorted(jnp.zeros((2, 3))), (2, 3))


def test_nonzero_and_flatnonzero() -> None:
    x = jnp.ones((2, 3))

    res = jnp.nonzero(x, size=4)
    assert_shape(res[0], (4,))
    assert_shape(res[1], (4,))

    # Method
    res_m = x.nonzero(size=4)
    assert_shape(res_m[0], (4,))
    assert_shape(res_m[1], (4,))

    assert_shape(jnp.flatnonzero(x, size=5), (5,))


def test_argwhere() -> None:
    x1 = jnp.ones((5,))
    x2 = jnp.ones((2, 3))
    x3 = jnp.ones((2, 3, 4))

    assert_shape(jnp.argwhere(x1, size=4), (4, 1))
    assert_shape(jnp.argwhere(x2, size=5), (5, 2))
    assert_shape(jnp.argwhere(x3, size=6), (6, 3))


def test_nan_to_num() -> None:
    x = jnp.ones((2, 3, 4))
    assert_shape(jnp.nan_to_num(x), (2, 3, 4))


def test_digitize() -> None:
    x = jnp.ones((2, 3))
    bins = jnp.array([0.0, 1.0, 2.0])
    assert_shape(jnp.digitize(x, bins), (2, 3))


def test_where() -> None:
    cond = jnp.ones((2, 3), dtype=bool)
    x = jnp.zeros((2, 3))
    y = jnp.ones((2, 3))

    assert_shape(jnp.where(cond, x, y), (2, 3))
    assert_shape(jnp.where(cond, 0.0, 1.0), (2, 3))
    assert_shape(jnp.where(cond, x, 0.0), (2, 3))
    assert_shape(jnp.where(cond, 0.0, y), (2, 3))

    # Broadcast
    cond_bc = jnp.ones((3,), dtype=bool)
    x_bc = jnp.zeros((2, 1))
    assert_shape(jnp.where(cond_bc, x_bc, 0.0), (2, 3))

    # 1-arg where (nonzero)
    idx = jnp.where(cond, size=4)
    assert_shape(idx[0], (4,))
    assert_shape(idx[1], (4,))


def test_bincount() -> None:
    x = jnp.array([1, 2, 3])
    assert_shape(jnp.bincount(x, length=10), (10,))


def test_choose() -> None:
    a = jnp.zeros(3, dtype=jnp.int32)
    choices = [jnp.zeros(3), jnp.ones(3)]
    assert_shape(jnp.choose(a, choices), (3,))

    # Method
    assert_shape(a.choose(choices), (3,))


def test_clip() -> None:
    x = jnp.ones((2, 3))

    assert_shape(jnp.clip(x, 0.0, 2.0), (2, 3))
    assert_shape(jnp.clip(x, min=0.0, max=2.0), (2, 3))
    assert_shape(jnp.clip(x, min=0.0), (2, 3))
    assert_shape(jnp.clip(x, max=2.0), (2, 3))

    # Broadcast min/max
    min_arr = jnp.zeros((3,))
    max_arr = jnp.ones((2, 1))
    assert_shape(jnp.clip(x, min_arr, max_arr), (2, 3))

    # Method
    assert_shape(x.clip(0.0, 2.0), (2, 3))
    assert_shape(x.clip(min=0.0, max=2.0), (2, 3))
    assert_shape(x.clip(min_arr, max_arr), (2, 3))


def test_fmax_and_fmin() -> None:
    a = jnp.ones((2, 3))
    b = jnp.zeros((3,))

    assert_shape(jnp.fmax(a, b), (2, 3))
    assert_shape(jnp.fmax(a, 0.0), (2, 3))
    assert_shape(jnp.fmax(0.0, a), (2, 3))

    assert_shape(jnp.fmin(a, b), (2, 3))
    assert_shape(jnp.fmin(a, 0.0), (2, 3))
    assert_shape(jnp.fmin(0.0, a), (2, 3))


def test_piecewise() -> None:
    x = jnp.ones((2, 3))
    condlist = [x < 0, x >= 0]
    funclist = [-1.0, 1.0]
    assert_shape(jnp.piecewise(x, condlist, funclist), (2, 3))


def test_select() -> None:
    condlist = [jnp.zeros((3, 4), dtype=bool), jnp.ones((3, 4), dtype=bool)]
    choicelist = [jnp.zeros((3, 4)), jnp.ones((3, 4))]
    assert_shape(jnp.select(condlist, choicelist), (3, 4))


def test_set_operations() -> None:
    a = jnp.array([1, 2, 3])
    b = jnp.array([2, 3, 4])

    assert_shape(jnp.intersect1d(a, b, size=5), (5,))
    idx = jnp.intersect1d(a, b, return_indices=True, size=5)
    assert_shape(idx[0], (5,))
    assert_shape(idx[1], (5,))
    assert_shape(idx[2], (5,))

    elem = jnp.ones((2, 3, 4))
    assert_shape(jnp.isin(elem, a), (2, 3, 4))

    assert_shape(jnp.setdiff1d(a, b, size=6), (6,))
    assert_shape(jnp.setxor1d(a, b, size=7), (7,))
    assert_shape(jnp.union1d(a, b, size=8), (8,))

    assert_shape(jnp.unique(a, size=4), (4,))
    assert_shape(jnp.unique_values(a, size=4), (4,))

    u_all = jnp.unique_all(a)
    assert hasattr(u_all, "values")
    assert hasattr(u_all, "indices")
    assert hasattr(u_all, "inverse_indices")
    assert hasattr(u_all, "counts")

    u_cnt = jnp.unique_counts(a)
    assert hasattr(u_cnt, "values")
    assert hasattr(u_cnt, "counts")

    u_inv = jnp.unique_inverse(a)
    assert hasattr(u_inv, "values")
    assert hasattr(u_inv, "inverse_indices")
