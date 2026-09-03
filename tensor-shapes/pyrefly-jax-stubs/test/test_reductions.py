# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import jax
import jax.numpy as jnp
from shape_extensions import assert_shape, IntVar

N = IntVar("N")
M = IntVar("M")


# Only a tuple is a Flag domain, so any other sequence axis is gradual.
GRADUAL_SHAPE_RUNTIME_TESTS = {"test_non_tuple_sequence_axis_is_accepted"}


def reject_out_of_bounds_axis(x: jax.Array[[N, M]]) -> None:
    # E: Cannot evaluate type-level shape DSL call: axis out of bounds
    jnp.sum(x, axis=2)


def reject_duplicate_axis(x: jax.Array[[N, M]]) -> None:
    # E: Cannot evaluate type-level shape DSL call: duplicate axis
    jnp.sum(x, axis=(0, 0))


def test_reductions_accept_their_other_keywords() -> None:
    a = jnp.ones((3, 4))

    assert_shape(jnp.sum(a, axis=0, dtype=jnp.float32), (4,))
    assert_shape(jnp.sum(a, axis=0, where=None), (4,))
    assert_shape(a.sum(axis=0, dtype=jnp.float32), (4,))


def test_reduce_all_axes() -> None:
    a = jnp.ones((3, 4))

    assert_shape(jnp.sum(a), ())
    assert_shape(jnp.mean(a), ())
    assert_shape(jnp.max(a), ())
    assert_shape(jnp.min(a), ())
    assert_shape(jnp.prod(a), ())


def test_reduce_single_axis() -> None:
    a = jnp.ones((3, 4))

    assert_shape(jnp.sum(a, axis=0), (4,))
    assert_shape(jnp.sum(a, axis=1), (3,))
    assert_shape(jnp.mean(a, axis=0), (4,))
    assert_shape(jnp.max(a, axis=1), (3,))


def test_reduce_negative_axis() -> None:
    a = jnp.ones((3, 4))

    assert_shape(jnp.sum(a, axis=-1), (3,))
    assert_shape(jnp.sum(a, axis=-2), (4,))


def test_reduce_multiple_axes() -> None:
    a = jnp.ones((2, 3, 4))

    assert_shape(jnp.sum(a, axis=(0, 2)), (3,))
    assert_shape(jnp.mean(a, axis=(1, 2)), (2,))


def test_reduce_keepdims() -> None:
    a = jnp.ones((3, 4))

    assert_shape(jnp.sum(a, axis=1, keepdims=True), (3, 1))
    assert_shape(jnp.sum(a, axis=0, keepdims=True), (1, 4))
    assert_shape(jnp.mean(a, keepdims=True), (1, 1))


def test_reduce_methods() -> None:
    a = jnp.ones((3, 4))

    assert_shape(a.sum(), ())
    assert_shape(a.sum(axis=0), (4,))
    assert_shape(a.prod(), ())
    assert_shape(a.prod(axis=0), (4,))
    assert_shape(a.mean(axis=1), (3,))
    assert_shape(a.max(axis=1, keepdims=True), (3, 1))
    assert_shape(a.min(axis=0), (4,))


def test_non_tuple_sequence_axis_is_accepted() -> None:
    c = jnp.ones((2, 3, 4))

    assert jnp.sum(c, axis=[0, 2]).shape == (3,)
    assert jnp.sum(c, axis=range(2)).shape == (4,)
    assert c.mean(axis=[0, 2]).shape == (3,)
    assert c.mean(axis=range(2)).shape == (4,)


def test_reduce_rejects_out_of_bounds_axis() -> None:
    a = jnp.ones((3, 4))

    assert_shape(jnp.sum(a, axis=1), (3,))
    try:
        # E: Cannot evaluate type-level shape DSL call: axis out of bounds
        jnp.sum(a, axis=2)
    except ValueError:
        pass
    else:
        raise AssertionError("expected JAX to reject an out-of-bounds axis")


def test_boolean_reductions() -> None:
    a = jnp.ones((3, 4))

    assert_shape(jnp.all(a), ())
    assert_shape(jnp.all(a, axis=0), (4,))
    assert_shape(jnp.all(a, axis=1, keepdims=True), (3, 1))

    assert_shape(jnp.any(a), ())
    assert_shape(jnp.any(a, axis=0), (4,))
    assert_shape(jnp.any(a, axis=1, keepdims=True), (3, 1))

    assert_shape(jnp.count_nonzero(a), ())
    assert_shape(jnp.count_nonzero(a, axis=0), (4,))
    assert_shape(jnp.count_nonzero(a, axis=1, keepdims=True), (3, 1))


def test_extrema_and_stats_reductions() -> None:
    a = jnp.ones((3, 4))

    assert_shape(jnp.amax(a), ())
    assert_shape(jnp.amax(a, axis=0), (4,))
    assert_shape(jnp.amin(a), ())
    assert_shape(jnp.amin(a, axis=1), (3,))

    assert_shape(jnp.ptp(a), ())
    assert_shape(jnp.ptp(a, axis=0), (4,))
    assert_shape(jnp.ptp(a, axis=1, keepdims=True), (3, 1))

    assert_shape(jnp.std(a), ())
    assert_shape(jnp.std(a, axis=0), (4,))
    assert_shape(jnp.std(a, axis=1, keepdims=True), (3, 1))

    assert_shape(jnp.var(a), ())
    assert_shape(jnp.var(a, axis=0), (4,))
    assert_shape(jnp.var(a, axis=1, keepdims=True), (3, 1))

    assert_shape(jnp.median(a), ())
    assert_shape(jnp.median(a, axis=0), (4,))
    assert_shape(jnp.median(a, axis=1, keepdims=True), (3, 1))


def test_nan_reductions() -> None:
    a = jnp.ones((3, 4))

    assert_shape(jnp.nanmax(a), ())
    assert_shape(jnp.nanmax(a, axis=0), (4,))
    assert_shape(jnp.nanmin(a), ())
    assert_shape(jnp.nanmin(a, axis=1), (3,))

    assert_shape(jnp.nansum(a), ())
    assert_shape(jnp.nansum(a, axis=0), (4,))
    assert_shape(jnp.nanprod(a), ())
    assert_shape(jnp.nanprod(a, axis=1), (3,))

    assert_shape(jnp.nanmean(a), ())
    assert_shape(jnp.nanmean(a, axis=0), (4,))
    assert_shape(jnp.nanstd(a), ())
    assert_shape(jnp.nanstd(a, axis=1), (3,))
    assert_shape(jnp.nanvar(a), ())
    assert_shape(jnp.nanvar(a, axis=0), (4,))
    assert_shape(jnp.nanmedian(a), ())
    assert_shape(jnp.nanmedian(a, axis=1), (3,))


def test_average() -> None:
    a = jnp.ones((3, 4))

    assert_shape(jnp.average(a), ())
    assert_shape(jnp.average(a, axis=0), (4,))
    assert_shape(jnp.average(a, axis=1, keepdims=True), (3, 1))

    avg, w_sum = jnp.average(a, axis=0, returned=True)
    assert_shape(avg, (4,))
    assert_shape(w_sum, (4,))


def test_arg_reductions() -> None:
    a = jnp.ones((3, 4))

    assert_shape(jnp.argmax(a), ())
    assert_shape(jnp.argmax(a, axis=0), (4,))
    assert_shape(jnp.argmax(a, axis=1), (3,))
    assert_shape(jnp.argmax(a, axis=1, keepdims=True), (3, 1))

    assert_shape(jnp.argmin(a), ())
    assert_shape(jnp.argmin(a, axis=0), (4,))
    assert_shape(jnp.argmin(a, axis=1), (3,))

    assert_shape(jnp.nanargmax(a), ())
    assert_shape(jnp.nanargmax(a, axis=0), (4,))
    assert_shape(jnp.nanargmin(a), ())
    assert_shape(jnp.nanargmin(a, axis=1), (3,))


def test_cumulative_ops() -> None:
    a = jnp.ones((3, 4))

    assert_shape(jnp.cumsum(a, axis=0), (3, 4))
    assert_shape(jnp.cumsum(a, axis=1), (3, 4))
    assert_shape(jnp.cumprod(a, axis=0), (3, 4))
    assert_shape(jnp.cumprod(a, axis=1), (3, 4))

    assert_shape(jnp.cumulative_sum(a, axis=0), (3, 4))
    assert_shape(jnp.cumulative_prod(a, axis=1), (3, 4))

    assert_shape(jnp.nancumsum(a, axis=0), (3, 4))
    assert_shape(jnp.nancumprod(a, axis=1), (3, 4))

    assert jnp.cumsum(a).shape == (12,)
    assert jnp.cumprod(a).shape == (12,)


def test_quantile_and_percentile() -> None:
    a = jnp.ones((3, 4))

    assert_shape(jnp.quantile(a, 0.5), ())
    assert_shape(jnp.quantile(a, 0.5, axis=0), (4,))
    assert_shape(jnp.quantile(a, 0.5, axis=1, keepdims=True), (3, 1))

    assert_shape(jnp.percentile(a, 50), ())
    assert_shape(jnp.percentile(a, 50, axis=0), (4,))
    assert_shape(jnp.percentile(a, 50, axis=1, keepdims=True), (3, 1))

    assert_shape(jnp.nanquantile(a, 0.5), ())
    assert_shape(jnp.nanquantile(a, 0.5, axis=0), (4,))

    assert_shape(jnp.nanpercentile(a, 50), ())
    assert_shape(jnp.nanpercentile(a, 50, axis=0), (4,))


def test_diff_gradient_trapezoid() -> None:
    a = jnp.ones((3, 4))

    assert jnp.diff(a).shape == (3, 3)
    assert jnp.ediff1d(a).shape == (11,)
    assert [g.shape for g in jnp.gradient(a)] == [(3, 4), (3, 4)]
    assert_shape(jnp.trapezoid(a, axis=-1), (3,))
    assert_shape(jnp.trapezoid(a, axis=0), (4,))
    assert jnp.corrcoef(a).shape == (3, 3)
    assert jnp.cov(a).shape == (3, 3)


def test_additional_array_methods() -> None:
    a = jnp.ones((3, 4))

    assert_shape(a.all(), ())
    assert_shape(a.all(axis=0), (4,))
    assert_shape(a.any(), ())
    assert_shape(a.any(axis=1), (3,))
    assert_shape(a.std(), ())
    assert_shape(a.std(axis=0), (4,))
    assert_shape(a.var(), ())
    assert_shape(a.var(axis=1), (3,))
    assert_shape(a.ptp(), ())
    assert_shape(a.ptp(axis=0), (4,))
    assert_shape(a.argmax(), ())
    assert_shape(a.argmax(axis=0), (4,))
    assert_shape(a.argmin(), ())
    assert_shape(a.argmin(axis=1), (3,))
    assert_shape(a.cumsum(axis=0), (3, 4))
    assert_shape(a.cumprod(axis=1), (3, 4))
