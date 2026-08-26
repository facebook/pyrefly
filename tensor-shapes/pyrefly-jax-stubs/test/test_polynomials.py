# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import jax.numpy as jnp
from shape_extensions import assert_shape


def test_interp() -> None:
    x = jnp.ones((2, 3))
    xp = jnp.array([0.0, 1.0, 2.0])
    fp = jnp.array([0.0, 1.0, 4.0])
    assert_shape(jnp.interp(x, xp, fp).shape, (2, 3))
    assert_shape(jnp.interp(jnp.ones(5), xp, fp).shape, (5,))

    # Rejection of mismatched xp and fp shapes
    try:
        # E: Argument `Array[IntTuple[4]]` is not assignable to parameter `fp` with type `Array[IntTuple[3]]`
        jnp.interp(x, jnp.ones(3), jnp.ones(4))
    except ValueError:
        pass
    else:
        raise AssertionError("expected JAX to reject mismatched xp and fp in interp")

    # Rejection of non-1D xp and fp
    try:
        # E: Argument `Array[IntTuple[2, 2]]` is not assignable to parameter `xp` with type `Array[IntTuple[@_]]`
        # E: Argument `Array[IntTuple[2, 2]]` is not assignable to parameter `fp` with type `Array[IntTuple[@_]]`
        jnp.interp(x, jnp.ones((2, 2)), jnp.ones((2, 2)))
    except ValueError:
        pass
    else:
        raise AssertionError("expected JAX to reject non-1D arrays in interp")


def test_convolve_correlate() -> None:
    a = jnp.ones(5)
    v = jnp.ones(3)
    # Default modes
    assert_shape(jnp.convolve(a, v).shape, (7,))
    assert_shape(jnp.correlate(a, v).shape, (3,))

    # mode="full"
    assert_shape(jnp.convolve(a, v, mode="full").shape, (7,))
    assert_shape(jnp.convolve(v, a, mode="full").shape, (7,))
    assert_shape(jnp.correlate(a, v, mode="full").shape, (7,))
    assert_shape(jnp.correlate(v, a, mode="full").shape, (7,))

    # mode="same"
    assert_shape(jnp.convolve(a, v, mode="same").shape, (5,))
    assert_shape(jnp.convolve(v, a, mode="same").shape, (5,))
    assert_shape(jnp.correlate(a, v, mode="same").shape, (5,))
    assert_shape(jnp.correlate(v, a, mode="same").shape, (5,))

    # mode="valid"
    assert_shape(jnp.convolve(a, v, mode="valid").shape, (3,))
    assert_shape(jnp.convolve(v, a, mode="valid").shape, (3,))
    assert_shape(jnp.correlate(a, v, mode="valid").shape, (3,))
    assert_shape(jnp.correlate(v, a, mode="valid").shape, (3,))

    # Equal lengths
    eq = jnp.ones(4)
    assert_shape(jnp.convolve(eq, eq, mode="full").shape, (7,))
    assert_shape(jnp.convolve(eq, eq, mode="same").shape, (4,))
    assert_shape(jnp.convolve(eq, eq, mode="valid").shape, (1,))
    assert_shape(jnp.correlate(eq, eq, mode="valid").shape, (1,))

    # Rejection of non-1-dimensional inputs
    try:
        # E: Cannot evaluate type-level shape DSL call: convolve and correlate only support 1-dimensional inputs
        jnp.convolve(jnp.ones((2, 2)), jnp.ones(3))
    except ValueError:
        pass
    else:
        raise AssertionError("expected JAX to reject non-1D input to convolve")

    try:
        # E: Cannot evaluate type-level shape DSL call: convolve and correlate only support 1-dimensional inputs
        jnp.correlate(jnp.ones(3), jnp.ones((2, 2)))
    except ValueError:
        pass
    else:
        raise AssertionError("expected JAX to reject non-1D input to correlate")

    # Rejection of empty inputs
    try:
        # E: Cannot evaluate type-level shape DSL call: inputs cannot be empty
        jnp.convolve(jnp.ones(0), jnp.ones(3))
    except ValueError:
        pass
    else:
        raise AssertionError("expected JAX to reject empty input to convolve")

    # Rejection of invalid mode
    try:
        # E: Cannot evaluate type-level shape DSL call: mode must be one of ['full', 'same', 'valid']
        jnp.convolve(jnp.ones(3), jnp.ones(3), mode="invalid")
    except ValueError:
        pass
    else:
        raise AssertionError("expected JAX to reject invalid mode to convolve")


def test_histograms() -> None:
    a = jnp.ones(10)
    # Default bins=10
    h_def, edges_def = jnp.histogram(a)
    assert_shape(h_def.shape, (10,))
    assert_shape(edges_def.shape, (11,))

    h, edges = jnp.histogram(a, bins=5)
    assert_shape(h.shape, (5,))
    assert_shape(edges.shape, (6,))

    # Default 2D
    h2_def, xedges_def, yedges_def = jnp.histogram2d(a, a)
    assert_shape(h2_def.shape, (10, 10))
    assert_shape(xedges_def.shape, (11,))
    assert_shape(yedges_def.shape, (11,))

    h2, xedges, yedges = jnp.histogram2d(a, a, bins=4)
    assert_shape(h2.shape, (4, 4))
    assert_shape(xedges.shape, (5,))
    assert_shape(yedges.shape, (5,))

    assert_shape(jnp.histogram_bin_edges(a).shape, (11,))
    assert_shape(jnp.histogram_bin_edges(a, bins=5).shape, (6,))

    sample = jnp.ones((10, 2))
    hdd, ddedges = jnp.histogramdd(sample, bins=(3, 4))
    assert_shape(hdd.shape, (3, 4))
    assert len(ddedges) == 2


def test_polynomials() -> None:
    p = jnp.array([1.0, -2.0, 1.0])
    x = jnp.ones((2, 3))
    assert_shape(jnp.polyval(p, x).shape, (2, 3))

    r = jnp.roots(p)
    assert_shape(r.shape, (2,))

    poly_from_roots = jnp.poly(r)
    assert_shape(poly_from_roots.shape, (3,))

    # poly from 2D square matrix
    assert_shape(jnp.poly(jnp.ones((3, 3))).shape, (4,))

    p1 = jnp.array([1.0, 2.0])
    p2 = jnp.array([3.0, 4.0, 5.0])
    assert_shape(jnp.polyadd(p1, p2).shape, (3,))
    assert_shape(jnp.polyadd(p2, p1).shape, (3,))
    assert_shape(jnp.polysub(p1, p2).shape, (3,))
    assert_shape(jnp.polymul(p1, p2).shape, (4,))

    q, rem = jnp.polydiv(p, p1)
    assert_shape(q.shape, (2,))
    assert_shape(rem.shape, (3,))

    _, rem_trimmed = jnp.polydiv(p, p1, trim_leading_zeros=True)
    assert_shape(rem_trimmed.shape, (1,))

    assert_shape(jnp.polyder(p).shape, (2,))
    assert_shape(jnp.polyder(p, m=2).shape, (1,))
    assert jnp.polyder(p, m=3).size == 0
    assert_shape(jnp.polyint(p).shape, (4,))
    assert_shape(jnp.polyint(p, m=2).shape, (5,))

    x_data = jnp.array([0.0, 1.0, 2.0, 3.0])
    y_data = jnp.array([0.0, 1.0, 4.0, 9.0])
    c = jnp.polyfit(x_data, y_data, 2)
    assert_shape(c.shape, (3,))
    assert_shape(jnp.polyfit(x_data, y_data, 3).shape, (4,))

    c_cov, cov = jnp.polyfit(x_data, y_data, 2, cov=True)
    assert_shape(c_cov.shape, (3,))
    assert_shape(cov.shape, (3, 3))

    fit_full = jnp.polyfit(x_data, y_data, 2, full=True)
    assert len(fit_full) == 5

    # Rejection of non-square 2d array in poly
    try:
        # E: Cannot evaluate type-level shape DSL call: input must be 1d or non-empty square 2d array
        jnp.poly(jnp.ones((2, 3)))
    except ValueError:
        pass
    else:
        raise AssertionError("expected JAX to reject non-square 2d array in poly")

    # Rejection of negative derivative order in polyder
    try:
        # E: Cannot evaluate type-level shape DSL call: Order of derivative must be positive
        jnp.polyder(jnp.ones(3), m=-1)
    except ValueError:
        pass
    else:
        raise AssertionError("expected JAX to reject negative order in polyder")

    # Rejection of negative integral order in polyint
    try:
        # E: Cannot evaluate type-level shape DSL call: Order of integral must be positive
        jnp.polyint(jnp.ones(3), m=-1)
    except ValueError:
        pass
    else:
        raise AssertionError("expected JAX to reject negative order in polyint")

    # Rejection of negative degree in polyfit
    try:
        # E: Cannot evaluate type-level shape DSL call: deg must be non-negative
        jnp.polyfit(x_data, y_data, -1)
    except ValueError:
        pass
    else:
        raise AssertionError("expected JAX to reject negative deg in polyfit")
