# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
from typing import assert_type, TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from shape_extensions import assert_shape, IntTuple


def check_array_and_asarray_list_literal_types() -> None:
    assert_type(jnp.array(1), jax.Array[[]])
    assert_type(jnp.array([1, 2, 3]), jax.Array[[3]])
    assert_type(jnp.asarray([[1, 2], [3, 4]]), jax.Array[[2, 2]])
    assert_type(jnp.asarray([[], []]), jax.Array[[2, 0]])
    assert_type(jnp.array([1, 2], dtype=jnp.float32), jax.Array[[2]])


def check_context_does_not_override_scalar_shape(_x: jax.Array[[2, 2]]) -> None:
    _x = jnp.array(1)  # E: is not assignable to variable `_x`
    _x = jnp.asarray(1)  # E: is not assignable to variable `_x`
    assert_type(_x, jax.Array[[2, 2]])


def test_array_and_asarray_list_literals() -> None:
    assert_shape(jnp.array([1, 2, 3]).shape, (3,))
    assert_shape(jnp.array([[1, 2], [3, 4]]).shape, (2, 2))
    assert_shape(jnp.array([[], []]).shape, (2, 0))
    assert_shape(jnp.asarray([1, 2, 3]).shape, (3,))
    assert_shape(jnp.asarray([[1, 2], [3, 4]]).shape, (2, 2))
    assert_shape(jnp.asarray([[], []]).shape, (2, 0))


def check_array_compatibility(array: jax.Array[[2, 3]], raw: list[int]) -> None:
    assert_type(jnp.array(array), jax.Array[[2, 3]])
    assert_type(jnp.array(raw), jax.Array[IntTuple])
    assert_type(jnp.array([1, 2], ndmin=2), jax.Array[IntTuple])


if TYPE_CHECKING:
    assert_type(jnp.array([[1], [2, 3]]), jax.Array[IntTuple])
    assert_type(jnp.asarray(["not", "numeric"]), jax.Array[IntTuple])


def test_zeros_ones_and_empty() -> None:
    assert_shape(jnp.zeros(()).shape, ())
    assert_shape(jnp.zeros(4).shape, (4,))
    assert_shape(jnp.zeros((3, 4)).shape, (3, 4))
    assert_shape(jnp.zeros((2, 3, 4)).shape, (2, 3, 4))
    assert_shape(jnp.ones(()).shape, ())
    assert_shape(jnp.ones(4).shape, (4,))
    assert_shape(jnp.ones((3, 4)).shape, (3, 4))
    assert_shape(jnp.ones((2, 3, 4)).shape, (2, 3, 4))
    assert_shape(jnp.empty(()).shape, ())
    assert_shape(jnp.empty(4).shape, (4,))
    assert_shape(jnp.empty((3, 4)).shape, (3, 4))
    assert_shape(jnp.empty((2, 3, 4)).shape, (2, 3, 4))


def test_like_constructors() -> None:
    x23 = jnp.ones((2, 3))
    x234 = jnp.ones((2, 3, 4))

    assert_shape(jnp.empty_like(x23).shape, (2, 3))
    assert_shape(jnp.empty_like(x234).shape, (2, 3, 4))
    assert_shape(jnp.empty_like(x23, shape=()).shape, ())
    assert_shape(jnp.empty_like(x23, shape=4).shape, (4,))
    assert_shape(jnp.empty_like(x23, shape=(4, 5)).shape, (4, 5))
    assert_shape(jnp.empty_like(x23, shape=(2, 3, 4, 5)).shape, (2, 3, 4, 5))

    assert_shape(jnp.zeros_like(x23).shape, (2, 3))
    assert_shape(jnp.zeros_like(x234).shape, (2, 3, 4))
    assert_shape(jnp.zeros_like(x23, shape=()).shape, ())
    assert_shape(jnp.zeros_like(x23, shape=4).shape, (4,))
    assert_shape(jnp.zeros_like(x23, shape=(4, 5)).shape, (4, 5))
    assert_shape(jnp.zeros_like(x23, shape=(2, 3, 4, 5)).shape, (2, 3, 4, 5))

    assert_shape(jnp.ones_like(x23).shape, (2, 3))
    assert_shape(jnp.ones_like(x234).shape, (2, 3, 4))
    assert_shape(jnp.ones_like(x23, shape=()).shape, ())
    assert_shape(jnp.ones_like(x23, shape=4).shape, (4,))
    assert_shape(jnp.ones_like(x23, shape=(4, 5)).shape, (4, 5))
    assert_shape(jnp.ones_like(x23, shape=(2, 3, 4, 5)).shape, (2, 3, 4, 5))

    assert_shape(jnp.full_like(x23, 7.0).shape, (2, 3))
    assert_shape(jnp.full_like(x234, 7.0).shape, (2, 3, 4))
    assert_shape(jnp.full_like(x23, 7.0, shape=()).shape, ())
    assert_shape(jnp.full_like(x23, 7.0, shape=4).shape, (4,))
    assert_shape(jnp.full_like(x23, 7.0, shape=(4, 5)).shape, (4, 5))
    assert_shape(jnp.full_like(x23, 7.0, shape=(2, 3, 4, 5)).shape, (2, 3, 4, 5))


def test_non_tuple_shapes_are_gradual() -> None:
    # A tuple is exact at any rank. Other sequences remain gradual because only
    # a tuple is a `Flag` domain, and `range(n)` for a computed `n` has no
    # statically knowable content.
    assert_shape(jnp.zeros((2, 3, 4, 5)).shape, (2, 3, 4, 5))
    assert_shape(jnp.zeros([2, 3]).shape, IntTuple, runtime=(2, 3))
    assert_shape(jnp.ones([2, 3]).shape, IntTuple, runtime=(2, 3))
    assert_shape(jnp.empty([2, 3]).shape, IntTuple, runtime=(2, 3))
    assert_shape(jnp.full([2, 3], 1.0).shape, IntTuple, runtime=(2, 3))


def test_full() -> None:
    assert_shape(jnp.full((), 2.0).shape, ())
    assert_shape(jnp.full(4, 2.0).shape, (4,))
    assert_shape(jnp.full((3, 4), 2.0).shape, (3, 4))
    assert_shape(jnp.full((2, 3, 4), 2.0).shape, (2, 3, 4))


def test_arange_and_eye() -> None:
    assert_shape(jnp.arange(5).shape, (5,))
    # JAX names the sole argument `start`, so the keyword form is valid.
    assert_shape(jnp.arange(start=5).shape, (5,))
    assert_shape(jnp.eye(3).shape, (3, 3))
    assert_shape(jnp.eye(2, 3).shape, (2, 3))
    assert_shape(jnp.identity(4).shape, (4, 4))


def test_linspace_logspace_geomspace() -> None:
    assert_shape(jnp.linspace(0.0, 1.0).shape, (50,))
    assert_shape(jnp.linspace(0.0, 1.0, 10).shape, (10,))
    assert_shape(jnp.logspace(0.0, 2.0, 20).shape, (20,))
    assert_shape(jnp.geomspace(1.0, 100.0, 15).shape, (15,))

    start = jnp.zeros((2, 1))
    stop = jnp.ones((1, 3))
    assert_shape(jnp.linspace(start, stop, 10).shape, (10, 2, 3))
    assert_shape(jnp.linspace(start, stop, 10, axis=1).shape, (2, 10, 3))
    assert_shape(jnp.linspace(start, stop, 10, axis=-1).shape, (2, 3, 10))
    samples, step = jnp.linspace(start, stop, 10, retstep=True)
    assert_shape(samples.shape, (10, 2, 3))
    assert_shape(step.shape, (2, 3))
    samples_pos, step_pos = jnp.linspace(start, stop, 10, True, True)
    assert_shape(samples_pos.shape, (10, 2, 3))
    assert_shape(step_pos.shape, (2, 3))
    n: int = 10
    assert_shape(
        jnp.linspace(start, stop, n, axis=-1).shape, (2, 3, int), runtime=(2, 3, 10)
    )
    assert_shape(jnp.logspace(start, stop, 20, axis=-1).shape, (2, 3, 20))
    assert_shape(
        jnp.geomspace(jnp.ones((2, 1)), jnp.full((1, 3), 10.0), 15, axis=-1).shape,
        (2, 3, 15),
    )


def test_diag_and_triangular() -> None:
    v4 = jnp.ones(4)
    m34 = jnp.ones((3, 4))

    # diag
    assert_shape(jnp.diag(v4).shape, (4, 4))
    assert_shape(jnp.diag(m34).shape, (3,))
    assert_shape(jnp.diagflat(v4).shape, (4, 4))

    # tri, tril, triu
    assert_shape(jnp.tri(4).shape, (4, 4))
    assert_shape(jnp.tri(3, 5).shape, (3, 5))
    assert_shape(jnp.tril(m34).shape, (3, 4))
    assert_shape(jnp.triu(m34).shape, (3, 4))


def test_vander_indices_meshgrid() -> None:
    v4 = jnp.ones(4)
    assert_shape(jnp.vander(v4).shape, (4, 4))
    assert_shape(jnp.vander(v4, 6).shape, (4, 6))

    # indices
    assert_shape(jnp.indices((3, 5)).shape, (2, 3, 5))
    assert_shape(jnp.indices((2, 3, 4)).shape, (3, 2, 3, 4))

    # meshgrid
    x = jnp.ones(3)
    y = jnp.ones(5)
    gx, gy = jnp.meshgrid(x, y)
    assert_shape(gx.shape, (5, 3))
    assert_shape(gy.shape, (5, 3))

    gx_ij, gy_ij = jnp.meshgrid(x, y, indexing="ij")
    assert_shape(gx_ij.shape, (3, 5))
    assert_shape(gy_ij.shape, (3, 5))


def test_fromfunction() -> None:
    assert_shape(jnp.fromfunction(lambda i: i, (4,)).shape, (4,))
    assert_shape(jnp.fromfunction(lambda i, j: i + j, (2, 3)).shape, (2, 3))


def test_window_functions() -> None:
    assert_shape(jnp.bartlett(10).shape, (10,))
    assert_shape(jnp.blackman(12).shape, (12,))
    assert_shape(jnp.hamming(14).shape, (14,))
    assert_shape(jnp.hanning(16).shape, (16,))
    assert_shape(jnp.kaiser(18, 5.0).shape, (18,))


def test_multi_argument_arange_lengths() -> None:
    assert_shape(jnp.arange(2, 7).shape, (5,))
    assert_shape(jnp.arange(0, 10, 2).shape, (5,))
    assert_shape(jnp.arange(0, 10, 3).shape, (4,))
    assert_shape(jnp.arange(10, 0, -2).shape, (5,))
    assert_shape(jnp.arange(10, 0, -3).shape, (4,))
    assert_shape(jnp.arange(7, 2).shape, (0,))
    assert_shape(jnp.arange(2, 7, -1).shape, (0,))

    # Floating-point lengths remain gradual because their rounding behavior is
    # not modeled by the integer shape DSL.
    assert_shape(jnp.arange(5.0).shape, (int,), runtime=(5,))
    assert_shape(jnp.arange(0.0, 1.0, 0.2).shape, (int,), runtime=(5,))


def test_single_argument_arange_clamps_a_negative_dimension() -> None:
    assert_shape(jnp.arange(-3).shape, (0,))


def test_dtype_argument_preserves_shape() -> None:
    assert_shape(jnp.zeros(4, jnp.int32).shape, (4,))
    assert_shape(jnp.ones((3, 4), jnp.float32).shape, (3, 4))


def test_array_and_asarray() -> None:
    # Python scalars
    assert_shape(jnp.array(5).shape, ())
    assert_shape(jnp.asarray(5).shape, ())
    assert_shape(jnp.array(2.5).shape, ())
    assert_shape(jnp.asarray(2.5).shape, ())
    assert_shape(jnp.array(True).shape, ())
    assert_shape(jnp.asarray(True).shape, ())
    assert_shape(jnp.array(1 + 2j).shape, ())
    assert_shape(jnp.asarray(1 + 2j).shape, ())
    assert_shape(jnp.array(5, dtype=jnp.float32).shape, ())
    assert_shape(jnp.asarray(5, dtype=jnp.float32).shape, ())

    # JAX Array inputs
    x0 = jnp.ones(())
    x1 = jnp.ones(4)
    x2 = jnp.ones((2, 3))
    x3 = jnp.ones((2, 3, 4))
    assert_shape(jnp.array(x0).shape, ())
    assert_shape(jnp.asarray(x0).shape, ())
    assert_shape(jnp.array(x1).shape, (4,))
    assert_shape(jnp.asarray(x1).shape, (4,))
    assert_shape(jnp.array(x2).shape, (2, 3))
    assert_shape(jnp.asarray(x2).shape, (2, 3))
    assert_shape(jnp.array(x3).shape, (2, 3, 4))
    assert_shape(jnp.asarray(x3).shape, (2, 3, 4))

    # NumPy array inputs
    assert_shape(jnp.array(np.ones((2, 3))).shape, (2, 3))
    assert_shape(jnp.asarray(np.ones((2, 3))).shape, (2, 3))

    # Generic inputs
    assert jnp.array([1, 2, 3]).shape == (3,)
    assert jnp.asarray([1, 2, 3]).shape == (3,)
    assert jnp.array([[1, 2], [3, 4]]).shape == (2, 2)
    assert jnp.asarray([[1, 2], [3, 4]]).shape == (2, 2)


def test_astype() -> None:
    x = jnp.ones((2, 3))
    assert_shape(jnp.astype(x, jnp.int32).shape, (2, 3))
    assert_shape(jnp.astype(x, None).shape, (2, 3))
    assert_shape(jnp.astype(5, jnp.float32).shape, ())
    assert_shape(jnp.astype(2.5, jnp.int32).shape, ())
    assert_shape(jnp.astype(np.ones((2, 3)), jnp.int32).shape, (2, 3))


def test_shape_ndim_size() -> None:
    x = jnp.ones((2, 3, 4))
    assert_shape(jnp.shape(x), (2, 3, 4))
    assert jnp.shape(x) == (2, 3, 4)
    assert jnp.ndim(x) == 3
    assert jnp.size(x) == 24
    assert jnp.size(x, axis=0) == 2
    assert jnp.size(x, axis=1) == 3
    assert jnp.size(x, axis=2) == 4

    # Scalars
    assert jnp.shape(5) == ()
    assert jnp.ndim(5) == 0
    assert jnp.size(5) == 1

    # Generic inputs
    assert jnp.shape([[1, 2], [3, 4]]) == (2, 2)
    assert jnp.ndim([[1, 2], [3, 4]]) == 2
    assert jnp.size([[1, 2], [3, 4]]) == 4


def test_constants() -> None:
    assert jnp.pi > 3.14
    assert jnp.e > 2.71
    assert jnp.euler_gamma > 0.57
    assert jnp.inf > 1e10
    assert math.isnan(jnp.nan)
    assert jnp.newaxis is None

    x = jnp.array([jnp.pi, jnp.e])
    assert_shape(x.shape, (2,))
    assert_shape(x[:, jnp.newaxis].shape, (2, 1))


def test_dtypes_and_type_inspection() -> None:
    # Scalar classes and hierarchy
    assert issubclass(jnp.floating, jnp.inexact)
    assert issubclass(jnp.integer, jnp.number)
    assert issubclass(jnp.number, jnp.generic)

    # Dtype objects and info
    fi = jnp.finfo(jnp.float32)
    assert fi.bits == 32
    assert fi.max > 1e30

    ii = jnp.iinfo(jnp.int32)
    assert ii.bits == 32
    assert ii.max == 2147483647

    # Casting & inspection
    assert jnp.can_cast(jnp.int32, jnp.int64)
    assert not jnp.can_cast(jnp.int64, jnp.int32)
    assert jnp.isdtype(jnp.float32, "real floating")
    assert jnp.isdtype(jnp.int32, "integral")
    assert jnp.issubdtype(jnp.int32, jnp.integer)
    assert jnp.promote_types(jnp.int32, jnp.float32) == jnp.float32
    assert jnp.result_type(jnp.int32, jnp.float32) == jnp.float32

    # Various dtype constants exist and can be used with array constructors
    assert_shape(jnp.zeros(2, dtype=jnp.bfloat16).shape, (2,))
    assert_shape(jnp.zeros(2, dtype=jnp.float8_e4m3fn).shape, (2,))
    assert_shape(jnp.zeros(2, dtype=jnp.int8).shape, (2,))
    assert_shape(jnp.zeros(2, dtype=jnp.uint32).shape, (2,))
    assert_shape(jnp.zeros(2, dtype=jnp.complex64).shape, (2,))


def test_device_and_out_sharding() -> None:
    dev = getattr(jax, "devices")()[0]
    assert_shape(jnp.zeros((2, 3), device=dev).shape, (2, 3))
    assert_shape(jnp.zeros((2, 3), device=None).shape, (2, 3))
    assert_shape(jnp.ones((2, 3), device=dev).shape, (2, 3))
    assert_shape(jnp.array([1, 2, 3], device=dev, out_sharding=None).shape, (3,))
    assert_shape(jnp.empty((4,), out_sharding=None).shape, (4,))
