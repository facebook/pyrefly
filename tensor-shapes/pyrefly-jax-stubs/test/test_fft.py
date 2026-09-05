# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import jax
import jax.numpy as jnp
from shape_extensions import assert_shape, IntTuple


def generic_fft_preserves_shape[Shape: IntTuple](
    x: jax.Array[Shape],
) -> jax.Array[Shape]:
    return jnp.fft.fft(x)


def generic_fftshift_preserves_shape[Shape: IntTuple](
    x: jax.Array[Shape],
) -> jax.Array[Shape]:
    return jnp.fft.fftshift(x)


def test_1d_fft_and_ifft() -> None:
    vec8 = jnp.ones(8)
    mat48 = jnp.ones((4, 8))
    tensor248 = jnp.ones((2, 4, 8))

    assert_shape(jnp.fft.fft(vec8).shape, (8,))
    assert_shape(jnp.fft.ifft(vec8).shape, (8,))
    assert_shape(jnp.fft.fft(mat48).shape, (4, 8))
    assert_shape(jnp.fft.ifft(mat48).shape, (4, 8))
    assert_shape(jnp.fft.fft(tensor248).shape, (2, 4, 8))
    assert_shape(jnp.fft.ifft(tensor248).shape, (2, 4, 8))

    # Specifying n
    assert_shape(jnp.fft.fft(vec8, n=16).shape, (16,))
    assert_shape(jnp.fft.ifft(vec8, n=16).shape, (16,))
    assert_shape(jnp.fft.fft(mat48, n=16).shape, (4, 16))
    assert_shape(jnp.fft.fft(mat48, n=16, axis=0).shape, (16, 8))


def test_rfft_and_irfft() -> None:
    vec8 = jnp.ones(8)
    mat48 = jnp.ones((4, 8))

    # rfft output length is N // 2 + 1
    assert_shape(jnp.fft.rfft(vec8).shape, (5,))
    assert_shape(jnp.fft.rfft(mat48).shape, (4, 5))
    assert_shape(jnp.fft.rfft(mat48, axis=0).shape, (3, 8))
    assert_shape(jnp.fft.rfft(vec8, n=16).shape, (9,))

    # irfft default output length is 2 * (N - 1)
    assert_shape(jnp.fft.irfft(vec8).shape, (14,))
    assert_shape(jnp.fft.irfft(mat48).shape, (4, 14))
    assert_shape(jnp.fft.irfft(mat48, axis=0).shape, (6, 8))
    assert_shape(jnp.fft.irfft(vec8, n=16).shape, (16,))


def test_hfft_and_ihfft() -> None:
    vec8 = jnp.ones(8)
    mat48 = jnp.ones((4, 8))

    assert_shape(jnp.fft.hfft(vec8).shape, (14,))
    assert_shape(jnp.fft.hfft(mat48).shape, (4, 14))
    assert_shape(jnp.fft.hfft(vec8, n=16).shape, (16,))

    assert_shape(jnp.fft.ihfft(vec8).shape, (5,))
    assert_shape(jnp.fft.ihfft(mat48).shape, (4, 5))
    assert_shape(jnp.fft.ihfft(vec8, n=16).shape, (9,))


def test_2d_fft_operations() -> None:
    mat48 = jnp.ones((4, 8))
    tensor248 = jnp.ones((2, 4, 8))

    assert_shape(jnp.fft.fft2(mat48).shape, (4, 8))
    assert_shape(jnp.fft.ifft2(mat48).shape, (4, 8))
    assert_shape(jnp.fft.fft2(tensor248).shape, (2, 4, 8))
    assert_shape(jnp.fft.ifft2(tensor248).shape, (2, 4, 8))

    assert_shape(jnp.fft.rfft2(mat48).shape, (4, 5))
    assert_shape(jnp.fft.rfft2(tensor248).shape, (2, 4, 5))

    assert_shape(jnp.fft.irfft2(mat48).shape, (4, 14))
    assert_shape(jnp.fft.irfft2(tensor248).shape, (2, 4, 14))


def test_nd_fft_operations() -> None:
    tensor248 = jnp.ones((2, 4, 8))

    assert_shape(jnp.fft.fftn(tensor248).shape, (2, 4, 8))
    assert_shape(jnp.fft.ifftn(tensor248).shape, (2, 4, 8))
    assert_shape(jnp.fft.rfftn(tensor248).shape, (2, 4, 5))
    assert_shape(jnp.fft.irfftn(tensor248).shape, (2, 4, 14))


def test_fftfreq_and_rfftfreq() -> None:
    assert_shape(jnp.fft.fftfreq(8).shape, (8,))
    assert_shape(jnp.fft.fftfreq(10, d=0.5).shape, (10,))
    assert_shape(jnp.fft.rfftfreq(8).shape, (5,))
    assert_shape(jnp.fft.rfftfreq(9).shape, (5,))
    assert_shape(jnp.fft.rfftfreq(10).shape, (6,))


def test_fftshift_and_ifftshift() -> None:
    mat48 = jnp.ones((4, 8))
    tensor248 = jnp.ones((2, 4, 8))

    assert_shape(jnp.fft.fftshift(mat48).shape, (4, 8))
    assert_shape(jnp.fft.ifftshift(mat48).shape, (4, 8))
    assert_shape(jnp.fft.fftshift(tensor248).shape, (2, 4, 8))
    assert_shape(jnp.fft.ifftshift(tensor248).shape, (2, 4, 8))


def test_fft_rejects_out_of_bounds_axis() -> None:
    mat48 = jnp.ones((4, 8))

    assert_shape(jnp.fft.rfft(mat48, axis=1).shape, (4, 5))
    try:
        # E: Cannot evaluate type-level shape DSL call: FFT axis out of bounds
        jnp.fft.rfft(mat48, axis=3)
    except (ValueError, IndexError):
        pass
    else:
        raise AssertionError("expected JAX to reject out-of-bounds FFT axis")
