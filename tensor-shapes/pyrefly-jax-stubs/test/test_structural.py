# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import jax.numpy as jnp
from shape_extensions import assert_shape


def test_expand_dims() -> None:
    x = jnp.ones((2, 3))

    assert_shape(jnp.expand_dims(x, 0).shape, (1, 2, 3))
    assert_shape(jnp.expand_dims(x, 1).shape, (2, 1, 3))
    assert_shape(jnp.expand_dims(x, 2).shape, (2, 3, 1))
    assert_shape(jnp.expand_dims(x, -1).shape, (2, 3, 1))
    assert_shape(jnp.expand_dims(x, -2).shape, (2, 1, 3))
    assert_shape(jnp.expand_dims(x, -3).shape, (1, 2, 3))

    scalar = jnp.ones(())
    assert_shape(jnp.expand_dims(scalar, 0).shape, (1,))
    assert_shape(jnp.expand_dims(scalar, -1).shape, (1,))


def test_squeeze() -> None:
    assert_shape(jnp.squeeze(jnp.ones((1, 2, 1, 3, 1))).shape, (2, 3))
    assert_shape(jnp.squeeze(jnp.ones((2, 3))).shape, (2, 3))
    assert_shape(jnp.squeeze(jnp.ones((1, 2, 1, 3)), 0).shape, (2, 1, 3))
    assert_shape(jnp.squeeze(jnp.ones((1, 2, 1, 3)), -2).shape, (1, 2, 3))
    assert_shape(jnp.squeeze(jnp.ones((1,))).shape, ())

    # Array method
    assert_shape(jnp.ones((1, 2, 1)).squeeze().shape, (2,))
    assert_shape(jnp.ones((1, 2, 1)).squeeze(0).shape, (2, 1))
    assert_shape(jnp.ones((1, 2, 1)).squeeze(-1).shape, (1, 2))


def test_concatenate_and_concat() -> None:
    # 1-D
    assert_shape(jnp.concatenate([jnp.ones(2), jnp.ones(3)], axis=0).shape, (5,))

    # 2-D along axis 0 and 1
    assert_shape(
        jnp.concatenate([jnp.ones((2, 3)), jnp.ones((4, 3))], axis=0).shape, (6, 3)
    )
    assert_shape(
        jnp.concatenate([jnp.ones((2, 3)), jnp.ones((2, 4))], axis=1).shape, (2, 7)
    )
    assert_shape(
        jnp.concatenate([jnp.ones((2, 3)), jnp.ones((4, 3))], axis=-2).shape, (6, 3)
    )

    # Multiple arrays
    assert_shape(
        jnp.concatenate([jnp.ones((2, 3)), jnp.ones((1, 3)), jnp.ones((4, 3))]).shape,
        (7, 3),
    )

    # concat alias
    assert_shape(jnp.concat([jnp.ones((2, 3)), jnp.ones((4, 3))]).shape, (6, 3))
    assert_shape(jnp.concat([jnp.ones((2, 3)), jnp.ones((2, 4))], axis=1).shape, (2, 7))


def test_stack() -> None:
    # 1-D to 2-D
    assert_shape(jnp.stack([jnp.ones(3), jnp.ones(3)], axis=0).shape, (2, 3))
    assert_shape(jnp.stack([jnp.ones(3), jnp.ones(3)], axis=1).shape, (3, 2))
    assert_shape(jnp.stack([jnp.ones(3), jnp.ones(3)], axis=-1).shape, (3, 2))

    # 2-D to 3-D
    assert_shape(
        jnp.stack([jnp.ones((2, 3)), jnp.ones((2, 3))], axis=0).shape, (2, 2, 3)
    )
    assert_shape(
        jnp.stack([jnp.ones((2, 3)), jnp.ones((2, 3))], axis=1).shape, (2, 2, 3)
    )
    assert_shape(
        jnp.stack([jnp.ones((2, 3)), jnp.ones((2, 3))], axis=-1).shape, (2, 3, 2)
    )

    # 3 arrays
    assert_shape(
        jnp.stack([jnp.ones((2, 3)), jnp.ones((2, 3)), jnp.ones((2, 3))]).shape,
        (3, 2, 3),
    )


def test_vstack() -> None:
    # 1-D treated as row vectors (1, N)
    assert_shape(jnp.vstack([jnp.ones(3), jnp.ones(3)]).shape, (2, 3))
    assert_shape(jnp.vstack([jnp.ones(3), jnp.ones(3), jnp.ones(3)]).shape, (3, 3))

    # 2-D concatenated along axis 0
    assert_shape(jnp.vstack([jnp.ones((2, 3)), jnp.ones((4, 3))]).shape, (6, 3))

    # 3-D concatenated along axis 0
    assert_shape(
        jnp.vstack([jnp.ones((2, 3, 4)), jnp.ones((5, 3, 4))]).shape, (7, 3, 4)
    )


def test_hstack() -> None:
    # 1-D concatenated along axis 0
    assert_shape(jnp.hstack([jnp.ones(2), jnp.ones(3)]).shape, (5,))
    assert_shape(jnp.hstack([jnp.ones(2), jnp.ones(3), jnp.ones(4)]).shape, (9,))

    # 2-D concatenated along axis 1
    assert_shape(jnp.hstack([jnp.ones((2, 3)), jnp.ones((2, 4))]).shape, (2, 7))

    # 3-D concatenated along axis 1
    assert_shape(
        jnp.hstack([jnp.ones((2, 3, 4)), jnp.ones((2, 5, 4))]).shape, (2, 8, 4)
    )


def test_broadcast_to() -> None:
    assert_shape(jnp.broadcast_to(jnp.ones((2, 3)), (2, 3)).shape, (2, 3))
    assert_shape(jnp.broadcast_to(jnp.ones((2, 3)), (4, 2, 3)).shape, (4, 2, 3))
    assert_shape(jnp.broadcast_to(jnp.ones((1, 3)), (2, 3)).shape, (2, 3))
    assert_shape(jnp.broadcast_to(jnp.ones(()), (2, 3)).shape, (2, 3))
    assert_shape(jnp.broadcast_to(jnp.ones(()), 5).shape, (5,))


def test_ravel() -> None:
    assert_shape(jnp.ravel(jnp.ones((3, 4))).shape, (12,))
    assert_shape(jnp.ravel(jnp.ones((2, 3, 4))).shape, (24,))
    assert_shape(jnp.ravel(jnp.ones(())).shape, (1,))

    # Array method
    assert_shape(jnp.ones((3, 4)).ravel().shape, (12,))
    assert_shape(jnp.ones((2, 3, 4)).ravel().shape, (24,))


def test_column_stack() -> None:
    # 1-D arrays stacked as columns (N, len(tup))
    assert_shape(jnp.column_stack([jnp.ones(3), jnp.ones(3)]).shape, (3, 2))
    assert_shape(
        jnp.column_stack([jnp.ones(4), jnp.ones(4), jnp.ones(4)]).shape, (4, 3)
    )

    # 2-D arrays stacked as-is along axis 1 (like hstack)
    assert_shape(jnp.column_stack([jnp.ones((2, 3)), jnp.ones((2, 4))]).shape, (2, 7))


def test_dstack() -> None:
    # 1-D reshaped to (1, N, 1) and concatenated along axis 2
    assert_shape(jnp.dstack([jnp.ones(3), jnp.ones(3)]).shape, (1, 3, 2))

    # 2-D reshaped to (M, N, 1) and concatenated along axis 2
    assert_shape(jnp.dstack([jnp.ones((2, 3)), jnp.ones((2, 3))]).shape, (2, 3, 2))

    # 3-D concatenated along axis 2
    assert_shape(
        jnp.dstack([jnp.ones((2, 3, 4)), jnp.ones((2, 3, 5))]).shape, (2, 3, 9)
    )


def test_swapaxes() -> None:
    assert_shape(jnp.swapaxes(jnp.ones((2, 3)), 0, 1).shape, (3, 2))
    assert_shape(jnp.swapaxes(jnp.ones((2, 3, 4)), 0, 2).shape, (4, 3, 2))
    assert_shape(jnp.swapaxes(jnp.ones((2, 3, 4)), -1, -2).shape, (2, 4, 3))
    assert_shape(jnp.swapaxes(jnp.ones((2, 3)), 1, 1).shape, (2, 3))

    # Array method
    assert_shape(jnp.ones((2, 3, 4)).swapaxes(0, 1).shape, (3, 2, 4))
    assert_shape(jnp.ones((2, 3, 4)).swapaxes(-1, 0).shape, (4, 3, 2))


def test_moveaxis() -> None:
    x = jnp.ones((2, 3, 4, 5))
    assert_shape(jnp.moveaxis(x, 0, -1).shape, (3, 4, 5, 2))
    assert_shape(jnp.moveaxis(x, -1, 0).shape, (5, 2, 3, 4))
    assert_shape(jnp.moveaxis(x, 1, 2).shape, (2, 4, 3, 5))
    assert_shape(jnp.moveaxis(x, 2, 1).shape, (2, 4, 3, 5))


def test_rollaxis() -> None:
    x = jnp.ones((2, 3, 4, 5))
    assert_shape(jnp.rollaxis(x, 2).shape, (4, 2, 3, 5))
    assert_shape(jnp.rollaxis(x, 1, 3).shape, (2, 4, 3, 5))
    assert_shape(jnp.rollaxis(x, 3, 1).shape, (2, 5, 3, 4))
    assert_shape(jnp.rollaxis(x, -1, 0).shape, (5, 2, 3, 4))


def test_flip() -> None:
    x = jnp.ones((2, 3))
    assert_shape(jnp.flip(x).shape, (2, 3))
    assert_shape(jnp.flip(x, 0).shape, (2, 3))
    assert_shape(jnp.flip(x, 1).shape, (2, 3))
    assert_shape(jnp.flip(x, -1).shape, (2, 3))

    x3 = jnp.ones((2, 3, 4))
    assert_shape(jnp.flip(x3, (0, 2)).shape, (2, 3, 4))
    assert_shape(jnp.flip(x, [0, 1]).shape, (2, 3))


def test_fliplr_and_flipud() -> None:
    assert_shape(jnp.fliplr(jnp.ones((2, 3))).shape, (2, 3))
    assert_shape(jnp.fliplr(jnp.ones((2, 3, 4))).shape, (2, 3, 4))

    assert_shape(jnp.flipud(jnp.ones(3)).shape, (3,))
    assert_shape(jnp.flipud(jnp.ones((2, 3))).shape, (2, 3))


def test_roll() -> None:
    x = jnp.ones((2, 3))
    assert_shape(jnp.roll(x, 1).shape, (2, 3))
    assert_shape(jnp.roll(x, 1, axis=0).shape, (2, 3))
    assert_shape(jnp.roll(x, 1, axis=1).shape, (2, 3))
    assert_shape(jnp.roll(x, (1, 2), axis=(0, 1)).shape, (2, 3))
    assert_shape(jnp.roll(x, 1, axis=[0, 1]).shape, (2, 3))


def test_permute_dims() -> None:
    x = jnp.ones((2, 3, 4))
    assert_shape(jnp.permute_dims(x, (1, 2, 0)).shape, (3, 4, 2))
    assert_shape(jnp.permute_dims(x, (2, 0, 1)).shape, (4, 2, 3))
    assert_shape(jnp.permute_dims(x, (-1, -2, -3)).shape, (4, 3, 2))


def test_matrix_transpose() -> None:
    assert_shape(jnp.matrix_transpose(jnp.ones((2, 3))).shape, (3, 2))
    assert_shape(jnp.matrix_transpose(jnp.ones((2, 3, 4))).shape, (2, 4, 3))
    assert_shape(jnp.matrix_transpose(jnp.ones((2, 3, 4, 5))).shape, (2, 3, 5, 4))

    # linalg.matrix_transpose
    assert_shape(jnp.linalg.matrix_transpose(jnp.ones((2, 3))).shape, (3, 2))
    assert_shape(jnp.linalg.matrix_transpose(jnp.ones((2, 3, 4))).shape, (2, 4, 3))


def test_block() -> None:
    assert_shape(jnp.block([[jnp.ones((2, 2)), jnp.zeros((2, 2))]]).shape, (2, 4))


def test_splitting() -> None:
    x = jnp.ones((2, 4))

    # split
    res_split = jnp.split(x, 2, axis=0)
    assert len(res_split) == 2
    assert_shape(res_split[0].shape, (1, 4))
    assert_shape(res_split[1].shape, (1, 4))

    # array_split
    res_arr = jnp.array_split(x, 2, axis=1)
    assert len(res_arr) == 2
    assert_shape(res_arr[0].shape, (2, 2))
    assert_shape(res_arr[1].shape, (2, 2))

    # hsplit
    res_h = jnp.hsplit(x, 2)
    assert len(res_h) == 2
    assert_shape(res_h[0].shape, (2, 2))
    assert_shape(res_h[1].shape, (2, 2))

    # vsplit
    res_v = jnp.vsplit(x, 2)
    assert len(res_v) == 2
    assert_shape(res_v[0].shape, (1, 4))
    assert_shape(res_v[1].shape, (1, 4))

    # dsplit
    x3 = jnp.ones((2, 2, 4))
    res_d = jnp.dsplit(x3, 2)
    assert len(res_d) == 2
    assert_shape(res_d[0].shape, (2, 2, 2))
    assert_shape(res_d[1].shape, (2, 2, 2))

    # unstack
    res_unstack = jnp.unstack(x, axis=0)
    assert len(res_unstack) == 2
    assert_shape(res_unstack[0].shape, (4,))
    assert_shape(res_unstack[1].shape, (4,))

    # Rejection of scalar or 0D array
    try:
        # E: Argument `Literal[1]` is not assignable to parameter `x`
        jnp.unstack(1)
    except ValueError:
        pass
    try:
        # E: Argument `Array[IntTuple[()]]` is not assignable to parameter `x`
        jnp.unstack(jnp.ones(()))
    except ValueError:
        pass

    try:
        # E: Argument `Literal[1]` is not assignable to parameter `ary`
        jnp.split(1, 2)
    except IndexError:
        pass
    try:
        # E: Argument `Array[IntTuple[()]]` is not assignable to parameter `ary`
        jnp.split(jnp.ones(()), 2)
    except IndexError:
        pass

    try:
        # E: Argument `Literal[1]` is not assignable to parameter `ary`
        jnp.hsplit(1, 2)
    except IndexError:
        pass
    try:
        # E: Argument `Array[IntTuple[()]]` is not assignable to parameter `ary`
        jnp.hsplit(jnp.ones(()), 2)
    except IndexError:
        pass

    try:
        # E: Argument `Literal[1]` is not assignable to parameter `ary`
        jnp.vsplit(1, 2)
    except IndexError:
        pass
    try:
        # E: Argument `Array[IntTuple[()]]` is not assignable to parameter `ary`
        jnp.vsplit(jnp.ones(()), 2)
    except IndexError:
        pass

    try:
        # E: Argument `Literal[1]` is not assignable to parameter `ary`
        jnp.dsplit(1, 2)
    except IndexError:
        pass
    try:
        # E: Argument `Array[IntTuple[()]]` is not assignable to parameter `ary`
        jnp.dsplit(jnp.ones(()), 2)
    except IndexError:
        pass


def test_pad() -> None:
    assert_shape(jnp.pad(jnp.ones((2, 3)), 1).shape, (4, 5))
    assert_shape(jnp.pad(jnp.ones((2, 3)), ((1, 2), (3, 4))).shape, (5, 10))


def test_repeat() -> None:
    x = jnp.ones((2, 3))
    assert_shape(jnp.repeat(x, 2).shape, (12,))
    assert_shape(jnp.repeat(x, 2, axis=0).shape, (4, 3))
    assert_shape(jnp.repeat(x, 2, axis=1).shape, (2, 6))

    # Method
    assert_shape(x.repeat(2).shape, (12,))
    assert_shape(x.repeat(2, axis=0).shape, (4, 3))
    assert_shape(x.repeat(2, axis=1).shape, (2, 6))


def test_resize() -> None:
    x = jnp.ones((2, 3))
    assert_shape(jnp.resize(x, (4, 4)).shape, (4, 4))
    assert_shape(jnp.resize(x, 7).shape, (7,))


def test_tile() -> None:
    x = jnp.ones((2, 3))
    assert_shape(jnp.tile(x, 2).shape, (2, 6))
    assert_shape(jnp.tile(x, (2, 2)).shape, (4, 6))
    assert_shape(jnp.tile(1.0, 3).shape, (3,))


def test_rot90() -> None:
    x = jnp.ones((2, 3))
    # Default k=1 (odd: transposed last two dimensions)
    assert_shape(jnp.rot90(x).shape, (3, 2))
    # Even k: same shape
    assert_shape(jnp.rot90(x, 0).shape, (2, 3))
    assert_shape(jnp.rot90(x, 2).shape, (2, 3))
    assert_shape(jnp.rot90(x, 4).shape, (2, 3))
    assert_shape(jnp.rot90(x, -2).shape, (2, 3))
    # Odd k: transposed
    assert_shape(jnp.rot90(x, 1).shape, (3, 2))
    assert_shape(jnp.rot90(x, 3).shape, (3, 2))
    assert_shape(jnp.rot90(x, -1).shape, (3, 2))

    # Higher rank array with axes=(-2, -1)
    x3 = jnp.ones((2, 3, 4))
    assert_shape(jnp.rot90(x3, 1, axes=(-2, -1)).shape, (2, 4, 3))
    assert_shape(jnp.rot90(x3, 2, axes=(-2, -1)).shape, (2, 3, 4))

    # Rejection of 1-D array
    try:
        # E: Cannot evaluate type-level shape DSL call: rot90 requires array of at least 2 dimensions
        jnp.rot90(jnp.ones((3,)))
    except ValueError:
        pass
    else:
        raise AssertionError("expected JAX to reject rot90 with < 2 dimensions")


def test_broadcast_arrays_and_shapes() -> None:
    a1, a2 = jnp.broadcast_arrays(jnp.ones((2, 1)), jnp.ones((1, 3)))
    assert_shape(a1.shape, (2, 3))
    assert_shape(a2.shape, (2, 3))

    b1, b2 = jnp.broadcast_arrays(jnp.ones((2, 3)), 1.0)
    assert_shape(b1.shape, (2, 3))
    assert_shape(b2.shape, (2, 3))

    (c1,) = jnp.broadcast_arrays(jnp.ones((2, 3)))
    assert_shape(c1.shape, (2, 3))

    assert jnp.broadcast_shapes((2, 1), (1, 3)) == (2, 3)
