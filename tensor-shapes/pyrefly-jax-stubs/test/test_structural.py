# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import jax.numpy as jnp
from shape_extensions import assert_shape


def test_expand_dims() -> None:
    x = jnp.ones((2, 3))

    assert_shape(jnp.expand_dims(x, 0), (1, 2, 3))
    assert_shape(jnp.expand_dims(x, 1), (2, 1, 3))
    assert_shape(jnp.expand_dims(x, 2), (2, 3, 1))
    assert_shape(jnp.expand_dims(x, -1), (2, 3, 1))
    assert_shape(jnp.expand_dims(x, -2), (2, 1, 3))
    assert_shape(jnp.expand_dims(x, -3), (1, 2, 3))

    scalar = jnp.ones(())
    assert_shape(jnp.expand_dims(scalar, 0), (1,))
    assert_shape(jnp.expand_dims(scalar, -1), (1,))


def test_squeeze() -> None:
    assert_shape(jnp.squeeze(jnp.ones((1, 2, 1, 3, 1))), (2, 3))
    assert_shape(jnp.squeeze(jnp.ones((2, 3))), (2, 3))
    assert_shape(jnp.squeeze(jnp.ones((1, 2, 1, 3)), 0), (2, 1, 3))
    assert_shape(jnp.squeeze(jnp.ones((1, 2, 1, 3)), -2), (1, 2, 3))
    assert_shape(jnp.squeeze(jnp.ones((1,))), ())

    # Array method
    assert_shape(jnp.ones((1, 2, 1)).squeeze(), (2,))
    assert_shape(jnp.ones((1, 2, 1)).squeeze(0), (2, 1))
    assert_shape(jnp.ones((1, 2, 1)).squeeze(-1), (1, 2))


def test_concatenate_and_concat() -> None:
    # 1-D
    assert_shape(jnp.concatenate([jnp.ones(2), jnp.ones(3)], axis=0), (5,))

    # 2-D along axis 0 and 1
    assert_shape(jnp.concatenate([jnp.ones((2, 3)), jnp.ones((4, 3))], axis=0), (6, 3))
    assert_shape(jnp.concatenate([jnp.ones((2, 3)), jnp.ones((2, 4))], axis=1), (2, 7))
    assert_shape(jnp.concatenate([jnp.ones((2, 3)), jnp.ones((4, 3))], axis=-2), (6, 3))

    # Multiple arrays
    assert_shape(
        jnp.concatenate([jnp.ones((2, 3)), jnp.ones((1, 3)), jnp.ones((4, 3))]),
        (7, 3),
    )

    # concat alias
    assert_shape(jnp.concat([jnp.ones((2, 3)), jnp.ones((4, 3))]), (6, 3))
    assert_shape(jnp.concat([jnp.ones((2, 3)), jnp.ones((2, 4))], axis=1), (2, 7))


def test_stack() -> None:
    # 1-D to 2-D
    assert_shape(jnp.stack([jnp.ones(3), jnp.ones(3)], axis=0), (2, 3))
    assert_shape(jnp.stack([jnp.ones(3), jnp.ones(3)], axis=1), (3, 2))
    assert_shape(jnp.stack([jnp.ones(3), jnp.ones(3)], axis=-1), (3, 2))

    # 2-D to 3-D
    assert_shape(jnp.stack([jnp.ones((2, 3)), jnp.ones((2, 3))], axis=0), (2, 2, 3))
    assert_shape(jnp.stack([jnp.ones((2, 3)), jnp.ones((2, 3))], axis=1), (2, 2, 3))
    assert_shape(jnp.stack([jnp.ones((2, 3)), jnp.ones((2, 3))], axis=-1), (2, 3, 2))

    # 3 arrays
    assert_shape(
        jnp.stack([jnp.ones((2, 3)), jnp.ones((2, 3)), jnp.ones((2, 3))]),
        (3, 2, 3),
    )


def test_vstack() -> None:
    # 1-D treated as row vectors (1, N)
    assert_shape(jnp.vstack([jnp.ones(3), jnp.ones(3)]), (2, 3))
    assert_shape(jnp.vstack([jnp.ones(3), jnp.ones(3), jnp.ones(3)]), (3, 3))

    # 2-D concatenated along axis 0
    assert_shape(jnp.vstack([jnp.ones((2, 3)), jnp.ones((4, 3))]), (6, 3))

    # 3-D concatenated along axis 0
    assert_shape(jnp.vstack([jnp.ones((2, 3, 4)), jnp.ones((5, 3, 4))]), (7, 3, 4))


def test_hstack() -> None:
    # 1-D concatenated along axis 0
    assert_shape(jnp.hstack([jnp.ones(2), jnp.ones(3)]), (5,))
    assert_shape(jnp.hstack([jnp.ones(2), jnp.ones(3), jnp.ones(4)]), (9,))

    # 2-D concatenated along axis 1
    assert_shape(jnp.hstack([jnp.ones((2, 3)), jnp.ones((2, 4))]), (2, 7))

    # 3-D concatenated along axis 1
    assert_shape(jnp.hstack([jnp.ones((2, 3, 4)), jnp.ones((2, 5, 4))]), (2, 8, 4))


def test_broadcast_to() -> None:
    assert_shape(jnp.broadcast_to(jnp.ones((2, 3)), (2, 3)), (2, 3))
    assert_shape(jnp.broadcast_to(jnp.ones((2, 3)), (4, 2, 3)), (4, 2, 3))
    assert_shape(jnp.broadcast_to(jnp.ones((1, 3)), (2, 3)), (2, 3))
    assert_shape(jnp.broadcast_to(jnp.ones(()), (2, 3)), (2, 3))
    assert_shape(jnp.broadcast_to(jnp.ones(()), 5), (5,))


def test_ravel() -> None:
    assert_shape(jnp.ravel(jnp.ones((3, 4))), (12,))
    assert_shape(jnp.ravel(jnp.ones((2, 3, 4))), (24,))
    assert_shape(jnp.ravel(jnp.ones(())), (1,))

    # Array method
    assert_shape(jnp.ones((3, 4)).ravel(), (12,))
    assert_shape(jnp.ones((2, 3, 4)).ravel(), (24,))


def test_column_stack() -> None:
    # 1-D arrays stacked as columns (N, len(tup))
    assert_shape(jnp.column_stack([jnp.ones(3), jnp.ones(3)]), (3, 2))
    assert_shape(jnp.column_stack([jnp.ones(4), jnp.ones(4), jnp.ones(4)]), (4, 3))

    # 2-D arrays stacked as-is along axis 1 (like hstack)
    assert_shape(jnp.column_stack([jnp.ones((2, 3)), jnp.ones((2, 4))]), (2, 7))


def test_dstack() -> None:
    # 1-D reshaped to (1, N, 1) and concatenated along axis 2
    assert_shape(jnp.dstack([jnp.ones(3), jnp.ones(3)]), (1, 3, 2))

    # 2-D reshaped to (M, N, 1) and concatenated along axis 2
    assert_shape(jnp.dstack([jnp.ones((2, 3)), jnp.ones((2, 3))]), (2, 3, 2))

    # 3-D concatenated along axis 2
    assert_shape(jnp.dstack([jnp.ones((2, 3, 4)), jnp.ones((2, 3, 5))]), (2, 3, 9))


def test_swapaxes() -> None:
    assert_shape(jnp.swapaxes(jnp.ones((2, 3)), 0, 1), (3, 2))
    assert_shape(jnp.swapaxes(jnp.ones((2, 3, 4)), 0, 2), (4, 3, 2))
    assert_shape(jnp.swapaxes(jnp.ones((2, 3, 4)), -1, -2), (2, 4, 3))
    assert_shape(jnp.swapaxes(jnp.ones((2, 3)), 1, 1), (2, 3))

    # Array method
    assert_shape(jnp.ones((2, 3, 4)).swapaxes(0, 1), (3, 2, 4))
    assert_shape(jnp.ones((2, 3, 4)).swapaxes(-1, 0), (4, 3, 2))


def test_moveaxis() -> None:
    x = jnp.ones((2, 3, 4, 5))
    assert_shape(jnp.moveaxis(x, 0, -1), (3, 4, 5, 2))
    assert_shape(jnp.moveaxis(x, -1, 0), (5, 2, 3, 4))
    assert_shape(jnp.moveaxis(x, 1, 2), (2, 4, 3, 5))
    assert_shape(jnp.moveaxis(x, 2, 1), (2, 4, 3, 5))


def test_rollaxis() -> None:
    x = jnp.ones((2, 3, 4, 5))
    assert_shape(jnp.rollaxis(x, 2), (4, 2, 3, 5))
    assert_shape(jnp.rollaxis(x, 1, 3), (2, 4, 3, 5))
    assert_shape(jnp.rollaxis(x, 3, 1), (2, 5, 3, 4))
    assert_shape(jnp.rollaxis(x, -1, 0), (5, 2, 3, 4))


def test_flip() -> None:
    x = jnp.ones((2, 3))
    assert_shape(jnp.flip(x), (2, 3))
    assert_shape(jnp.flip(x, 0), (2, 3))
    assert_shape(jnp.flip(x, 1), (2, 3))
    assert_shape(jnp.flip(x, -1), (2, 3))

    x3 = jnp.ones((2, 3, 4))
    assert_shape(jnp.flip(x3, (0, 2)), (2, 3, 4))
    assert_shape(jnp.flip(x, [0, 1]), (2, 3))


def test_fliplr_and_flipud() -> None:
    assert_shape(jnp.fliplr(jnp.ones((2, 3))), (2, 3))
    assert_shape(jnp.fliplr(jnp.ones((2, 3, 4))), (2, 3, 4))

    assert_shape(jnp.flipud(jnp.ones(3)), (3,))
    assert_shape(jnp.flipud(jnp.ones((2, 3))), (2, 3))


def test_roll() -> None:
    x = jnp.ones((2, 3))
    assert_shape(jnp.roll(x, 1), (2, 3))
    assert_shape(jnp.roll(x, 1, axis=0), (2, 3))
    assert_shape(jnp.roll(x, 1, axis=1), (2, 3))
    assert_shape(jnp.roll(x, (1, 2), axis=(0, 1)), (2, 3))
    assert_shape(jnp.roll(x, 1, axis=[0, 1]), (2, 3))


def test_permute_dims() -> None:
    x = jnp.ones((2, 3, 4))
    assert_shape(jnp.permute_dims(x, (1, 2, 0)), (3, 4, 2))
    assert_shape(jnp.permute_dims(x, (2, 0, 1)), (4, 2, 3))
    assert_shape(jnp.permute_dims(x, (-1, -2, -3)), (4, 3, 2))


def test_matrix_transpose() -> None:
    assert_shape(jnp.matrix_transpose(jnp.ones((2, 3))), (3, 2))
    assert_shape(jnp.matrix_transpose(jnp.ones((2, 3, 4))), (2, 4, 3))
    assert_shape(jnp.matrix_transpose(jnp.ones((2, 3, 4, 5))), (2, 3, 5, 4))

    # linalg.matrix_transpose
    assert_shape(jnp.linalg.matrix_transpose(jnp.ones((2, 3))), (3, 2))
    assert_shape(jnp.linalg.matrix_transpose(jnp.ones((2, 3, 4))), (2, 4, 3))
