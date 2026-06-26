# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import numpy as np
from shape_extensions import assert_shape


def test_zeros_1d_int_shape() -> None:
    assert_shape(np.zeros(5), (5,))


def test_ones_1d_int_shape() -> None:
    assert_shape(np.ones(4), (4,))


def test_full_1d_int_shape() -> None:
    assert_shape(np.full(3, 7.0), (3,))


def test_empty_1d_int_shape() -> None:
    assert_shape(np.empty(6), (6,))


def test_zeros_tuple_shape() -> None:
    assert_shape(np.zeros((3, 4)), (3, 4))


def test_ones_tuple_shape() -> None:
    assert_shape(np.ones((2, 5)), (2, 5))


def test_full_tuple_shape() -> None:
    assert_shape(np.full((3, 3), -1.0), (3, 3))


def test_empty_tuple_shape() -> None:
    assert_shape(np.empty((6,)), (6,))


def test_eye_square_shape() -> None:
    assert_shape(np.eye(4), (4, 4))


def test_identity_square_shape() -> None:
    assert_shape(np.identity(5), (5, 5))
