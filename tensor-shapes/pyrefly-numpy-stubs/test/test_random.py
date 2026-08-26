# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type

import numpy as np
from shape_extensions import assert_shape


def test_randn_1d_shape() -> None:
    assert_shape(np.random.randn(5).shape, (5,))


def test_randn_2d_shape() -> None:
    assert_shape(np.random.randn(5, 3).shape, (5, 3))


def test_randn_2d_singleton_dimension_shape() -> None:
    assert_shape(np.random.randn(5, 1).shape, (5, 1))


def test_randn_3d_shape() -> None:
    result = np.random.randn(5, 3, 2)
    assert_type(result, np.ndarray[[5, 3, 2], np.dtype[np.float64]])
    assert_shape(result.shape, (5, 3, 2))


def test_randn_4d_shape() -> None:
    result = np.random.randn(2, 3, 4, 5)
    assert_type(result, np.ndarray[[2, 3, 4, 5], np.dtype[np.float64]])
    assert_shape(result.shape, (2, 3, 4, 5))


def check_randn_no_args_returns_float() -> None:
    assert_type(np.random.randn(), float)
