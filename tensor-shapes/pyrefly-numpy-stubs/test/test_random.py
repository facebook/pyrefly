# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import numpy as np
from shape_extensions import assert_shape


def test_randn_1d_shape() -> None:
    assert_shape(np.random.randn(5), (5,))


def test_randn_2d_shape() -> None:
    assert_shape(np.random.randn(5, 3), (5, 3))


def test_randn_2d_singleton_dimension_shape() -> None:
    assert_shape(np.random.randn(5, 1), (5, 1))
