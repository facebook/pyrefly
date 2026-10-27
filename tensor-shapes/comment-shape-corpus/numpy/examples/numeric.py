# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type

import numpy as np
from numeric import outer, roll, tensordot
from shape_extensions import assert_shape

left = np.ones((3,), dtype=np.float64)
right = np.ones((4,), dtype=np.float64)
assert_type(outer(left, right), np.ndarray[[3, 4], np.dtype[np.float64]])
assert_shape(outer(left, right).shape, (3, 4))
output = np.empty((3, 4), dtype=np.float64)
assert_type(outer(left, right, out=output), np.ndarray[[3, 4], np.dtype[np.float64]])
outer(left, right, out=np.empty((3, 5), dtype=np.float64))  # E: not assignable

matrix = np.ones((3, 2), dtype=np.float64)
weights = np.ones((2, 4), dtype=np.float64)
assert_type(
    tensordot(matrix, weights, axes=1), np.ndarray[[3, 4], np.dtype[np.float64]]
)
tensordot(matrix, np.ones((5, 4), dtype=np.float64), axes=1)  # E: not assignable

values = np.ones((2, 3, 4), dtype=np.float32)
assert_type(roll(values, 1), np.ndarray[[2, 3, 4], np.dtype[np.float32]])
assert_type(
    roll(values, (1, -2), axis=(0, 2)), np.ndarray[[2, 3, 4], np.dtype[np.float32]]
)
