# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type

import numpy as np
from shape_base import expand_dims, kron, tile
from shape_extensions import assert_shape

matrix = np.ones((3, 4), dtype=np.float32)
assert_type(expand_dims(matrix, axis=0), np.ndarray[[1, 3, 4], np.dtype[np.float32]])
assert_shape(expand_dims(matrix, axis=0).shape, (1, 3, 4))
scalar = np.ones((), dtype=np.float64)
assert_type(expand_dims(scalar, axis=0), np.ndarray[[1], np.dtype[np.float64]])

assert_type(tile(matrix, (2, 3)), np.ndarray[[6, 12], np.dtype[np.float32]])
assert_shape(tile(matrix, (0, 3)).shape, (0, 12))
tile(matrix, (2, 3, 4))  # E: not assignable

left = np.ones((2, 3), dtype=np.float64)
right = np.ones((4, 5), dtype=np.float64)
assert_type(kron(left, right), np.ndarray[[8, 15], np.dtype[np.float64]])
kron(left, np.ones((4,), dtype=np.float64))  # E: not assignable
