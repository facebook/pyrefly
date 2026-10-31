# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type

import numpy as np
from shape_extensions import assert_shape
from twodim_base import diag, tril, triu

vector = np.ones((4,), dtype=np.float32)
matrix = diag(vector)
assert_type(matrix, np.ndarray[[4, 4], np.dtype[np.float32]])
assert_type(diag(matrix), np.ndarray[[4], np.dtype[np.float32]])
assert_shape(diag(matrix).shape, (4,))

batch = np.ones((2, 3, 4), dtype=np.float64)
assert_type(tril(batch), np.ndarray[[2, 3, 4], np.dtype[np.float64]])
assert_type(triu(batch, k=-1), np.ndarray[[2, 3, 4], np.dtype[np.float64]])

# These cases are intentionally outside the excerpt's precise overloads.
diag(np.ones((3, 4), dtype=np.float64))  # E: not assignable
diag(batch)  # E: not assignable
