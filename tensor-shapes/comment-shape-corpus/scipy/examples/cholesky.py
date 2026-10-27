# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type

import numpy as np
from cholesky import cho_factor, cho_solve, cholesky
from shape_extensions import assert_shape

matrix = np.eye(3, dtype=np.float64)
assert_shape(cholesky(matrix).shape, (3, 3))

batches = np.ones((2, 1, 1), dtype=np.float64) * matrix
assert_shape(cholesky(batches).shape, (2, 3, 3))
assert_type(cholesky(batches), np.ndarray[[2, 3, 3], np.dtype[np.float64]])

factor = cho_factor(matrix, lower=True)
assert_shape(factor[0].shape, (3, 3))
assert_shape(factor[1].shape, ())
batched_factor = cho_factor(batches)
assert_shape(batched_factor[0].shape, (2, 3, 3))
assert_shape(batched_factor[1].shape, (2,))
assert_shape(cho_solve(factor, np.ones((3,), dtype=np.float64)).shape, (3,))
assert_shape(cho_solve(factor, np.ones((3, 2), dtype=np.float64)).shape, (3, 2))

# Shape checks do not establish positive definiteness or numerical validity.
cholesky(np.ones((3, 4), dtype=np.float64))  # E: not assignable
cho_solve(factor, np.ones((4,), dtype=np.float64))  # E: not assignable
