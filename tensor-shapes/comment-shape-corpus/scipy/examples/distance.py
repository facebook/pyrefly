# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type

import numpy as np
from distance import cdist, pdist
from shape_extensions import assert_shape

left = np.ones((3, 2), dtype=np.float64)
right = np.ones((4, 2), dtype=np.float64)
assert_shape(cdist(left, right).shape, (3, 4))
assert_shape(cdist(left, right, out=np.empty((3, 4), dtype=np.float64)).shape, (3, 4))
assert_shape(pdist(left).shape, (3,))
assert_shape(pdist(right).shape, (6,))
assert_type(cdist(left, right), np.ndarray[[3, 4], np.dtype[np.float64]])
assert_type(pdist(right), np.ndarray[[6], np.dtype[np.float64]])

# The observation counts can differ; the feature dimension must agree.
cdist(left, np.ones((5, 7), dtype=np.float64))  # E: not assignable
ndims = np.ones((3,), dtype=np.float64)
pdist(ndims)  # E: not assignable
cdist(left, right, out=np.empty((3, 5), dtype=np.float64))  # E: not assignable
