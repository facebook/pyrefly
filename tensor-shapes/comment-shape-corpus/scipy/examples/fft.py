# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type

import numpy as np
from fft_helper import fftfreq, fftshift, ifftshift, rfftfreq
from shape_extensions import assert_shape

assert_shape(fftfreq(17).shape, (17,))
assert_shape(rfftfreq(17).shape, (9,))
assert_shape(rfftfreq(16).shape, (9,))
assert_type(fftfreq(17), np.ndarray[[17], np.dtype[np.float64]])
assert_type(rfftfreq(17), np.ndarray[[9], np.dtype[np.float64]])

values = np.ones((2, 3, 4), dtype=np.float32)
shifted = fftshift(values, axes=(0, 2))
assert_shape(shifted.shape, (2, 3, 4))
assert_type(shifted, np.ndarray[[2, 3, 4], np.dtype[np.float32]])
assert_type(ifftshift(shifted), np.ndarray[[2, 3, 4], np.dtype[np.float32]])
