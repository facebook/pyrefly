# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from jax._array import Array as Array

type StaticScalar = np.bool_ | np.number | bool | int | float | complex
type ArrayLike = Array | np.ndarray | StaticScalar
