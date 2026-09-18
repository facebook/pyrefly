# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

import numpy as np
from jax._array import Array
from jax._src.typing import DTypeLike as DTypeLike

type ArrayLike = (
    Array[Any] | np.ndarray[Any] | np.bool_ | np.number | bool | int | float | complex
)
