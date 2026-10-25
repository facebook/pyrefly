# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type, TYPE_CHECKING

import numpy as np


def test_float32_ndarray_multiplication_by_series() -> None:
    if TYPE_CHECKING:
        import pandas as pd

        def multiply(
            array: np.ndarray[[int], np.dtype[np.float32]],
            series: pd.Series[float],
        ) -> None:
            assert_type(array * series, pd.Series[float])
