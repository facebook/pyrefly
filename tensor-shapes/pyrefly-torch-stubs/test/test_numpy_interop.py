# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type, TYPE_CHECKING

import numpy as np
import torch
from shape_extensions import assert_shape
from torch import Tensor


def test_numpy_conversions_preserve_shape() -> None:
    array = np.zeros((2, 3))
    tensor = torch.from_numpy(array)
    assert_shape(tensor.shape, (2, 3))

    converted = tensor.numpy()
    assert_shape(converted.shape, (2, 3))

    if TYPE_CHECKING:
        assert_type(tensor, Tensor[[2, 3]])
        assert_type(converted, np.ndarray[[2, 3]])
