# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Phase 6: Specialized operations tests (FFT, Loss, Padding, Random, Properties)
from typing import assert_type, Literal

import torch
import torch.fft
import torch.nn
from torch import Tensor

# ==== FFT Operations ====


# ==== Tensor Property Operations ====


def test_numel():
    """Number of elements"""
    x: Tensor[[3, 4, 5]] = torch.randn(3, 4, 5)
    result = torch.numel(x)
    # Returns int (symbolic multiplication of dimensions)
    assert_type(result, Literal[60])
