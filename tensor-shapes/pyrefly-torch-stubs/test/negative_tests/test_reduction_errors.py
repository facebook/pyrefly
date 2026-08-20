# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import Tensor


def check_invalid_reduction_dims(x: Tensor[[2, 3, 4]]) -> None:
    torch.sum(x, dim=999)  # E: dimension out of range
    x.std(dim=(0, -3))  # E: duplicate dimension


def check_duplicate_scalar_dims(x: Tensor[[]]) -> None:
    torch.sum(x, dim=(0, -1))  # E: duplicate dimension
