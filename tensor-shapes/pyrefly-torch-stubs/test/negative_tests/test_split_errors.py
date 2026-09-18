# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import Tensor


def check_split_errors(x: Tensor[[2, 3, 4]]) -> None:
    x.split(1, dim=3)  # E: split dimension out of range
    torch.split(x, (1, 2), dim=-4)  # E: split dimension out of range
    x.split(0, dim=1)  # E: split size can only be zero
    torch.split(x, -1, dim=1)  # E: split size must be non-negative
    x.split((1, -1, 3), dim=1)  # E: split sections must be non-negative
    torch.split(x, (1, -1, 3), dim=1)  # E: split sections must be non-negative
    x.split((1, 1), dim=1)  # E: split sections must sum to the selected dimension
    torch.split(  # E: split sections must sum to the selected dimension
        x, (1, 1), dim=1
    )
