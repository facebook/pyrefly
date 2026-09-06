# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import Tensor


def check_chunk_errors(x: Tensor[[2, 3, 4]]) -> None:
    x.chunk(0, dim=1)  # E: chunk count must be greater than zero
    torch.chunk(x, -1, dim=1)  # E: chunk count must be greater than zero
    x.chunk(2, dim=3)  # E: chunk dimension out of range
    torch.chunk(x, 2, dim=-4)  # E: chunk dimension out of range
