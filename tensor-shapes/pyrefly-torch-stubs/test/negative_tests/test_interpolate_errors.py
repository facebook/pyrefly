# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


rank_two = cast(Tensor[[3, 8]], ...)
rank_three = cast(Tensor[[2, 3, 8]], ...)
rank_four = cast(Tensor[[2, 3, 8, 8]], ...)
rank_six = cast(Tensor[[2, 3, 8, 8, 8, 8]], ...)

F.interpolate(rank_two, 2)  # E: interpolate requires rank 3, 4, or 5
F.interpolate(rank_six, 2)  # E: interpolate requires rank 3, 4, or 5
F.upsample(rank_two, 2)  # E: interpolate requires rank 3, 4, or 5
F.upsample(rank_six, 2)  # E: interpolate requires rank 3, 4, or 5

F.interpolate(rank_three, (2, 3))  # E: size must match the spatial rank
F.interpolate(rank_four, (2,))  # E: size must match the spatial rank
F.interpolate(rank_four, scale_factor=(2,))  # E: scale_factor must match
F.upsample(rank_three, (2, 3))  # E: size must match the spatial rank
F.upsample(rank_four, (2,))  # E: size must match the spatial rank
F.upsample(rank_four, scale_factor=(2,))  # E: scale_factor must match

F.interpolate(rank_four, (0, 2))  # E: interpolate size must be positive
F.interpolate(rank_four, scale_factor=(2, -1))  # E: scale_factor must be positive
F.upsample(rank_four, (2, 0))  # E: interpolate size must be positive
F.upsample(rank_four, scale_factor=(-1, 2))  # E: scale_factor must be positive

F.interpolate(rank_four)  # E: interpolate requires size or scale_factor
F.interpolate(rank_four, None, None)  # E: interpolate requires size or scale_factor
F.interpolate(rank_four, 2, 2)  # E: accepts only one of size or scale_factor
F.upsample(rank_four)  # E: interpolate requires size or scale_factor
F.upsample(rank_four, None, None)  # E: interpolate requires size or scale_factor
F.upsample(rank_four, 2, 2)  # E: accepts only one of size or scale_factor

F.interpolate(rank_four, size=2.0)  # E: No matching overload
F.upsample(rank_four, size=(2, 2.0))  # E: No matching overload
F.upsample(rank_four, size=2, recompute_scale_factor=False)  # E: Unexpected keyword
F.upsample(rank_four, size=2, antialias=True)  # E: Unexpected keyword

nn.Upsample()(rank_four)  # E: interpolate requires size or scale_factor
nn.Upsample(size=2, scale_factor=2)(rank_four)  # E: accepts only one
# A float `size` is invalid at runtime; only `scale_factor` may be a float.
nn.Upsample(size=1.5)  # E: No matching overload
nn.Upsample(size=2)(rank_two)  # E: interpolate requires rank 3, 4, or 5
