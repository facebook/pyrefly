# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast, reveal_type

import torch.nn.functional as F
from torch import Tensor


rank_one = cast(Tensor[[8]], ...)
rank_two = cast(Tensor[[3, 8]], ...)
rank_three = cast(Tensor[[2, 3, 8]], ...)
rank_four = cast(Tensor[[2, 3, 8, 8]], ...)
rank_five = cast(Tensor[[2, 3, 8, 8, 8]], ...)
rank_six = cast(Tensor[[2, 3, 8, 8, 8, 8]], ...)
short_one_dimensional = cast(Tensor[[2, 3, 2]], ...)
short_first_axis = cast(Tensor[[2, 3, 2, 8]], ...)

F.max_pool1d(rank_one, 2)  # E: pooling requires spatial rank + 1 or + 2 input
F.max_pool1d(rank_four, 2)  # E: pooling requires spatial rank + 1 or + 2 input
F.max_pool2d(rank_two, 2)  # E: pooling requires spatial rank + 1 or + 2 input
F.max_pool2d(rank_five, 2)  # E: pooling requires spatial rank + 1 or + 2 input
F.max_pool3d(rank_three, 2)  # E: pooling requires spatial rank + 1 or + 2 input
F.max_pool3d(rank_six, 2)  # E: pooling requires spatial rank + 1 or + 2 input
F.avg_pool1d(rank_one, 2)  # E: pooling requires spatial rank + 1 or + 2 input
F.avg_pool1d(rank_four, 2)  # E: pooling requires spatial rank + 1 or + 2 input
F.avg_pool2d(rank_two, 2)  # E: pooling requires spatial rank + 1 or + 2 input
F.avg_pool2d(rank_five, 2)  # E: pooling requires spatial rank + 1 or + 2 input
F.avg_pool3d(rank_three, 2)  # E: pooling requires spatial rank + 1 or + 2 input
F.avg_pool3d(rank_six, 2)  # E: pooling requires spatial rank + 1 or + 2 input

reveal_type(  # E: revealed type: Tensor[[2, 3, 0]]
    F.max_pool1d(short_one_dimensional, 3)
)
reveal_type(  # E: revealed type: Tensor[[2, 3, 0, 4]]
    F.avg_pool2d(short_first_axis, (3, 2))
)

F.max_pool1d(rank_three, (2, 2))  # E: No matching overload
F.max_pool2d(rank_four, (2,))  # E: No matching overload
F.max_pool2d(rank_four, 2, stride=(2,))  # E: No matching overload
F.max_pool2d(rank_four, 2, padding=(0,))  # E: No matching overload
F.max_pool2d(rank_four, 2, dilation=(1,))  # E: No matching overload
F.max_pool3d(rank_five, (2, 2))  # E: No matching overload
F.avg_pool1d(rank_three, (2, 2))  # E: not a valid `Flag  # E: kernel must match
F.avg_pool2d(rank_four, (2,))  # E: not a valid `Flag  # E: kernel must match
F.avg_pool2d(rank_four, 2, stride=(2,))  # E: not a valid `Flag  # E: stride must match
F.avg_pool2d(  # E: not a valid `Flag  # E: padding must match
    rank_four, 2, padding=(0,)
)
F.avg_pool3d(rank_five, (2, 2))  # E: not a valid `Flag  # E: kernel must match

F.max_pool2d(rank_four, 2.0)  # E: No matching overload
F.max_pool2d(rank_four, (2, 2.0))  # E: No matching overload
F.avg_pool2d(rank_four, 2.0)  # E: not a valid `Flag
F.avg_pool2d(rank_four, (2, 2.0))  # E: not a valid `Flag

F.max_pool2d(rank_four, 0)  # E: pooling kernel must be positive
F.max_pool2d(rank_four, 2, stride=0)  # E: pooling stride must be positive
F.max_pool2d(rank_four, 2, padding=-1)  # E: pooling padding must be nonnegative
F.max_pool2d(rank_four, 2, dilation=0)  # E: pooling dilation must be positive
F.avg_pool2d(rank_four, 0)  # E: pooling kernel must be positive
F.avg_pool2d(rank_four, 2, stride=0)  # E: pooling stride must be positive
F.avg_pool2d(rank_four, 2, padding=-1)  # E: pooling padding must be nonnegative

# Dilation enlarges the effective kernel but does not relax PyTorch's raw-kernel limit.
F.max_pool2d(rank_four, 3, padding=2, dilation=3)  # E: padding must be at most half
F.avg_pool2d(rank_four, 3, padding=2)  # E: padding must be at most half
F.max_pool2d(rank_four, (3, 4), padding=(1, 3))  # E: padding must be at most half
