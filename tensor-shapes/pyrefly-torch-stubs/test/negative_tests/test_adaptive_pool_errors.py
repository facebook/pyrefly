# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from torch import Tensor
from torch.nn import functional as F


def test_invalid_adaptive_pool_output_arity(x: Tensor[[2, 3, 12, 15]]) -> None:
    F.adaptive_avg_pool2d(x, (2,))  # E: No matching overload
    F.adaptive_avg_pool2d(x, None)  # E: No matching overload
    F.adaptive_max_pool2d(x, None)  # E: No matching overload


def test_invalid_adaptive_pool3d_whole_none(x: Tensor[[2, 3, 8, 12, 15]]) -> None:
    F.adaptive_avg_pool3d(x, None)  # E: No matching overload
    F.adaptive_max_pool3d(x, None, return_indices=True)  # E: No matching overload


def test_invalid_adaptive_pool_input_rank(
    one_d: Tensor[[12]],
    two_d: Tensor[[12, 15]],
    three_d: Tensor[[8, 12, 15]],
    four_d: Tensor[[2, 8, 12, 15]],
    five_d: Tensor[[2, 3, 8, 12, 15]],
    six_d: Tensor[[2, 3, 4, 8, 12, 15]],
) -> None:
    F.adaptive_avg_pool1d(one_d, 4)  # E: adaptive_pool1d requires 2D or 3D input
    F.adaptive_avg_pool1d(four_d, 4)  # E: adaptive_pool1d requires 2D or 3D input
    F.adaptive_avg_pool2d(two_d, 4)  # E: adaptive_pool2d requires 3D or 4D input
    F.adaptive_avg_pool2d(five_d, 4)  # E: adaptive_pool2d requires 3D or 4D input
    F.adaptive_max_pool3d(three_d, 4)  # E: adaptive_pool3d requires 4D or 5D input
    F.adaptive_max_pool3d(six_d, 4)  # E: adaptive_pool3d requires 4D or 5D input


def test_invalid_adaptive_pool_fallback_input_rank(
    one_d: Tensor[[12]],
    two_d: Tensor[[12, 15]],
    three_d: Tensor[[8, 12, 15]],
    six_d: Tensor[[2, 3, 4, 8, 12, 15]],
    return_indices: bool,
) -> None:
    F.adaptive_max_pool1d(  # E: adaptive_pool1d requires 2D or 3D input
        one_d, 4, return_indices=True
    )
    F.adaptive_max_pool2d(  # E: adaptive_pool2d requires 3D or 4D input
        two_d, (None, 4)
    )
    F.adaptive_max_pool2d(  # E: adaptive_pool2d requires 3D or 4D input
        two_d, 4, return_indices=return_indices
    )
    F.adaptive_avg_pool2d(  # E: adaptive_pool2d requires 3D or 4D input
        two_d, (None, 4)
    )
    F.adaptive_max_pool3d(  # E: adaptive_pool3d requires 4D or 5D input
        three_d, (None, 4, None), return_indices=True
    )
    F.adaptive_max_pool3d(  # E: adaptive_pool3d requires 4D or 5D input
        six_d, (None, 4, None), return_indices=False
    )
    F.adaptive_avg_pool3d(  # E: adaptive_pool3d requires 4D or 5D input
        six_d, (None, 4, None)
    )
