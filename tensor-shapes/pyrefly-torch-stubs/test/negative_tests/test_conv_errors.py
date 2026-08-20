# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from torch import Tensor
from torch.nn import functional as F


def test_zero_stride(input: Tensor[[2, 3, 10]], weight: Tensor[[4, 3, 3]]) -> None:
    F.conv1d(input, weight, stride=0)  # E: division by zero


def test_mismatched_input_and_weight_ranks(
    input: Tensor[[2, 3, 10, 20]], weight: Tensor[[4, 3, 3]]
) -> None:
    # E: convolution input and weight must have the same rank
    F.conv2d(input, weight)


def test_wrong_parameter_tuple_lengths(
    input1: Tensor[[2, 3, 10]],
    weight1: Tensor[[4, 3, 3]],
    input2: Tensor[[2, 3, 10, 20]],
    weight2: Tensor[[4, 3, 3, 5]],
    input3: Tensor[[2, 3, 8, 10, 12]],
    weight3: Tensor[[4, 3, 3, 3, 3]],
) -> None:
    F.conv1d(input1, weight1, stride=())  # E: not a valid `Flag[
    F.conv2d(input2, weight2, dilation=(1,))  # E: not a valid `Flag[
    F.conv3d(input3, weight3, padding=(0, 0, 0, 0))  # E: not a valid `Flag[
    # E: not a valid `Flag[
    F.conv_transpose1d(input1, weight1, output_padding=(0, 0))
    F.conv_transpose2d(input2, weight2, padding=())  # E: not a valid `Flag[
    # E: not a valid `Flag[
    F.conv_transpose3d(input3, weight3, stride=(1, 1))
