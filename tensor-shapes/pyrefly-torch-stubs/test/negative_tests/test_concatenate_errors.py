# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import assert_type

import torch
from torch import Tensor


def test_axis_keyword() -> None:
    result = torch.concatenate((torch.zeros((2, 3)), torch.zeros((2, 4))), axis=1)
    assert_type(result, Tensor[[2, 7]])
