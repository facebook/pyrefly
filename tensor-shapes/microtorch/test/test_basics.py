# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import assert_type

import microtorch as torch
from microtorch import Tensor


def test_basic_shapes() -> None:
    left = torch.randn((2, 3))
    right = torch.ones((3, 5))

    assert_type(left + left, Tensor[[2, 3]])
    assert_type(left @ right, Tensor[[2, 5]])
    assert_type(left.transpose(), Tensor[[3, 2]])
    assert_type(torch.diagonal(torch.ones((4,)), offset=1), Tensor[[5, 5]])
    assert_type(
        torch.concatenate((torch.zeros((2, 3)), torch.ones((2, 4))), axis=1),
        Tensor[[2, 7]],
    )
