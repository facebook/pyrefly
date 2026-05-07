# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test comparison operator shape preservation"""

from typing import assert_type, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor


def test_comparison(x: Tensor[2, 3]):
    y = x == 0
    assert_type(y, Tensor[2, 3])
