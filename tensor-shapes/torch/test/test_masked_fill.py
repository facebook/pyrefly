# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test masked_fill broadcasting behavior"""

from typing import assert_type, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor


def test_masked_fill_broadcasting[B, NHead, T](
    att: Tensor[B, NHead, T, T], mask: Tensor[1, 1, T, T]
):
    # mask == 0 should produce Tensor[1, 1, T, T] (bool)
    mask_bool = mask == 0
    assert_type(mask_bool, Tensor[1, 1, T, T])

    # masked_fill should preserve att's shape
    result = att.masked_fill(mask == 0, float("-inf"))
    assert_type(result, Tensor[B, NHead, T, T])
