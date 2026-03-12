# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test slicing with negative indices"""

from typing import assert_type, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor


def test_negative_slice[B, T, V](logits: Tensor[B, T, V]):
    """Test slicing with negative index"""
    temp = logits[:, -1, :]
    assert_type(temp, Tensor[B, V])
