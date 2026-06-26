# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test different slicing patterns"""

from typing import assert_type, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor


def test_simple_index[B, T, V](logits: Tensor[B, T, V]):
    """Test slicing with positive index"""
    temp = logits[:, 0, :]
    assert_type(temp, Tensor[B, V])


def test_slice_only[B, T, V](logits: Tensor[B, T, V]):
    """Test slicing without index"""
    temp = logits[:, :, :]
    assert_type(temp, Tensor[B, T, V])
