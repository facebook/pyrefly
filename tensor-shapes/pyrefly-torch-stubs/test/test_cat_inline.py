# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test cat with inline tuple literal"""

from typing import assert_type, TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from torch import Tensor


def test_cat_inline(idx: Tensor, idx_next: Tensor):
    """Test with inline tuple like nanogpt"""
    assert_type(idx, Tensor)
    assert_type(idx_next, Tensor)
    # Use inline tuple exactly like nanogpt
    result = torch.cat((idx, idx_next), dim=1)
    assert_type(result, Tensor)
