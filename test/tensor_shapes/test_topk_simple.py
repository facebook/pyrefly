# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test topk with shaped tensor"""

from typing import assert_type, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor


def test_topk_basic[B, V](logits: Tensor[B, V]):
    """Test topk with a shaped tensor"""
    v, _ = torch.topk(logits, 5)
    assert_type(v, Tensor[B, 5])
