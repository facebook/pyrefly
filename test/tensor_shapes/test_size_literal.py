# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test if size(-1) works with literal"""

from typing import assert_type, Literal, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor


def test_size_with_negative():
    """Test size with -1"""
    x: Tensor[10, 20, 30] = torch.randn(10, 20, 30)
    s = x.size(-1)
    assert_type(s, Literal[30])  # Should be int
