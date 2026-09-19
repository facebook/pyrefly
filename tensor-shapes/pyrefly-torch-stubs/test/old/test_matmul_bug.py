# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Investigate @ operator bug with symbolic dimensions
"""

from typing import assert_type, TYPE_CHECKING

import torch
from shape_extensions import IntVar

if TYPE_CHECKING:
    from torch import Tensor


# Test 1: Does transpose preserve shapes?
def test_transpose[N: IntVar, M: IntVar](x: Tensor[[N, M]]):
    """Check if transpose returns correct shape"""
    y = x.transpose(0, 1)
    # Should be [M, N]
    assert_type(y, Tensor[[M, N]])


test_transpose(torch.randn(5, 10))
