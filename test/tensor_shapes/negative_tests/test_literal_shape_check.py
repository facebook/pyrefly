# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test if type checking works with literal tensor types"""

from typing import reveal_type, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor


def test_literal_shape_mismatch() -> Tensor[4, 3]:
    """This should definitely error - literal shape mismatch"""
    x: Tensor[2, 3] = torch.randn(2, 3)
    reveal_type(x)

    # This should ERROR
    return x  # Tensor[2, 3] not assignable to Tensor[4, 3]


def test_literal_correct() -> Tensor[2, 3]:
    """This should work - literal shapes match"""
    x: Tensor[2, 3] = torch.randn(2, 3)
    return x  # OK
