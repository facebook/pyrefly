# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test if type checking works with literal tensor types"""

from typing import assert_type, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor


def test_literal_shape_mismatch() -> Tensor[4, 3]:
    """Literal shape mismatch is rejected."""
    x: Tensor[2, 3] = torch.randn(2, 3)
    assert_type(x, Tensor[2, 3])

    # E: Returned type `Tensor[2, 3]` is not assignable
    #    to declared return type `Tensor[4, 3]`
    return x


def test_literal_correct() -> Tensor[2, 3]:
    """Literal shapes match."""
    x: Tensor[2, 3] = torch.randn(2, 3)
    return x
