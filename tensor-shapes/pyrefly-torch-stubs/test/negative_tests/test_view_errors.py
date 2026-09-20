# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test flatten validation errors."""

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from torch import Tensor


def test_flatten_dimension_errors():
    x: Tensor[[2, 3, 4]] = torch.randn(2, 3, 4)
    # E: flatten start_dim out of range
    x.flatten(3)
    # E: flatten end_dim out of range
    x.flatten(0, -4)
    # E: flatten start_dim cannot come after end_dim
    torch.flatten(x, 2, 1)
    scalar: Tensor[[]] = torch.tensor(1)
    # E: flatten dimension out of range for scalar input
    scalar.flatten(1)
    # E: flatten start_dim out of range
    nn.Flatten(3)(x)
    # E: flatten dimension out of range for scalar input
    nn.Flatten()(scalar)
