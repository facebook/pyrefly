# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""`torch.where` keyword forms PyTorch rejects at runtime."""

import torch
from torch import Tensor


def check_where_keyword_errors(cond: Tensor, x: Tensor) -> None:
    # A scalar value parameter is named `self` at runtime, not `input`.
    # E: No matching overload
    torch.where(cond, input=0.0, other=x)
    # `out=` is only accepted when both values are tensors.
    # E: Argument `float` is not assignable to parameter `other`
    torch.where(cond, x, 0.0, out=x)
