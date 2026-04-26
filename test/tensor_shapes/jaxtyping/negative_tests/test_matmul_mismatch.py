# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test that matmul return type is checked against jaxtyping annotations."""

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from jaxtyping import Shaped
    from torch import Tensor


def matmul_return_mismatch(
    a: Shaped[Tensor, "batch 3 4"],
    b: Shaped[Tensor, "batch 4 5"],
) -> Shaped[Tensor, "batch 3 99"]:
    """Matmul produces batch×3×5, but return says batch×3×99."""
    return torch.matmul(a, b)
