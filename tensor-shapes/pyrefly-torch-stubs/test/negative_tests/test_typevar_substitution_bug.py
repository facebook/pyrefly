# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""More detailed test of type variable substitution"""

from typing import assert_type, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor


def accepts_symbolic_returns_symbolic[N](x: Tensor[N, 3]) -> Tensor[N, 3]:
    """Identity function with symbolic dimension - preserves shape"""
    return x


def test_detailed():
    """Check types at each step"""
    # Step 1: Create concrete tensor
    x_concrete: Tensor[2, 3] = torch.randn(2, 3)
    assert_type(x_concrete, Tensor[2, 3])

    # Step 2: Call function
    result = accepts_symbolic_returns_symbolic(x_concrete)
    assert_type(result, Tensor[2, 3])

    # Step 3: The concrete shape is preserved, so mismatched first dimensions are rejected.
    case1: Tensor[2, 3] = result
    # E: `Tensor[2, 3]` is not assignable to `Tensor[4, 3]`
    case2: Tensor[4, 3] = result
    # E: `Tensor[2, 3]` is not assignable to `Tensor[100, 3]`
    case3: Tensor[100, 3] = result
    _ = (case1, case2, case3)
