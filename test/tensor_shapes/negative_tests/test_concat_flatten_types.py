# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test concat and flatten with reveal_type to see actual return types"""

from typing import reveal_type, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor


def concat_symbolic[N, M](x: Tensor[N, 3], y: Tensor[M, 3]) -> Tensor[N + M, 3]:
    """Concat with symbolic dimension addition: N + M"""
    reveal_type(x)
    reveal_type(y)
    z = torch.cat((x, y), dim=0)
    reveal_type(z)
    return z


def flatten_symbolic[B, N, M](x: Tensor[B, N, M]) -> Tensor[B * N * M]:
    """Flatten with symbolic dimension multiplication"""
    reveal_type(x)
    return x.flatten()


def test_concat_what_is_actual_type() -> Tensor[100, 3]:
    """What type does concat actually return?"""
    x: Tensor[2, 3] = torch.randn(2, 3)
    y: Tensor[5, 3] = torch.randn(5, 3)
    z = concat_symbolic(x, y)
    reveal_type(z)  # Expected: Tensor[7, 3], but might be Tensor[N + M, 3]?

    # Test if wrong type is accepted
    return z  # Should ERROR if z is Tensor[7, 3]


def test_flatten_what_is_actual_type() -> Tensor[999]:
    """What type does flatten actually return?"""
    x: Tensor[2, 3, 4] = torch.randn(2, 3, 4)
    y = flatten_symbolic(x)
    reveal_type(y)  # Expected: Tensor[24], but might be Tensor[B * N * M]?

    # Test if wrong type is accepted
    return y  # Should ERROR if y is Tensor[24]
