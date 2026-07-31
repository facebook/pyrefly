# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test gradual Tensor arithmetic operands.

When a scalar expression evaluates to Any (e.g. 2**n where n is a non-literal
int), it could also be an unknown-rank Tensor, so broadcasting is gradual.
"""

from typing import assert_type, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from shape_extensions import IntTuple
    from torch import Tensor


def test_tensor_div_by_unknown(n_bits: int) -> None:
    x: Tensor[[4, 1]] = torch.randn(4, 1)
    x = 2 * x / (2**n_bits - 1.0) - 1.0
    assert_type(x, Tensor[IntTuple])


def test_tensor_mul_by_any(scale: int) -> None:
    x: Tensor[[8, 3]] = torch.randn(8, 3)
    y = x * (2**scale)
    assert_type(y, Tensor[IntTuple])


def test_tensor_add_any(offset: int) -> None:
    x: Tensor[[2, 5]] = torch.randn(2, 5)
    y = x + (2**offset)
    assert_type(y, Tensor[IntTuple])
