# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test that Tensor shape is preserved through arithmetic with Any/Unknown.

When a scalar expression evaluates to Any (e.g. 2**n where n is a non-literal
int), arithmetic with a shaped Tensor should preserve the Tensor's shape.
Tensor's arithmetic dunders accept any numeric type and return Self.
"""

from typing import assert_type, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor
    from torch_shapes import Dim


def test_tensor_div_by_unknown(n_bits: int) -> None:
    """Tensor[B, 1] / (2**n - 1.0) should preserve [B, 1]."""
    x: Tensor[4, 1] = torch.randn(4, 1)
    x = 2 * x / (2**n_bits - 1.0) - 1.0
    assert_type(x, Tensor[4, 1])


def test_tensor_mul_by_any(scale: int) -> None:
    """Tensor * Any should preserve shape."""
    x: Tensor[8, 3] = torch.randn(8, 3)
    y = x * (2**scale)
    assert_type(y, Tensor[8, 3])


def test_tensor_add_any(offset: int) -> None:
    """Tensor + Any should preserve shape."""
    x: Tensor[2, 5] = torch.randn(2, 5)
    y = x + (2**offset)
    assert_type(y, Tensor[2, 5])
