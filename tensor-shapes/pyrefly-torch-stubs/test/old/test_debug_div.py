# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test Tensor arithmetic with Any and nonliteral scalar expressions."""

from __future__ import annotations

from typing import Any, assert_type

import torch
from torch import Tensor


def test_tensor_arithmetic_with_any(x: Tensor[[2, 5]], other: Any) -> None:
    assert_type(x + other, Any)
    assert_type(other + x, Any)


def test_tensor_div_by_float_expression(n_bits: int) -> None:
    x: Tensor[[4, 1]] = torch.randn(4, 1)
    y = 2 * x / (2 ** (n_bits * 1.0) - 1.0) - 1.0
    assert_type(y, Tensor[[4, 1]])


def test_tensor_mul_by_float_expression(scale: int) -> None:
    x: Tensor[[8, 3]] = torch.randn(8, 3)
    y = x * (2 ** (scale * 1.0))
    assert_type(y, Tensor[[8, 3]])


def test_tensor_add_float_expression(offset: int) -> None:
    x: Tensor[[2, 5]] = torch.randn(2, 5)
    y = x + (2 ** (offset * 1.0))
    assert_type(y, Tensor[[2, 5]])
