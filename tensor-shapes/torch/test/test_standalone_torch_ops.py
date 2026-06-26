# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test standalone torch function operations with generic TypeVarTuple signatures."""

from typing import assert_type, cast

import torch
from torch import Tensor


def test_arithmetic_functions() -> None:
    """Test arithmetic functions preserve shape via generic signatures."""
    x = cast(Tensor[2, 3], ...)
    y = cast(Tensor, ...)

    # Test updated arithmetic functions
    assert_type(torch.add(x, y), Tensor[2, 3])
    assert_type(torch.sub(x, y), Tensor[2, 3])
    assert_type(torch.mul(x, y), Tensor[2, 3])
    assert_type(torch.div(x, y), Tensor[2, 3])
    assert_type(torch.pow(x, y), Tensor[2, 3])


def test_unary_functions() -> None:
    """Test unary functions preserve shape via generic signatures."""
    x = cast(Tensor[4, 5], ...)

    assert_type(torch.neg(x), Tensor[4, 5])
    assert_type(torch.abs(x), Tensor[4, 5])
    assert_type(torch.floor(x), Tensor[4, 5])
    assert_type(torch.ceil(x), Tensor[4, 5])
    assert_type(torch.round(x), Tensor[4, 5])


def test_math_functions() -> None:
    """Test math functions preserve shape via generic signatures."""
    x = cast(Tensor[3, 4], ...)

    assert_type(torch.sin(x), Tensor[3, 4])
    assert_type(torch.cos(x), Tensor[3, 4])
    assert_type(torch.tan(x), Tensor[3, 4])
    assert_type(torch.exp(x), Tensor[3, 4])
    assert_type(torch.log(x), Tensor[3, 4])
    assert_type(torch.sqrt(x), Tensor[3, 4])
    assert_type(torch.tanh(x), Tensor[3, 4])


def test_bitwise_functions() -> None:
    """Test bitwise functions preserve shape via generic signatures."""
    x = cast(Tensor[2, 4], ...)
    y = cast(Tensor, ...)

    assert_type(torch.bitwise_and(x, y), Tensor[2, 4])
    assert_type(torch.bitwise_or(x, y), Tensor[2, 4])
    assert_type(torch.bitwise_xor(x, y), Tensor[2, 4])
    assert_type(torch.bitwise_not(x), Tensor[2, 4])
