# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test standalone torch function operations with generic TypeVarTuple signatures."""

from typing import assert_type, cast

import torch
from torch import Tensor


def test_unary_functions() -> None:
    """Test unary functions preserve shape via generic signatures."""
    x = cast(Tensor[[4, 5]], ...)

    assert_type(torch.neg(x), Tensor[[4, 5]])
    assert_type(torch.abs(x), Tensor[[4, 5]])
    assert_type(torch.floor(x), Tensor[[4, 5]])
    assert_type(torch.ceil(x), Tensor[[4, 5]])
    assert_type(torch.round(x), Tensor[[4, 5]])


def test_math_functions() -> None:
    """Test math functions preserve shape via generic signatures."""
    x = cast(Tensor[[3, 4]], ...)

    assert_type(torch.sin(x), Tensor[[3, 4]])
    assert_type(torch.cos(x), Tensor[[3, 4]])
    assert_type(torch.tan(x), Tensor[[3, 4]])
    assert_type(torch.exp(x), Tensor[[3, 4]])
    assert_type(torch.log(x), Tensor[[3, 4]])
    assert_type(torch.sqrt(x), Tensor[[3, 4]])
    assert_type(torch.tanh(x), Tensor[[3, 4]])
