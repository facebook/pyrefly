# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for jaxtyping annotation parsing: Float[Tensor, "batch channels"]"""

from typing import assert_type

import torch
from jaxtyping import Float, Shaped
from torch import Tensor


def test_concrete_dims():
    """Fixed integer dimensions in jaxtyping annotations."""
    x = torch.randn(3, 4)
    assert_type(x, Shaped[Tensor, "3 4"])


def test_named_dims(
    x: Shaped[Tensor, "batch channels"],
) -> Shaped[Tensor, "batch channels"]:
    """Named dimensions are consistent across parameter and return type."""
    assert_type(x, Shaped[Tensor, "batch channels"])
    return x


def test_mixed_dims(x: Float[Tensor, "batch 3"]) -> Float[Tensor, "batch 3"]:
    """Mix of named and integer dimensions."""
    assert_type(x, Shaped[Tensor, "batch 3"])
    return x


def test_matmul_shapes(
    a: Shaped[Tensor, "batch m n"],
    b: Shaped[Tensor, "batch n p"],
) -> Shaped[Tensor, "batch m p"]:
    """Matrix multiply with named batch, m, n, p dimensions."""
    result = torch.matmul(a, b)
    assert_type(result, Shaped[Tensor, "batch m p"])
    return result


def test_single_dim(x: Shaped[Tensor, "features"]) -> Shaped[Tensor, "features"]:  # noqa: F821
    """Single named dimension."""
    assert_type(x, Shaped[Tensor, "features"])
    return x


def test_scalar(x: Shaped[Tensor, ""]) -> Shaped[Tensor, ""]:
    """Empty shape string means scalar tensor (rank 0)."""
    assert_type(x, Shaped[Tensor, ""])
    return x
