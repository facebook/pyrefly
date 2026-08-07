# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests that jaxtyping shapes propagate correctly through operations.

Covers 4 ops across meta-shape and fixture mechanisms:
- torch.matmul (meta-shape): named, integer, mixed dims
- x.sin() (fixture, Self return): named, integer, mixed, arithmetic dims
- x.view() (meta-shape, -1 inference): named + mixed dims
- torch.det() (fixture, shape-transforming): named dims
"""

from typing import assert_type

import torch
from jaxtyping import Float, Shaped
from torch import Tensor


# --- Matmul (meta-shape op) ---


def test_matmul_named(
    a: Shaped[Tensor, "batch m n"],
    b: Shaped[Tensor, "batch n p"],
) -> None:
    """Matmul with named dims: batch dimension preserved, inner dim contracted."""
    result = torch.matmul(a, b)
    assert_type(result, Shaped[Tensor, "batch m p"])


def test_matmul_integer(
    a: Shaped[Tensor, "3 4"],
    b: Shaped[Tensor, "4 5"],
) -> None:
    """Matmul with integer dims: 3x4 @ 4x5 = 3x5."""
    result = torch.matmul(a, b)
    assert_type(result, Shaped[Tensor, "3 5"])


def test_matmul_mixed(
    a: Shaped[Tensor, "batch 3 4"],
    b: Shaped[Tensor, "batch 4 5"],
) -> None:
    """Matmul with mixed dims: named batch, integer inner dims."""
    result = torch.matmul(a, b)
    assert_type(result, Shaped[Tensor, "batch 3 5"])


# --- Sin (fixture op, Self return) ---


def test_sin_named(x: Float[Tensor, "batch channels"]) -> None:
    """Sin preserves named dims via Self return."""
    result = x.sin()
    assert_type(result, Shaped[Tensor, "batch channels"])


def test_sin_integer(x: Float[Tensor, "3 4"]) -> None:
    """Sin preserves integer dims via Self return."""
    result = x.sin()
    assert_type(result, Shaped[Tensor, "3 4"])


def test_sin_mixed(x: Float[Tensor, "batch 3"]) -> None:
    """Sin preserves mixed dims via Self return."""
    result = x.sin()
    assert_type(result, Shaped[Tensor, "batch 3"])


def test_sin_arithmetic(x: Shaped[Tensor, "n n+1"]) -> None:
    """Sin preserves arithmetic dims via Self return."""
    result = x.sin()
    assert_type(result, Shaped[Tensor, "n n+1"])


# --- View (meta-shape op, -1 inference) ---


def test_view_named(x: Shaped[Tensor, "batch 6"]) -> None:
    """View with -1 inference: named leading dim preserved, trailing split."""
    result = x.view(-1, 2, 3)
    assert_type(result, Shaped[Tensor, "batch 2 3"])


def test_view_mixed(x: Shaped[Tensor, "batch 3 4"]) -> None:
    """View with -1 inference: named leading dim, flatten trailing dims."""
    result = x.view(-1, 12)
    assert_type(result, Shaped[Tensor, "batch 12"])


# --- Det (fixture op, shape-transforming) ---


def test_det_named(x: Shaped[Tensor, "batch m n"]) -> None:
    """Det drops trailing 2 dims: [batch, m, n] -> [batch]."""
    result = torch.det(x)
    assert_type(result, Shaped[Tensor, "batch"])
