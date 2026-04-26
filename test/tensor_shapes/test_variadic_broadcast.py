# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test that broadcasting works between tensors with different variadic middles.

When two tensors have incompatible variadic middles (different TypeVarTuples),
broadcasting should degrade to shapeless batch dims rather than erroring.
The concrete suffix dims should still be broadcast correctly.
"""

from typing import assert_type, TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from torch import Tensor
    from torch_shapes import Dim


# --- Bare inputs: two Linear.forward calls produce different *Bs ---


class Fuser(nn.Module):
    """Two Linear projections added together."""

    def __init__(self) -> None:
        super().__init__()
        self.w_x = nn.Linear(256, 128)
        self.w_y = nn.Linear(256, 128)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        # w_x and w_y each produce Tensor[*BsN, 128] with different *BsN.
        # Addition should degrade batch dims rather than erroring.
        return self.w_x(x) + self.w_y(y)


def test_add_bare_linear_outputs():
    """Adding two Linear outputs from bare inputs doesn't error."""
    m = Fuser()
    x: Tensor[4, 256] = torch.randn(4, 256)
    y: Tensor[4, 256] = torch.randn(4, 256)
    m(x, y)


# --- Shaped inputs: explicitly variadic with different TypeVarTuples ---


def add_different_variadics[*As, *Bs, C](
    a: Tensor[*As, C], b: Tensor[*Bs, C]
) -> Tensor:
    """Add two tensors with different variadic batch dims.
    The suffix C is shared; the middles *As and *Bs differ.
    Should degrade to shapeless batch + broadcast suffix."""
    return a + b


def test_add_shaped_different_variadics():
    """Adding tensors with different *As, *Bs preserves suffix."""
    a: Tensor[2, 3, 10] = torch.randn(2, 3, 10)
    b: Tensor[4, 5, 10] = torch.randn(4, 5, 10)
    out = add_different_variadics(a, b)


# --- Same variadic: should still broadcast correctly ---


def add_same_variadic[*Bs, C](a: Tensor[*Bs, C], b: Tensor[*Bs, C]) -> Tensor[*Bs, C]:
    """Add two tensors with the same variadic and same suffix.
    Since *Bs and C match, the result preserves the full shape."""
    return a + b


def test_add_same_variadic():
    """Adding tensors with same *Bs and same suffix works."""
    a: Tensor[2, 3, 10] = torch.randn(2, 3, 10)
    b: Tensor[2, 3, 10] = torch.randn(2, 3, 10)
    out = add_same_variadic(a, b)
    assert_type(out, Tensor[2, 3, 10])


# --- Mul with different variadics ---


def mul_different_variadics[*As, *Bs, C](
    a: Tensor[*As, C], b: Tensor[*Bs, C]
) -> Tensor:
    """Multiply tensors with different variadic middles.
    Should degrade gracefully like addition."""
    return a * b


def test_mul_different_variadics():
    """Multiplying tensors with different *As, *Bs doesn't error."""
    a: Tensor[2, 3, 10] = torch.randn(2, 3, 10)
    b: Tensor[4, 5, 10] = torch.randn(4, 5, 10)
    mul_different_variadics(a, b)
