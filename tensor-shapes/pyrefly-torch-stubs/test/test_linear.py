# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type, TYPE_CHECKING

import torch
import torch.nn as nn
from shape_extensions import (
    assert_raises,
    assert_shape,
    Elements,
    Int,
    IntTuple,
    IntVar,
)
from torch import Tensor


def test_linear_shapes() -> None:
    linear = nn.Linear(3, 4)
    assert_shape(linear(torch.ones(3)).shape, (4,))
    assert_shape(linear(torch.ones((2, 3))).shape, (2, 4))
    assert_shape(linear(torch.ones((5, 2, 3))).shape, (5, 2, 4))
    assert_shape(linear.weight.shape, (4, 3))
    assert linear.bias is not None
    assert_shape(linear.bias.shape, (4,))


def test_linear_rejects_invalid_input_features() -> None:
    linear = nn.Linear(3, 4)
    assert_shape(linear(torch.ones((2, 3))).shape, (2, 4))

    with assert_raises(RuntimeError):
        linear(torch.ones((2, 5)))  # E: is not assignable to parameter `input`


def test_lazy_linear_shapes() -> None:
    assert_shape(nn.LazyLinear(128)(torch.ones((4, 256))).shape, (4, 128))
    assert_shape(nn.LazyLinear(64)(torch.ones((2, 8, 512))).shape, (2, 8, 64))
    assert_shape(nn.LazyLinear(7)(torch.ones(5)).shape, (7,))


def test_lazy_linear_rejects_scalar_input() -> None:
    lazy = nn.LazyLinear(4)
    assert_shape(lazy(torch.ones(3)).shape, (4,))

    with assert_raises(IndexError):
        nn.LazyLinear(4)(torch.ones(()))  # E: is not assignable to parameter `input`


if TYPE_CHECKING:

    def check_symbolic_linear[N: IntVar, M: IntVar, B: IntVar](
        n: Int[N], m: Int[M], x: Tensor[[B, N]]
    ) -> None:
        linear = nn.Linear(n, m)
        assert_type(linear, nn.Linear[N, M])
        assert_type(linear(x), Tensor[[B, M]])

    def check_variadic_linear[Batch: IntTuple, N: IntVar, M: IntVar](
        linear: nn.Linear[N, M], x: Tensor[[*Elements[Batch], N]]
    ) -> None:
        assert_type(linear(x), Tensor[[*Elements[Batch], M]])

    def check_symbolic_lazy_linear[M: IntVar, B: IntVar, N: IntVar](
        m: Int[M], x: Tensor[[B, N]]
    ) -> None:
        linear = nn.LazyLinear(m)
        assert_type(linear, nn.LazyLinear[M])
        assert_type(linear(x), Tensor[[B, M]])
