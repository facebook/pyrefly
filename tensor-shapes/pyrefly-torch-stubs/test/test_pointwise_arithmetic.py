# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type, TYPE_CHECKING

import torch
from shape_extensions import assert_raises, assert_shape, IntVar
from torch import Tensor


def test_arithmetic_function_shapes() -> None:
    left = torch.ones((2, 1))
    right = torch.ones((1, 3))

    assert_shape(torch.add(left, right).shape, (2, 3))
    assert_shape(torch.pow(left, right).shape, (2, 3))


def test_arithmetic_method_shapes() -> None:
    left = torch.ones((2, 1))
    right = torch.ones((1, 3))

    assert_shape(left.add(right).shape, (2, 3))
    assert_shape(left.pow(right).shape, (2, 3))
    # TODO: BUG: Broadcast the other operand for min/max methods.
    assert_shape(left.maximum(right).shape, (2, 1), runtime=(2, 3))
    assert_shape(left.minimum(right).shape, (2, 1), runtime=(2, 3))
    assert_shape(left.fmax(right).shape, (2, 1), runtime=(2, 3))
    assert_shape(left.fmin(right).shape, (2, 1), runtime=(2, 3))


def test_arithmetic_operator_shapes() -> None:
    left = torch.ones((2, 1))
    right = torch.ones((1, 3))

    assert_shape((left + right).shape, (2, 3))
    assert_shape((left**right).shape, (2, 3))


def test_arithmetic_scalar_shapes() -> None:
    x = torch.ones((2, 3))

    assert_shape((x + 1.0).shape, (2, 3))
    assert_shape(torch.add(x, 1.0).shape, (2, 3))
    assert_shape(torch.sub(x, 1.0).shape, (2, 3))
    assert_shape(x.add(1.0).shape, (2, 3))
    assert_shape(x.sub(1.0).shape, (2, 3))
    assert_shape(torch.mul(x, 2.0).shape, (2, 3))
    assert_shape(x.div(2.0).shape, (2, 3))


def test_arithmetic_rejects_incompatible_shapes() -> None:
    left = torch.ones((2, 3))
    right = torch.ones((4, 5))
    assert_shape((left + torch.ones((2, 3))).shape, (2, 3))

    with assert_raises(RuntimeError):
        # E: Cannot broadcast dimension
        left + right

    with assert_raises(RuntimeError):
        # E: Cannot broadcast dimension
        torch.add(left, right)
    with assert_raises(RuntimeError):
        # E: Cannot broadcast dimension
        left.add(right)


if TYPE_CHECKING:

    def check_symbolic_arithmetic[N: IntVar, M: IntVar](
        left: Tensor[[N, 1]], right: Tensor[[1, M]]
    ) -> None:
        assert_type(left + right, Tensor[[N, M]])
        assert_type(left - right, Tensor[[N, M]])
        assert_type(left * right, Tensor[[N, M]])
        assert_type(left / right, Tensor[[N, M]])
        assert_type(left**right, Tensor[[N, M]])
        assert_type(torch.add(left, right), Tensor[[N, M]])
        assert_type(torch.sub(left, right), Tensor[[N, M]])
        assert_type(torch.mul(left, right), Tensor[[N, M]])
        assert_type(torch.div(left, right), Tensor[[N, M]])
        assert_type(torch.pow(left, right), Tensor[[N, M]])
        assert_type(left.add(right), Tensor[[N, M]])
        assert_type(left.sub(right), Tensor[[N, M]])
        assert_type(left.mul(right), Tensor[[N, M]])
        assert_type(left.div(right), Tensor[[N, M]])
        assert_type(left.pow(right), Tensor[[N, M]])

    def check_scalar_operators[N: IntVar, M: IntVar](tensor: Tensor[[N, M]]) -> None:
        assert_type(tensor + 1, Tensor[[N, M]])
        assert_type(tensor - 1.0, Tensor[[N, M]])
        assert_type(tensor * 1j, Tensor[[N, M]])
        assert_type(tensor % 2, Tensor[[N, M]])
        assert_type(tensor / 2, Tensor[[N, M]])
        assert_type(tensor // 2, Tensor[[N, M]])

    def check_incompatible_power_shapes(
        left: Tensor[[2, 3]], right: Tensor[[4, 5]]
    ) -> None:
        # E: Cannot broadcast dimension
        left**right
        # E: Cannot broadcast dimension
        torch.pow(left, right)
        # E: Cannot broadcast dimension
        left.pow(right)
