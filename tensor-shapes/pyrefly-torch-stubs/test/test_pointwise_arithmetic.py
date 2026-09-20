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

    # TODO: BUG: Tensor operands broadcast to determine each result shape.
    for result in (torch.add(left, right), torch.pow(left, right)):
        assert_shape(result.shape, (2, 1), runtime=(2, 3))


def test_arithmetic_method_shapes() -> None:
    left = torch.ones((2, 1))
    right = torch.ones((1, 3))

    # TODO: BUG: Tensor operands broadcast to determine each result shape.
    for result in (left.add(right), left.pow(right)):
        assert_shape(result.shape, (2, 1), runtime=(2, 3))


def test_arithmetic_operator_shapes() -> None:
    left = torch.ones((2, 1))
    right = torch.ones((1, 3))

    assert_shape((left + right).shape, (2, 3))
    # TODO: BUG: Tensor exponentiation should broadcast its operands.
    assert_shape((left**right).shape, (2, 1), runtime=(2, 3))


def test_arithmetic_scalar_shapes() -> None:
    x = torch.ones((2, 3))

    assert_shape((x + 1.0).shape, (2, 3))
    # TODO: BUG: `add` and `sub` should accept scalar operands.
    # E: Argument `float` is not assignable to parameter `other`
    assert_shape(torch.add(x, 1.0).shape, (2, 3))
    # E: Argument `float` is not assignable to parameter `other`
    assert_shape(torch.sub(x, 1.0).shape, (2, 3))
    # E: Argument `float` is not assignable to parameter `other`
    assert_shape(x.add(1.0).shape, (2, 3))
    # E: Argument `float` is not assignable to parameter `other`
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

    # TODO: BUG: Functions and methods should reject incompatible shapes statically.
    with assert_raises(RuntimeError):
        torch.add(left, right)


if TYPE_CHECKING:

    def check_symbolic_arithmetic[N: IntVar, M: IntVar](
        left: Tensor[[N, 1]], right: Tensor[[1, M]]
    ) -> None:
        assert_type(left + right, Tensor[[N, M]])
        assert_type(left - right, Tensor[[N, M]])
        assert_type(left * right, Tensor[[N, M]])
        assert_type(left / right, Tensor[[N, M]])
        # TODO: BUG: Tensor exponentiation should preserve broadcast symbols.
        assert_type(left**right, Tensor[[N, 1]])
        # TODO: BUG: Functions and methods should preserve broadcast symbols.
        assert_type(torch.add(left, right), Tensor[[N, 1]])
        assert_type(torch.sub(left, right), Tensor[[N, 1]])
        assert_type(torch.mul(left, right), Tensor[[N, 1]])
        assert_type(torch.div(left, right), Tensor[[N, 1]])
        assert_type(torch.pow(left, right), Tensor[[N, 1]])
        assert_type(left.add(right), Tensor[[N, 1]])
        assert_type(left.sub(right), Tensor[[N, 1]])
        assert_type(left.mul(right), Tensor[[N, 1]])
        assert_type(left.div(right), Tensor[[N, 1]])
        assert_type(left.pow(right), Tensor[[N, 1]])
