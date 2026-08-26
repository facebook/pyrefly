# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type, TYPE_CHECKING

import torch
from shape_extensions import assert_raises, assert_shape, IntVar
from torch import Tensor


def test_bitwise_function_shapes() -> None:
    left = torch.ones((2, 1), dtype=torch.int64)
    right = torch.ones((1, 3), dtype=torch.int64)

    assert_shape(torch.bitwise_and(left, right).shape, (2, 3))
    assert_shape(torch.bitwise_not(left).shape, (2, 1))


def test_bitwise_method_shapes() -> None:
    left = torch.ones((2, 1), dtype=torch.int64)
    right = torch.ones((1, 3), dtype=torch.int64)

    assert_shape(left.bitwise_or(right).shape, (2, 3))
    assert_shape(left.bitwise_not().shape, (2, 1))


def test_bitwise_operator_shapes() -> None:
    left = torch.ones((2, 1), dtype=torch.int64)
    right = torch.ones((1, 3), dtype=torch.int64)

    assert_shape((left & right).shape, (2, 3))
    assert_shape((left | right).shape, (2, 3))
    assert_shape((left ^ right).shape, (2, 3))
    assert_shape((~left).shape, (2, 1))


def test_bitwise_scalar_shapes() -> None:
    x = torch.ones((2, 3), dtype=torch.int64)

    assert_shape((x & 1).shape, (2, 3))
    assert_shape((1 | x).shape, (2, 3))
    assert_shape(torch.bitwise_and(x, 1).shape, (2, 3))
    assert_shape(torch.bitwise_or(x, 1).shape, (2, 3))
    assert_shape(x.bitwise_xor(1).shape, (2, 3))


def test_bitwise_rejects_incompatible_shapes() -> None:
    left = torch.ones((2, 3), dtype=torch.int64)
    right = torch.ones((4, 5), dtype=torch.int64)
    assert_shape((left & torch.ones((2, 3), dtype=torch.int64)).shape, (2, 3))

    with assert_raises(RuntimeError):
        # E: Cannot broadcast dimension
        _ = left & right

    with assert_raises(RuntimeError):
        # E: Cannot broadcast dimension
        torch.bitwise_and(left, right)
    with assert_raises(RuntimeError):
        # E: Cannot broadcast dimension
        left.bitwise_and(right)


def test_bitwise_shift_shapes() -> None:
    left = torch.ones((2, 1), dtype=torch.int64)
    right = torch.ones((1, 3), dtype=torch.int64)

    assert_shape(torch.bitwise_left_shift(left, right).shape, (2, 3))
    assert_shape(left.bitwise_right_shift(right).shape, (2, 3))


def test_bitwise_shift_scalar_shapes() -> None:
    x = torch.ones((2, 3), dtype=torch.int64)

    assert_shape(torch.bitwise_left_shift(x, 1).shape, (2, 3))
    assert_shape(x.bitwise_right_shift(1).shape, (2, 3))


if TYPE_CHECKING:

    def check_symbolic_bitwise[N: IntVar, M: IntVar](
        left: Tensor[[N, 1]], right: Tensor[[1, M]]
    ) -> None:
        assert_type(left & right, Tensor[[N, M]])
        assert_type(left | right, Tensor[[N, M]])
        assert_type(left ^ right, Tensor[[N, M]])

        assert_type(torch.bitwise_and(left, right), Tensor[[N, M]])
        assert_type(torch.bitwise_or(left, right), Tensor[[N, M]])
        assert_type(torch.bitwise_xor(left, right), Tensor[[N, M]])
        assert_type(left.bitwise_and(right), Tensor[[N, M]])
        assert_type(left.bitwise_or(right), Tensor[[N, M]])
        assert_type(left.bitwise_xor(right), Tensor[[N, M]])

        assert_type(torch.bitwise_left_shift(left, right), Tensor[[N, M]])
        assert_type(torch.bitwise_right_shift(left, right), Tensor[[N, M]])
        assert_type(left.bitwise_left_shift(right), Tensor[[N, M]])
        assert_type(left.bitwise_right_shift(right), Tensor[[N, M]])

    def check_incompatible_bitwise_shift_shapes(
        left: Tensor[[2, 3]], right: Tensor[[4, 5]]
    ) -> None:
        # E: Cannot broadcast dimension
        torch.bitwise_left_shift(left, right)
        # E: Cannot broadcast dimension
        left.bitwise_right_shift(right)
