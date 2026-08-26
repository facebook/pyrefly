# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, assert_type, TYPE_CHECKING

import torch
from shape_extensions import assert_raises, assert_shape, IntVar
from torch import Tensor


def test_comparison_function_shapes() -> None:
    left = torch.ones((2, 1))
    right = torch.ones((1, 3))

    assert_shape(torch.eq(left, right).shape, (2, 3))
    assert_shape(torch.lt(left, right).shape, (2, 3))


def test_comparison_method_shapes() -> None:
    left = torch.ones((2, 1))
    right = torch.ones((1, 3))

    assert_shape(left.eq(right).shape, (2, 3))
    assert_shape(left.lt(right).shape, (2, 3))


def test_equality_operator_shapes() -> None:
    left = torch.ones((2, 1))
    right = torch.ones((1, 3))

    assert_shape((left == right).shape, (2, 3))
    assert_shape((left != right).shape, (2, 3))


def test_comparison_scalar_shapes() -> None:
    x = torch.ones((2, 3))

    assert_shape(torch.eq(x, 0).shape, (2, 3))
    assert_shape(torch.lt(x, 0).shape, (2, 3))
    assert_shape(x.eq(0).shape, (2, 3))
    assert_shape(x.lt(0).shape, (2, 3))


def test_comparison_rejects_incompatible_shapes() -> None:
    left = torch.ones((2, 3))
    right = torch.ones((4, 5))
    assert_shape((left == torch.ones((2, 3))).shape, (2, 3))

    with assert_raises(RuntimeError):
        # E: Cannot broadcast dimension
        _ = left == right

    with assert_raises(RuntimeError):
        # E: Cannot broadcast dimension
        torch.eq(left, right)
    with assert_raises(RuntimeError):
        # E: Cannot broadcast dimension
        left.eq(right)


if TYPE_CHECKING:

    def check_symbolic_comparisons[N: IntVar, M: IntVar](
        left: Tensor[[N, 1]], right: Tensor[[1, M]]
    ) -> None:
        assert_type(left == right, Tensor[[N, M]])
        assert_type(left != right, Tensor[[N, M]])

        # Ordering operators remain gradual because scalar tensors also participate
        # in Python's truth-valued comparison protocol.
        assert_type(left < right, Any)
        assert_type(left <= right, Any)
        assert_type(left > right, Any)
        assert_type(left >= right, Any)

        assert_type(torch.eq(left, right), Tensor[[N, M]])
        assert_type(torch.ne(left, right), Tensor[[N, M]])
        assert_type(torch.lt(left, right), Tensor[[N, M]])
        assert_type(torch.le(left, right), Tensor[[N, M]])
        assert_type(torch.gt(left, right), Tensor[[N, M]])
        assert_type(torch.ge(left, right), Tensor[[N, M]])
        assert_type(left.eq(right), Tensor[[N, M]])
        assert_type(left.ne(right), Tensor[[N, M]])
        assert_type(left.lt(right), Tensor[[N, M]])
        assert_type(left.le(right), Tensor[[N, M]])
        assert_type(left.gt(right), Tensor[[N, M]])
        assert_type(left.ge(right), Tensor[[N, M]])
