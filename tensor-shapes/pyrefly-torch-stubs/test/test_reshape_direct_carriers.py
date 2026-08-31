# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence
from typing import TYPE_CHECKING, assert_type, reveal_type

import torch
from shape_extensions import IntTuple

if TYPE_CHECKING:
    from torch import Tensor


def test_reshape_argument_forms() -> None:
    x: Tensor[[6]] = torch.randn(6)

    assert_type(x.reshape(2, 3), Tensor[[2, 3]])
    assert_type(x.view((2, 3)), Tensor[[2, 3]])
    assert_type(torch.reshape(x, (2, 3)), Tensor[[2, 3]])

    dimensions: Sequence[int] = [2, 3]
    assert_type(x.reshape([2, 3]), Tensor)
    assert_type(x.view(dimensions), Tensor)
    assert_type(torch.reshape(x, dimensions), Tensor)


def test_scalar_and_empty_reshape() -> None:
    scalar: Tensor[[]] = torch.randn(())
    assert_type(scalar.view(()), Tensor[[]])
    assert_type(scalar.reshape(-1), Tensor[[1]])

    empty = torch.empty(0, 3)
    reveal_type(empty.reshape(0, 3))  # E: revealed type: Tensor[[0, 3]]
    reveal_type(empty.view((3, 0)))  # E: revealed type: Tensor[[3, 0]]
    # E: revealed type: Tensor[[0, 1, 3]]
    reveal_type(torch.reshape(empty, (0, 1, 3)))


def test_explicit_target_element_count() -> None:
    x: Tensor[[6]] = torch.randn(6)
    # A fully specified target is accepted when its element count matches the input;
    # the mismatching case is diagnosed in negative_tests/test_view_errors.py.
    assert_type(x.reshape(3, 2), Tensor[[3, 2]])
    assert_type(x.view(1, 6, 1), Tensor[[1, 6, 1]])


def check_open_shapes[Shape: IntTuple](
    x: Tensor[Shape], target: tuple[int, ...]
) -> None:
    # An open input has no element count to compare the target against, so the rule
    # recovers gradually instead of asserting a target it cannot validate. Variadic and
    # symbolic inputs keep their precision — see test_variadic_view.py.
    assert_type(x.reshape((2, 3)), Tensor)
    assert_type(x.view(target), Tensor)
