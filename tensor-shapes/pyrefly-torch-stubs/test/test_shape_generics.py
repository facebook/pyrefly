# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, assert_type, TYPE_CHECKING

import torch
from shape_extensions import assert_shape, IntTuple, IntVar
from torch import Tensor


def tensor_identity[Shape: IntTuple](tensor: Tensor[Shape]) -> Tensor[Shape]:
    return tensor


def tuple_identity[*Elements](values: tuple[*Elements]) -> tuple[*Elements]:
    return values


class Box[Value]:
    __slots__ = ("value",)

    def __init__(self, value: Value) -> None:
        self.value = value


class First: ...


class Second: ...


def construct[Value](class_: type[Value]) -> Value: ...


def vector_identity[Size: IntVar](tensor: Tensor[[Size]]) -> Tensor[[Size]]:
    return tensor


class IntContainer[Value]: ...


def test_generic_round_trips() -> None:
    tensor = tensor_identity(torch.randn((10, 20)))
    assert_type(tensor, Tensor[[10, 20]])
    assert_shape(tensor.shape, (10, 20))
    values = tuple_identity((Box(0), Box("value")))
    assert_type(values, tuple[Box[int], Box[str]])
    assert values[0].value == 0


if TYPE_CHECKING:
    assert_type(construct(First), First)
    assert_type(construct(Second), Second)

    def check_expression_binding[Size: IntVar](tensor: Tensor[[(2 * Size)]]) -> None:
        assert_type(vector_identity(tensor), Tensor[[(2 * Size)]])

    def check_expression_canonicalization[Size: IntVar](
        left: Tensor[[Size - 1]], right: Tensor[[-1 + Size]]
    ) -> None:
        assert_type(left, Tensor[[-1 + Size]])
        assert_type(right, Tensor[[Size - 1]])

    # Integer values are valid only for shape parameters with an integer bound.
    container: IntContainer[5] = IntContainer()  # E: Expected a type form
    assert_type(container, IntContainer[Any])
