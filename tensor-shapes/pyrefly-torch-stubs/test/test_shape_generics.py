# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, assert_type, TYPE_CHECKING

import torch
from shape_extensions import assert_shape, Int, IntTuple, IntVar
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


def int_identity[Value: IntVar](value: Int[Value]) -> Int[Value]:
    return value


def half_int[Value: IntVar](value: Int[Value // 2]) -> Int[Value]:
    return value * 2  # type: ignore


def paired_int[Value: IntVar](value: Int[Value], half: Int[Value // 2]) -> Int[Value]:
    return value


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

    def check_concrete_shape_binding() -> None:
        tensor = tensor_identity(torch.randn((2, 3)))
        assert_type(tensor, Tensor[[2, 3]])
        _wrong: Tensor[[4, 3]] = tensor  # E: is not assignable
        _ = _wrong

    def check_int_expression_binding[Left: IntVar, Right: IntVar](
        left: Int[Left], right: Int[Right]
    ) -> None:
        product = left * right
        assert_type(int_identity(product), Int[Left * Right])
        # A type variable cannot be inferred from within an arithmetic expression.
        half_int(product // 2)  # E: Type variable cannot be inferred
        assert_type(paired_int(product, product // 2), Int[Left * Right])

    def check_int_literal_and_gradual_binding(dynamic: int) -> None:
        implicit: Int = 4
        explicit: Int[Any] = 4
        assert_type(int_identity(4), Int[4])
        assert_type(int_identity(dynamic), Int)
        assert_type(int_identity(implicit), Int)
        assert_type(int_identity(explicit), Int[Any])

    def bad_numel_return[N: IntVar, M: IntVar, Result: IntVar](
        tensor: Tensor[[N, M]],
    ) -> Int[Result]:
        # E: Returned type `Int[(N * M)]` is not assignable
        return tensor.numel()

    def bad_view_return[N: IntVar, M: IntVar, Result: IntVar](
        tensor: Tensor[[N, M]],
    ) -> Tensor[[Result]]:
        # E: Returned type `Tensor[[(N * M)]]` is not assignable
        return tensor.view(-1)

    def check_invalid_return_inference() -> None:
        # A return-only dimension has no argument-based constraint, so it is gradual.
        assert_type(bad_numel_return(torch.randn((3, 4))), Int)
        assert_type(bad_view_return(torch.randn((3, 4))), Tensor[[Any]])

    def check_expression_canonicalization[Size: IntVar](
        left: Tensor[[Size - 1]], right: Tensor[[-1 + Size]]
    ) -> None:
        assert_type(left, Tensor[[-1 + Size]])
        assert_type(right, Tensor[[Size - 1]])

    # Integer values are valid only for shape parameters with an integer bound.
    container: IntContainer[5] = IntContainer()  # E: Expected a type form
    assert_type(container, IntContainer[Any])
