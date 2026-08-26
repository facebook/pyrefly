# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, assert_type, Callable, overload, TYPE_CHECKING

import torch
from shape_extensions import assert_shape, Elements, Int, IntTuple, IntVar
from torch import Tensor


def tensor_identity[Shape: IntTuple](tensor: Tensor[Shape]) -> Tensor[Shape]:
    return tensor


def tuple_identity[*Elements](values: tuple[*Elements]) -> tuple[*Elements]:
    return values


def forward_arguments[*Arguments](
    callback: Callable[[*Arguments], object], *arguments: *Arguments
) -> None:
    callback(*arguments)


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


if TYPE_CHECKING:  # noqa: C901

    @overload
    def depth_result(depth: Int[1]) -> Tensor[[32]]: ...

    @overload
    def depth_result[Depth: IntVar](depth: Int[Depth]) -> Tensor[[32 * Depth]]: ...

    def depth_result[Depth: IntVar](
        depth: Int[Depth],
    ) -> Tensor[[32]] | Tensor[[32 * Depth]]: ...

    def sum_dimensions[N: IntVar, M: IntVar](
        tensor: Tensor[[N, M]],
    ) -> Tensor[[N + M]]: ...

    def product_dimensions[N: IntVar, M: IntVar](
        tensor: Tensor[[N, M]],
    ) -> Tensor[[N * M]]: ...

    def duplicate_dimension[N: IntVar](tensor: Tensor[[N]]) -> Tensor[[2 * N]]: ...

    def prefix_and_suffix[
        Prefix: IntVar,
        Middle: IntTuple,
        Penultimate: IntVar,
        Last: IntVar,
    ](
        tensor: Tensor[[Prefix, *Elements[Middle], Penultimate, Last]],
    ) -> Tensor[[Prefix, *Elements[Middle], Penultimate, Last]]:
        return tensor

    def split_first[FirstDim: IntVar, Rest: IntTuple](
        tensor: Tensor[[FirstDim, *Elements[Rest]]],
    ) -> tuple[Tensor[[FirstDim]], Tensor[Rest]]: ...

    def split_last[Initial: IntTuple, LastDim: IntVar](
        tensor: Tensor[[*Elements[Initial], LastDim]],
    ) -> tuple[Tensor[Initial], Tensor[[LastDim]]]: ...

    assert_type(construct(First), First)
    assert_type(construct(Second), Second)
    assert_type(depth_result(1), Tensor[[32]])
    assert_type(depth_result(6), Tensor[[192]])

    def check_expression_binding[Size: IntVar](tensor: Tensor[[(2 * Size)]]) -> None:
        assert_type(vector_identity(tensor), Tensor[[(2 * Size)]])

    def check_concrete_shape_binding() -> None:
        tensor = tensor_identity(torch.randn((2, 3)))
        assert_type(tensor, Tensor[[2, 3]])
        _wrong: Tensor[[4, 3]] = tensor  # E: is not assignable
        _wrong_rank: Tensor[[2, 3, 4]] = tensor  # E: is not assignable
        _ = _wrong
        _ = _wrong_rank

    def check_symbolic_shape_subtyping[N: IntVar, M: IntVar](
        tensor: Tensor[[N, M]], gradual: Tensor[IntTuple]
    ) -> None:
        assert_type(tensor_identity(tensor), Tensor[[N, M]])
        _swapped: Tensor[[M, N]] = tensor  # E: is not assignable
        _gradual: Tensor[IntTuple] = tensor
        _concrete: Tensor[[2, 3]] = gradual
        _ = (_swapped, _gradual, _concrete)

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

    def check_expression_equivalence[Left: IntVar, Right: IntVar](
        addition: Tensor[[Left + Right]],
        multiplication: Tensor[[Left * Right]],
        concrete: Tensor[[2 + 3, 4]],
        power: Tensor[[8 * 2**Left]],
        offset: Tensor[[Left + 1]],
    ) -> None:
        assert_type(addition, Tensor[[Right + Left]])
        assert_type(multiplication, Tensor[[Right * Left]])
        assert_type(concrete, Tensor[[5, 4]])
        assert_type(power, Tensor[[2 ** (Left + 3)]])
        _wrong: Tensor[[Left * Right]] = addition  # E: is not assignable
        _different_offset: Tensor[[Left + 2]] = offset  # E: is not assignable
        _ = _wrong
        _ = _different_offset

    def check_generic_expression_substitution[N: IntVar, M: IntVar](
        concrete: Tensor[[2, 3]], symbolic: Tensor[[N, M]]
    ) -> None:
        assert_type(sum_dimensions(concrete), Tensor[[5]])
        assert_type(product_dimensions(concrete), Tensor[[6]])
        assert_type(sum_dimensions(symbolic), Tensor[[N + M]])
        assert_type(product_dimensions(symbolic), Tensor[[N * M]])
        assert_type(
            duplicate_dimension(product_dimensions(symbolic)),
            Tensor[[2 * N * M]],
        )

    def check_variadic_shape_binding(
        tensor: Tensor[[1, 2, 3, 4, 5, 6]],
    ) -> None:
        assert_type(prefix_and_suffix(tensor), Tensor[[1, 2, 3, 4, 5, 6]])
        assert_type(
            split_first(tensor),
            tuple[Tensor[[1]], Tensor[[2, 3, 4, 5, 6]]],
        )
        assert_type(
            split_last(tensor),
            tuple[Tensor[[1, 2, 3, 4, 5]], Tensor[[6]]],
        )

    def check_symbolic_variadic_shape_binding[
        A: IntVar,
        B: IntVar,
        Middle: IntTuple,
        D: IntVar,
        E: IntVar,
        F: IntVar,
    ](tensor: Tensor[[A, B, *Elements[Middle], D, E, F]]) -> None:
        assert_type(
            prefix_and_suffix(tensor),
            Tensor[[A, B, *Elements[Middle], D, E, F]],
        )

    def check_variadic_argument_forwarding[*Arguments](
        callback: Callable[[*Arguments], object], arguments: tuple[*Arguments]
    ) -> None:
        callback(*arguments)
        forward_arguments(callback, *arguments)

    # Integer values are valid only for shape parameters with an integer bound.
    container: IntContainer[5] = IntContainer()  # E: Expected a type form
    assert_type(container, IntContainer[Any])
