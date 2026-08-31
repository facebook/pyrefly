# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Phase 1.1: Missing shape operations tests
from typing import Any, Literal, assert_type, reveal_type

import torch
from shape_extensions import Int, IntTuple, IntVar
from torch import Tensor


# Test: torch.unbind (removes a dimension)
def test_unbind_dim0():
    x: Tensor[[3, 4]] = torch.randn(3, 4)
    # unbind along dim 0 removes first dimension
    # Returns tuple of 3 tensors, each of shape [4]
    # Note: Type checking tuple elements is limited, so we just verify the call works
    _ = torch.unbind(x, dim=0)


def test_unbind_dim1():
    x: Tensor[[3, 4]] = torch.randn(3, 4)
    # unbind along dim 1 removes second dimension
    _ = torch.unbind(x, dim=1)


def test_unbind_method():
    x: Tensor[[2, 3, 4]] = torch.randn(2, 3, 4)
    _ = x.unbind(dim=1)


# Test: torch.movedim (moves dimensions to new positions)
def test_movedim_single():
    x: Tensor[[2, 3, 4]] = torch.randn(2, 3, 4)
    # Move dimension 0 to position 2: [2, 3, 4] -> [3, 4, 2]
    result = torch.movedim(x, source=0, destination=2)
    assert_type(result, Tensor[[3, 4, 2]])


def test_movedim_multiple():
    x: Tensor[[2, 3, 4, 5]] = torch.randn(2, 3, 4, 5)
    assert_type(
        torch.movedim(x, source=(0, 1), destination=(2, 3)),
        Tensor[[4, 5, 2, 3]],
    )


def test_movedim_negative():
    x: Tensor[[2, 3, 4]] = torch.randn(2, 3, 4)
    # Move last dimension to first: [2, 3, 4] -> [4, 2, 3]
    result = torch.movedim(x, source=-1, destination=0)
    assert_type(result, Tensor[[4, 2, 3]])


def test_movedim_method():
    x: Tensor[[2, 3, 4]] = torch.randn(2, 3, 4)
    result = x.movedim(source=1, destination=0)
    assert_type(result, Tensor[[3, 2, 4]])


# Test: torch.moveaxis (alias for movedim)
def test_moveaxis():
    x: Tensor[[2, 3, 4]] = torch.randn(2, 3, 4)
    result = torch.moveaxis(x, source=0, destination=2)
    assert_type(result, Tensor[[3, 4, 2]])


def test_movedim_method_function_alias_parity():
    x: Tensor[[2, 3, 4, 5]] = torch.randn(2, 3, 4, 5)
    assert_type(torch.movedim(x, (0, 2), (2, 0)), Tensor[[4, 3, 2, 5]])
    assert_type(torch.moveaxis(x, (0, 2), (2, 0)), Tensor[[4, 3, 2, 5]])
    assert_type(
        torch.movedim(input=x, source=(0, 2), destination=(2, 0)),
        Tensor[[4, 3, 2, 5]],
    )
    assert_type(
        torch.moveaxis(input=x, source=(0, 2), destination=(2, 0)),
        Tensor[[4, 3, 2, 5]],
    )
    assert_type(x.movedim((0, 2), (2, 0)), Tensor[[4, 3, 2, 5]])
    assert_type(x.moveaxis((0, 2), (2, 0)), Tensor[[4, 3, 2, 5]])
    assert_type(x.movedim(source=(0, 2), destination=(2, 0)), Tensor[[4, 3, 2, 5]])


def test_movedim_negative_tuple_axes():
    x: Tensor[[2, 3, 4, 5]] = torch.randn(2, 3, 4, 5)
    assert_type(torch.movedim(x, (-1, -3), (0, 2)), Tensor[[5, 2, 3, 4]])


def test_movedim_empty_tuple_identity():
    scalar: Tensor[[]] = torch.tensor(1)
    x: Tensor[[2, 3, 4]] = torch.randn(2, 3, 4)
    assert_type(torch.movedim(scalar, (), ()), Tensor[[]])
    assert_type(x.moveaxis((), ()), Tensor[[2, 3, 4]])


def test_movedim_full_axis_permutations():
    x: Tensor[[2, 3, 4]] = torch.randn(2, 3, 4)
    assert_type(torch.movedim(x, (0, 1, 2), (0, 1, 2)), Tensor[[2, 3, 4]])
    assert_type(torch.movedim(x, (0, 1, 2), (0, 2, 1)), Tensor[[2, 4, 3]])
    assert_type(torch.movedim(x, (0, 1, 2), (1, 0, 2)), Tensor[[3, 2, 4]])
    assert_type(torch.movedim(x, (0, 1, 2), (1, 2, 0)), Tensor[[4, 2, 3]])
    assert_type(torch.movedim(x, (0, 1, 2), (2, 0, 1)), Tensor[[3, 4, 2]])
    assert_type(torch.movedim(x, (0, 1, 2), (2, 1, 0)), Tensor[[4, 3, 2]])


def test_movedim_tuple_symbolic_shape[A: IntVar, B: IntVar, C: IntVar, D: IntVar](
    x: Tensor[[A, B, C, D]],
) -> None:
    assert_type(torch.moveaxis(x, (0, 2), (2, 0)), Tensor[[C, B, A, D]])
    assert_type(x.movedim((-1, -3), (0, 2)), Tensor[[D, A, B, C]])


def test_movedim_scalar_rank0():
    scalar: Tensor[[]] = torch.tensor(1)
    # A rank-0 tensor has one implicit axis spelled either 0 or -1, so every
    # combination of those spellings is a legal no-op.
    assert_type(torch.movedim(scalar, 0, 0), Tensor[[]])
    assert_type(torch.movedim(scalar, -1, -1), Tensor[[]])
    assert_type(torch.movedim(scalar, 0, -1), Tensor[[]])
    assert_type(torch.moveaxis(scalar, -1, 0), Tensor[[]])
    assert_type(scalar.movedim(0, -1), Tensor[[]])
    assert_type(scalar.moveaxis(-1, -1), Tensor[[]])


def test_movedim_tuple_rank0():
    scalar: Tensor[[]] = torch.tensor(1)
    # The tuple overloads name the same implicit axis as the scalar ones, so
    # every one-axis move is a legal no-op.
    assert_type(torch.movedim(scalar, (0,), (0,)), Tensor[[]])
    assert_type(torch.movedim(scalar, (-1,), (-1,)), Tensor[[]])
    assert_type(torch.moveaxis(scalar, (0,), (-1,)), Tensor[[]])
    assert_type(scalar.movedim((-1,), (0,)), Tensor[[]])
    assert_type(scalar.moveaxis((0,), (0,)), Tensor[[]])


def movedim_symbolic[A: IntVar, B: IntVar, C: IntVar](
    x: Tensor[[A, B, C]],
) -> Tensor[[B, C, A]]:
    return torch.moveaxis(x, 0, 2)


def test_movedim_gradual_and_broad_arguments[
    S0: IntVar,
    S1: IntVar,
    D0: IntVar,
    D1: IntVar,
](
    scalar_source: int,
    scalar_destination: int,
    fixed_source: tuple[int, int],
    fixed_destination: tuple[int, int],
    symbolic_source: tuple[Int[S0], Int[S1]],
    symbolic_destination: tuple[Int[D0], Int[D1]],
    mixed_source: tuple[Literal[0], Int[S0]],
    mixed_destination: tuple[Int[D0], Literal[2]],
    tuple_source: tuple[int, ...],
    tuple_destination: tuple[int, ...],
    dynamic: Any,
):
    x: Tensor[[2, 3, 4]] = torch.randn(2, 3, 4)
    scalar: Tensor[[]] = torch.tensor(1)
    gradual: Tensor[IntTuple] = x
    assert_type(torch.movedim(gradual, 0, 2), Tensor[IntTuple])
    assert_type(torch.movedim(x, scalar_source, scalar_destination), Tensor[IntTuple])
    assert_type(torch.moveaxis(x, fixed_source, fixed_destination), Tensor[IntTuple])
    assert_type(
        torch.movedim(x, symbolic_source, symbolic_destination), Tensor[IntTuple]
    )
    assert_type(x.moveaxis(mixed_source, mixed_destination), Tensor[IntTuple])
    assert_type(torch.moveaxis(x, tuple_source, tuple_destination), Tensor[IntTuple])
    assert_type(x.movedim(dynamic, dynamic), Tensor[IntTuple])
    # A rank-0 tensor with a non-literal axis degrades instead of guessing, and
    # must not reach the permutation arithmetic that divides by the rank.
    assert_type(
        torch.movedim(scalar, (scalar_source,), (scalar_destination,)),
        Tensor[IntTuple],
    )


# Test: torch.unfold (sliding window view)
def test_unfold_basic():
    x: Tensor[[8]] = torch.randn(8)
    # unfold with size=3, step=1: (8-3)/1 + 1 = 6 windows of size 3
    # Output shape: [6, 3]
    result = torch.unfold(x, dimension=0, size=3, step=1)
    assert_type(result, Tensor[[6, 3]])


def test_unfold_2d():
    x: Tensor[[4, 6]] = torch.randn(4, 6)
    # unfold dimension 1 with size=2, step=2: (6-2)/2 + 1 = 3 windows
    # Output shape: [4, 3, 2]
    result = torch.unfold(x, dimension=1, size=2, step=2)
    assert_type(result, Tensor[[4, 3, 2]])


def test_unfold_method():
    x: Tensor[[10]] = torch.randn(10)
    # unfold with size=4, step=2: (10-4)/2 + 1 = 4 windows
    result = x.unfold(dimension=0, size=4, step=2)
    assert_type(result, Tensor[[4, 4]])


def test_unfold_3d():
    x: Tensor[[2, 5, 8]] = torch.randn(2, 5, 8)
    # unfold dimension 2 with size=3, step=1: (8-3)/1 + 1 = 6 windows
    # Output shape: [2, 5, 6, 3]
    result = x.unfold(dimension=2, size=3, step=1)
    assert_type(result, Tensor[[2, 5, 6, 3]])


def test_unfold_negative_dimension():
    x: Tensor[[8, 5]] = torch.randn(8, 5)
    result = torch.unfold(x, dimension=-2, size=3, step=2)
    assert_type(result, Tensor[[3, 5, 3]])


def test_unfold_zero_size():
    x: Tensor[[5]] = torch.randn(5)
    reveal_type(
        torch.unfold(x, dimension=0, size=0, step=2)
    )  # revealed type: Tensor[[3, 0]]
    empty = torch.randn(0)
    reveal_type(
        empty.unfold(dimension=0, size=0, step=2)
    )  # revealed type: Tensor[[1, 0]]


def test_unfold_scalar():
    scalar: Tensor[[]] = torch.tensor(1)
    reveal_type(torch.unfold(scalar, 0, 0, 2))  # revealed type: Tensor[[0]]
    reveal_type(scalar.unfold(-1, 0, 2))  # revealed type: Tensor[[0]]
    assert_type(torch.unfold(scalar, 0, 1, 2), Tensor[[1]])
    assert_type(scalar.unfold(-1, 1, 2), Tensor[[1]])


def test_unfold_shapeless(x: Tensor):
    assert_type(x.unfold(dimension=0, size=3, step=1), Tensor)
