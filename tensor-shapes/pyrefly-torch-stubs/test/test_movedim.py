# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, assert_type, Literal, TYPE_CHECKING

import torch
from shape_extensions import assert_raises, assert_shape, Int, IntTuple, IntVar
from torch import Tensor


def test_movedim_shapes() -> None:
    tensor = torch.ones((2, 3, 4))
    assert_shape(torch.movedim(tensor, 0, 2).shape, (3, 4, 2))
    assert_shape(torch.moveaxis(tensor, -1, 0).shape, (4, 2, 3))
    assert_shape(tensor.movedim(1, 0).shape, (3, 2, 4))
    assert_shape(tensor.moveaxis(0, 2).shape, (3, 4, 2))

    rank_four = torch.ones((2, 3, 4, 5))
    assert_shape(torch.movedim(rank_four, (0, 2), (2, 0)).shape, (4, 3, 2, 5))
    assert_shape(rank_four.moveaxis((-1, -3), (0, 2)).shape, (5, 2, 3, 4))


def test_movedim_permutations() -> None:
    tensor = torch.ones((2, 3, 4))
    assert_shape(torch.movedim(tensor, (0, 1, 2), (0, 1, 2)).shape, (2, 3, 4))
    assert_shape(torch.movedim(tensor, (0, 1, 2), (0, 2, 1)).shape, (2, 4, 3))
    assert_shape(torch.movedim(tensor, (0, 1, 2), (1, 0, 2)).shape, (3, 2, 4))
    assert_shape(torch.movedim(tensor, (0, 1, 2), (1, 2, 0)).shape, (4, 2, 3))
    assert_shape(torch.movedim(tensor, (0, 1, 2), (2, 0, 1)).shape, (3, 4, 2))
    assert_shape(torch.movedim(tensor, (0, 1, 2), (2, 1, 0)).shape, (4, 3, 2))


def test_movedim_scalar_and_empty_axes() -> None:
    scalar = torch.tensor(1)
    assert_shape(torch.movedim(scalar, 0, -1).shape, ())
    assert_shape(torch.moveaxis(scalar, -1, 0).shape, ())
    assert_shape(scalar.movedim((0,), (-1,)).shape, ())
    assert_shape(torch.movedim(scalar, (), ()).shape, ())

    tensor = torch.ones((2, 3, 4))
    assert_shape(tensor.moveaxis((), ()).shape, (2, 3, 4))


def test_movedim_rejects_invalid_axes() -> None:
    tensor = torch.ones((2, 3, 4))
    assert_shape(torch.movedim(tensor, 0, 2).shape, (3, 4, 2))

    with assert_raises(IndexError):
        torch.moveaxis(tensor, 3, 0)  # E: source dimension out of range

    with assert_raises(IndexError):
        tensor.movedim(0, -4)  # E: destination dimension out of range

    with assert_raises(RuntimeError):
        # E: source and destination must have equal length
        torch.movedim(tensor, (0,), (1, 2))

    with assert_raises(RuntimeError):
        tensor.moveaxis((0, 0), (1, 2))  # E: source dimensions must be unique

    with assert_raises(RuntimeError):
        # E: source dimensions must be unique
        torch.moveaxis(tensor, (0, -3), (1, 2))

    with assert_raises(RuntimeError):
        # E: destination dimensions must be unique
        torch.movedim(tensor, (0, 1), (2, -1))

    with assert_raises(IndexError):
        torch.moveaxis(tensor, (3,), (0,))  # E: source dimension out of range

    with assert_raises(IndexError):
        tensor.movedim((0,), (-4,))  # E: destination dimension out of range


def test_movedim_rejects_mixed_axis_forms() -> None:
    tensor = torch.ones((2, 3, 4))
    assert_shape(torch.movedim(tensor, 0, 1).shape, (3, 2, 4))

    with assert_raises(TypeError):
        torch.moveaxis(tensor, 0, (1,))  # E: No matching overload

    with assert_raises(TypeError):
        torch.movedim(tensor, (0,), 1)  # E: No matching overload

    with assert_raises(TypeError):
        tensor.moveaxis(0, (1,))  # E: No matching overload


def test_movedim_rejects_invalid_scalar_axes() -> None:
    scalar = torch.tensor(1)
    assert_shape(torch.movedim(scalar, 0, -1).shape, ())

    with assert_raises(IndexError):
        torch.moveaxis(scalar, 1, 0)  # E: source dimension out of range

    with assert_raises(IndexError):
        scalar.movedim(0, -2)  # E: destination dimension out of range

    with assert_raises(IndexError):
        torch.movedim(scalar, 2, 2)  # E: source dimension out of range

    with assert_raises(RuntimeError):
        # E: source and destination must have equal length
        torch.movedim(scalar, (0,), (1, 0))

    with assert_raises(IndexError):
        scalar.movedim((1,), (0,))  # E: source dimension out of range

    with assert_raises(IndexError):
        torch.moveaxis(scalar, (0,), (-2,))  # E: destination dimension out of range

    with assert_raises(RuntimeError):
        scalar.moveaxis((0, -1), (0, -1))  # E: source dimensions must be unique


if TYPE_CHECKING:

    def check_symbolic_shape[A: IntVar, B: IntVar, C: IntVar, D: IntVar](
        x: Tensor[[A, B, C, D]],
    ) -> None:
        assert_type(torch.moveaxis(x, (0, 2), (2, 0)), Tensor[[C, B, A, D]])
        assert_type(x.movedim((-1, -3), (0, 2)), Tensor[[D, A, B, C]])

    def check_gradual_axes[S0: IntVar, S1: IntVar, D0: IntVar, D1: IntVar](
        x: Tensor[[2, 3, 4]],
        scalar_source: int,
        scalar_destination: int,
        symbolic_source: tuple[Int[S0], Int[S1]],
        symbolic_destination: tuple[Int[D0], Int[D1]],
        mixed_source: tuple[Literal[0], Int[S0]],
        mixed_destination: tuple[Int[D0], Literal[2]],
        dynamic: Any,
    ) -> None:
        assert_type(
            torch.movedim(x, scalar_source, scalar_destination), Tensor[IntTuple]
        )
        assert_type(
            torch.movedim(x, symbolic_source, symbolic_destination), Tensor[IntTuple]
        )
        assert_type(x.moveaxis(mixed_source, mixed_destination), Tensor[IntTuple])
        assert_type(x.movedim(dynamic, dynamic), Tensor[IntTuple])

    def check_symbolic_axis_errors[N: IntVar](axis: Int[N]) -> None:
        tensor: Tensor[[2, 3, 4]] = torch.ones((2, 3, 4))
        # Concrete errors must not be hidden by an unknown axis paired with them.
        torch.movedim(tensor, (axis,), (3,))  # E: destination dimension out of range
        torch.movedim(tensor, (3,), (axis,))  # E: source dimension out of range
        # E: destination dimensions must be unique
        torch.movedim(tensor, (axis, 1), (0, 0))
        # E: source dimensions must be unique
        torch.movedim(tensor, (0, 0), (axis, 1))

        scalar: Tensor[[]] = torch.tensor(1)
        torch.movedim(scalar, (axis,), (2,))  # E: destination dimension out of range
        # E: source dimensions must be unique
        torch.movedim(scalar, (0, -1), (axis, axis))
        # E: destination dimensions must be unique
        torch.movedim(scalar, (axis, 0), (0, -1))

    def check_mixed_overloads(tensor: Tensor[[2, 3, 4]]) -> None:
        torch.moveaxis(tensor, 0, (1,))  # E: No matching overload
        torch.movedim(tensor, (0,), 1)  # E: No matching overload
