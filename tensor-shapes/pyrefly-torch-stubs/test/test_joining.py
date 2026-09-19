# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, assert_type, TYPE_CHECKING

import torch
from shape_extensions import assert_raises, assert_shape, IntTuple, IntVar
from torch import Tensor


def test_concatenate_shapes() -> None:
    assert_shape(torch.cat((torch.ones(2), torch.ones(3))).shape, (5,))
    assert_shape(
        torch.cat((torch.ones((2, 3)), torch.ones((4, 3))), dim=0).shape,
        (6, 3),
    )
    assert_shape(
        torch.cat((torch.ones((2, 3)), torch.ones((2, 4))), dim=-1).shape,
        (2, 7),
    )
    assert_shape(
        torch.concat(
            (torch.ones((2, 3)), torch.ones((1, 3)), torch.ones((4, 3)))
        ).shape,
        (7, 3),
    )
    assert_shape(
        torch.concatenate((torch.zeros((2, 3)), torch.zeros((2, 4))), axis=1).shape,
        (2, 7),
    )


def test_stack_shapes() -> None:
    vector = torch.ones(3)
    assert_shape(torch.stack((vector, vector)).shape, (2, 3))
    assert_shape(torch.stack((vector, vector), dim=1).shape, (3, 2))
    assert_shape(torch.stack((vector, vector), dim=-1).shape, (3, 2))

    matrix = torch.ones((2, 3))
    assert_shape(torch.stack((matrix, matrix, matrix)).shape, (3, 2, 3))
    assert_shape(torch.stack((matrix, matrix, matrix), dim=1).shape, (2, 3, 3))
    assert_shape(torch.stack((matrix, matrix), dim=-1).shape, (2, 3, 2))


def test_concatenate_rejects_invalid_inputs() -> None:
    assert_shape(torch.cat((torch.ones(2), torch.ones(3))).shape, (5,))

    with assert_raises(ValueError):
        torch.cat(())  # E: cat expects a non-empty sequence of tensors

    matrix = torch.ones((2, 3))
    with assert_raises(IndexError):
        torch.cat((matrix, matrix), dim=2)  # E: cat dimension out of range

    with assert_raises(RuntimeError):
        # E: cat expects all tensors to have the same rank
        torch.cat((matrix, torch.ones((2, 3, 4))))

    with assert_raises(RuntimeError):
        # E: cat expects all tensor sizes to match outside the concatenated dimension
        torch.concatenate((matrix, torch.ones((4, 4))), axis=1)


def test_stack_rejects_invalid_inputs() -> None:
    assert_shape(torch.stack((torch.ones(2), torch.ones(2))).shape, (2, 2))

    with assert_raises(RuntimeError):
        torch.stack(())  # E: stack expects a non-empty sequence of tensors

    matrix = torch.ones((2, 3))
    with assert_raises(IndexError):
        torch.stack((matrix, matrix), dim=3)  # E: stack dimension out of range

    with assert_raises(RuntimeError):
        # E: stack expects all tensors to have the same shape
        torch.stack((matrix, torch.ones((2, 4))), dim=1)


if TYPE_CHECKING:

    def check_symbolic_shapes[N: IntVar, M: IntVar](
        x: Tensor[[N, 3]], y: Tensor[[M, 3]]
    ) -> None:
        assert_type(torch.cat((x, y)), Tensor[[N + M, 3]])
        assert_type(torch.concat((x, y, x)), Tensor[[2 * N + M, 3]])
        assert_type(torch.stack((x, x), dim=1), Tensor[[N, 2, 3]])

    def check_gradual_boundaries[N: IntVar](
        known: Tensor[[2, 3]],
        symbolic: Tensor[[2, N]],
        gradual: Tensor[[2, Any]],
        shapeless: Tensor,
        tensors: Sequence[Tensor[[2, 3]]],
        dim: int,
    ) -> None:
        assert_type(torch.cat((symbolic, symbolic)), Tensor[[4, N]])
        assert_type(torch.stack((symbolic, symbolic)), Tensor[[2, 2, N]])

        # Unknown sequence members, sizes, or axes make the result genuinely gradual.
        assert_type(torch.cat(tensors), Tensor[IntTuple])
        assert_type(torch.cat((known, symbolic)), Tensor[IntTuple])
        assert_type(torch.cat((known, gradual)), Tensor[IntTuple])
        assert_type(torch.concat((known, shapeless)), Tensor[IntTuple])
        assert_type(torch.stack((known, known), dim=dim), Tensor[IntTuple])

    def check_invalid_elements(x: Tensor[[2, 3]]) -> None:
        torch.cat((x, 1))  # E: is not assignable to parameter `tensors`
        torch.concat([x, "not a tensor"])  # E: is not assignable to parameter `tensors`
        # The broad compatibility overload intentionally accepts heterogeneous sequences.
        assert_type(torch.stack((x, object())), Tensor)
