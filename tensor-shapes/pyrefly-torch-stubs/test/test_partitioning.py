# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type, TYPE_CHECKING

import torch
from shape_extensions import assert_raises, assert_shape, Int, IntTuple, IntVar
from torch import Tensor


def test_chunk_shapes() -> None:
    chunks = torch.ones((4, 5, 18)).chunk(3, dim=2)
    assert len(chunks) == 3
    assert_shape(chunks[0].shape, (4, 5, 6))
    assert_shape(chunks[1].shape, (4, 5, 6))
    assert_shape(chunks[2].shape, (4, 5, 6))

    chunks = torch.chunk(torch.ones((4, 5, 13)), 6, dim=-1)
    assert len(chunks) == 5
    assert_shape(chunks[0].shape, (4, 5, 3))
    assert_shape(chunks[3].shape, (4, 5, 3))
    assert_shape(chunks[4].shape, (4, 5, 1))

    chunks = torch.ones((4, 5, 5)).chunk(8, dim=2)
    assert len(chunks) == 5
    assert_shape(chunks[0].shape, (4, 5, 1))
    assert_shape(chunks[4].shape, (4, 5, 1))

    chunks = torch.empty(0).chunk(3)
    assert len(chunks) == 3
    assert_shape(chunks[0].shape, (0,))
    assert_shape(chunks[2].shape, (0,))


def test_chunk_rejects_invalid_inputs() -> None:
    tensor = torch.ones((2, 3, 4))
    assert_shape(tensor.chunk(1, dim=1)[0].shape, (2, 3, 4))

    with assert_raises(RuntimeError):
        tensor.chunk(0, dim=1)  # E: chunk count must be greater than zero

    with assert_raises(RuntimeError):
        torch.chunk(tensor, -1, dim=1)  # E: chunk count must be greater than zero

    with assert_raises(IndexError):
        tensor.chunk(2, dim=3)  # E: chunk dimension out of range

    with assert_raises(IndexError):
        torch.chunk(tensor, 2, dim=-4)  # E: chunk dimension out of range


def test_split_shapes() -> None:
    parts = torch.ones((4, 5, 17)).split(6, dim=-1)
    assert len(parts) == 3
    assert_shape(parts[0].shape, (4, 5, 6))
    assert_shape(parts[1].shape, (4, 5, 6))
    assert_shape(parts[2].shape, (4, 5, 5))

    parts = torch.split(torch.ones((4, 9, 5)), (2, 3, 4), dim=1)
    assert len(parts) == 3
    assert_shape(parts[0].shape, (4, 2, 5))
    assert_shape(parts[1].shape, (4, 3, 5))
    assert_shape(parts[2].shape, (4, 4, 5))

    gradual_parts = torch.split(torch.ones((4, 9, 5)), [2, 3, 4], dim=1)
    assert len(gradual_parts) == 3
    assert_shape(gradual_parts[0].shape, IntTuple, runtime=(4, 2, 5))
    assert_shape(gradual_parts[2].shape, IntTuple, runtime=(4, 4, 5))


def test_split_rejects_invalid_dimensions() -> None:
    tensor = torch.ones((2, 3, 4))
    assert_shape(tensor.split(3, dim=1)[0].shape, (2, 3, 4))

    with assert_raises(IndexError):
        tensor.split(1, dim=3)  # E: split dimension out of range

    with assert_raises(IndexError):
        torch.split(tensor, (1, 2), dim=-4)  # E: split dimension out of range


def test_split_rejects_invalid_sizes() -> None:
    tensor = torch.ones((2, 3, 4))
    assert_shape(tensor.split(3, dim=1)[0].shape, (2, 3, 4))

    with assert_raises(RuntimeError):
        tensor.split(0, dim=1)  # E: split size can only be zero

    with assert_raises(RuntimeError):
        torch.split(tensor, -1, dim=1)  # E: split size must be non-negative

    with assert_raises(RuntimeError):
        tensor.split((1, -1, 3), dim=1)  # E: split sections must be non-negative

    with assert_raises(RuntimeError):
        # E: split sections must sum to the selected dimension
        torch.split(tensor, (1, 1), dim=1)


if TYPE_CHECKING:

    def check_chunk_symbolic[B: IntVar, N: IntVar](
        divisible: Tensor[[B, 3 * N]],
        arbitrary: Tensor[[B, N]],
        n: Int[N],
        chunks: int,
        dim: int,
    ) -> None:
        assert_type(
            divisible.chunk(3, dim=1),
            tuple[Tensor[[B, N]], Tensor[[B, N]], Tensor[[B, N]]],
        )
        assert_type(
            torch.chunk(divisible, 3, dim=1),
            tuple[Tensor[[B, N]], Tensor[[B, N]], Tensor[[B, N]]],
        )
        assert_type(arbitrary.chunk(3, dim=1), tuple[Tensor[[B, int]], ...])
        assert_type(divisible.chunk(n, dim=1), tuple[Tensor[[B, 3]], ...])
        assert_type(divisible.chunk(chunks, dim=1), tuple[Tensor[[B, int]], ...])
        assert_type(divisible.chunk(3, dim=dim), tuple[Tensor, ...])

    def check_split_symbolic[B: IntVar, T: IntVar, N: IntVar](
        divisible: Tensor[[B, T, 3 * N]],
        arbitrary: Tensor[[B, T, N]],
        n: Int[N],
        split_size: int,
    ) -> None:
        assert_type(
            divisible.split(n, dim=2),
            tuple[Tensor[[B, T, N]], Tensor[[B, T, N]], Tensor[[B, T, N]]],
        )
        assert_type(
            torch.split(divisible, (n, n, n), dim=2),
            tuple[Tensor[[B, T, N]], Tensor[[B, T, N]], Tensor[[B, T, N]]],
        )
        assert_type(divisible.split(3, dim=2), tuple[Tensor[[B, T, 3]], ...])
        assert_type(arbitrary.split(3, dim=2), tuple[Tensor[[B, T, int]], ...])
        assert_type(divisible.split(split_size, dim=2), tuple[Tensor[[B, T, int]], ...])

    def check_list_sections(x: Tensor[[4, 9, 5]]) -> None:
        assert_type(x.split([2, 3, 4], dim=1), tuple[Tensor, ...])
        assert_type(torch.split(x, [2, 3, 4], dim=1), tuple[Tensor, ...])
