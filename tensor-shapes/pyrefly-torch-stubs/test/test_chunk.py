# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import assert_type, reveal_type, TYPE_CHECKING

import torch
from shape_extensions import IntVar

if TYPE_CHECKING:
    from shape_extensions import Int
    from torch import Tensor


def test_chunk_divisible(x: Tensor[[4, 5, 18]]) -> None:
    assert_type(
        x.chunk(3, dim=2),
        tuple[Tensor[[4, 5, 6]], Tensor[[4, 5, 6]], Tensor[[4, 5, 6]]],
    )
    assert_type(
        torch.chunk(x, 3, dim=2),
        tuple[Tensor[[4, 5, 6]], Tensor[[4, 5, 6]], Tensor[[4, 5, 6]]],
    )


def test_chunk_nondivisible(x: Tensor[[4, 5, 13]]) -> None:
    assert_type(
        x.chunk(6, dim=2),
        tuple[
            Tensor[[4, 5, 3]],
            Tensor[[4, 5, 3]],
            Tensor[[4, 5, 3]],
            Tensor[[4, 5, 3]],
            Tensor[[4, 5, 1]],
        ],
    )
    assert_type(
        torch.chunk(x, 6, dim=2),
        tuple[
            Tensor[[4, 5, 3]],
            Tensor[[4, 5, 3]],
            Tensor[[4, 5, 3]],
            Tensor[[4, 5, 3]],
            Tensor[[4, 5, 1]],
        ],
    )


def test_chunk_more_chunks_than_extent(x: Tensor[[4, 5, 5]]) -> None:
    assert_type(
        x.chunk(8, dim=2),
        tuple[
            Tensor[[4, 5, 1]],
            Tensor[[4, 5, 1]],
            Tensor[[4, 5, 1]],
            Tensor[[4, 5, 1]],
            Tensor[[4, 5, 1]],
        ],
    )


def test_chunk_tail_and_negative_dim(x: Tensor[[4, 5, 17]]) -> None:
    assert_type(
        torch.chunk(x, 3, dim=-1),
        tuple[Tensor[[4, 5, 6]], Tensor[[4, 5, 6]], Tensor[[4, 5, 5]]],
    )


def test_chunk_zero_extent() -> None:
    reveal_type(
        torch.empty(0).chunk(3)
    )  # revealed type: tuple[Tensor[[0]], Tensor[[0]], Tensor[[0]]]
    reveal_type(
        torch.chunk(torch.empty(0), 3)
    )  # revealed type: tuple[Tensor[[0]], Tensor[[0]], Tensor[[0]]]


def test_chunk_symbolic_divisible[B: IntVar, N: IntVar](
    x: Tensor[[B, (3 * N)]],
) -> None:
    assert_type(
        x.chunk(3, dim=1), tuple[Tensor[[B, N]], Tensor[[B, N]], Tensor[[B, N]]]
    )
    assert_type(
        torch.chunk(x, 3, dim=1),
        tuple[Tensor[[B, N]], Tensor[[B, N]], Tensor[[B, N]]],
    )


def test_chunk_symbolic_remainder[B: IntVar, N: IntVar](x: Tensor[[B, N]]) -> None:
    assert_type(x.chunk(3, dim=1), tuple[Tensor[[B, int]], ...])
    assert_type(torch.chunk(x, 3, dim=1), tuple[Tensor[[B, int]], ...])


def test_chunk_symbolic_count[B: IntVar, N: IntVar](
    x: Tensor[[B, (3 * N)]], chunks: Int[N]
) -> None:
    assert_type(x.chunk(chunks, dim=1), tuple[Tensor[[B, 3]], ...])
    assert_type(torch.chunk(x, chunks, dim=1), tuple[Tensor[[B, 3]], ...])


def test_chunk_gradual_count(x: Tensor[[4, 5, 18]], chunks: int) -> None:
    assert_type(x.chunk(chunks, dim=2), tuple[Tensor[[4, 5, int]], ...])
    assert_type(torch.chunk(x, chunks, dim=2), tuple[Tensor[[4, 5, int]], ...])


def test_chunk_gradual_axis(x: Tensor[[4, 5, 18]], dim: int) -> None:
    assert_type(x.chunk(3, dim=dim), tuple[Tensor, ...])
    assert_type(torch.chunk(x, 3, dim=dim), tuple[Tensor, ...])
