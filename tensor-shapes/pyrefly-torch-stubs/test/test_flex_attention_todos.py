# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type, Callable, TYPE_CHECKING

import torch
from shape_extensions import assert_shape, Int, IntVar
from torch import Tensor
from torch.nn.attention.flex_attention import (
    and_masks,
    BlockMask,
    create_block_mask,
    create_mask,
    noop_mask,
    or_masks,
)


def causal_mask(batch: Tensor, head: Tensor, query: Tensor, key: Tensor) -> Tensor:
    return query >= key


def test_mask_modifiers() -> None:
    index = torch.tensor(0)
    query = torch.tensor(2)
    key = torch.tensor(3)

    assert_shape(noop_mask(index, index, query, key).shape, ())
    assert and_masks(noop_mask, causal_mask)(index, index, query, key).item() is False
    assert or_masks(noop_mask, causal_mask)(index, index, query, key).item() is True


def test_create_mask_shapes() -> None:
    mask = create_mask(causal_mask, 2, 3, 4, 5, device="cpu")
    default_axes = create_mask(noop_mask, None, None, 4, 5, device="cpu")

    assert_shape(mask.shape, (2, 3, 4, 5))
    assert_shape(default_axes.shape, (1, 1, 4, 5))
    assert mask[0, 0, 2].tolist() == [True, True, True, False, False]


def test_create_block_mask() -> None:
    mask = create_block_mask(causal_mask, 1, 1, 4, 4, device="cpu", BLOCK_SIZE=4)

    assert_shape(mask.kv_num_blocks.shape, (int, int, int), runtime=(1, 1, 1))
    assert mask.kv_num_blocks.shape == (1, 1, 1)
    assert mask.seq_lengths == (4, 4)


if TYPE_CHECKING:

    def check_symbolic_mask[B: IntVar, H: IntVar, Q: IntVar, K: IntVar](
        batch: Int[B], head: Int[H], query: Int[Q], key: Int[K]
    ) -> None:
        assert_type(
            create_mask(causal_mask, batch, head, query, key), Tensor[[B, H, Q, K]]
        )
        assert_type(
            create_mask(noop_mask, None, None, query, key), Tensor[[1, 1, Q, K]]
        )
        assert_type(create_block_mask(causal_mask, batch, head, query, key), BlockMask)
        assert_type(
            noop_mask(
                torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(0)
            ),
            Tensor[[]],
        )
        assert_type(
            and_masks(noop_mask, causal_mask),
            Callable[[Tensor, Tensor, Tensor, Tensor], Tensor],
        )
        assert_type(
            or_masks(noop_mask, causal_mask),
            Callable[[Tensor, Tensor, Tensor, Tensor], Tensor],
        )
