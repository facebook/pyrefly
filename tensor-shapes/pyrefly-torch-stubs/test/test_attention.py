# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type, TYPE_CHECKING

import torch
import torch.nn.functional as F
from shape_extensions import assert_raises, assert_shape, IntVar
from torch import Tensor


def test_scaled_dot_product_attention_shapes() -> None:
    query = torch.ones((2, 4, 3, 8))
    key = torch.ones((2, 4, 5, 8))
    value = torch.ones((2, 4, 5, 6))
    assert_shape(F.scaled_dot_product_attention(query, key, value).shape, (2, 4, 3, 6))
    assert_shape(
        F.scaled_dot_product_attention(query, key, value, is_causal=True).shape,
        (2, 4, 3, 6),
    )

    mask = torch.ones((3, 5), dtype=torch.bool)
    assert_shape(
        F.scaled_dot_product_attention(query, key, value, attn_mask=mask).shape,
        (2, 4, 3, 6),
    )


def test_scaled_dot_product_attention_additional_batch_ranks() -> None:
    query = torch.ones((4, 3, 8))
    key = torch.ones((4, 5, 8))
    value = torch.ones((4, 5, 6))
    assert_shape(F.scaled_dot_product_attention(query, key, value).shape, (4, 3, 6))

    query = torch.ones((2, 3, 4, 5, 8))
    key = torch.ones((2, 3, 4, 6, 8))
    value = torch.ones((2, 3, 4, 6, 7))
    assert_shape(
        F.scaled_dot_product_attention(query, key, value).shape,
        (2, 3, 4, 5, 7),
    )


def test_scaled_dot_product_attention_broadcasts_batch_dimensions() -> None:
    query = torch.ones((2, 1, 3, 8))
    key = torch.ones((1, 4, 5, 8))
    value = torch.ones((1, 4, 5, 6))
    assert_shape(F.scaled_dot_product_attention(query, key, value).shape, (2, 4, 3, 6))


def test_scaled_dot_product_attention_rejects_incompatible_shapes() -> None:
    query = torch.ones((2, 4, 3, 8))
    key = torch.ones((2, 4, 5, 8))
    value = torch.ones((2, 4, 5, 6))
    assert_shape(F.scaled_dot_product_attention(query, key, value).shape, (2, 4, 3, 6))

    with assert_raises(RuntimeError):
        # E: core dimension 'e' has conflicting extents 8 and 7
        F.scaled_dot_product_attention(query, torch.ones((2, 4, 5, 7)), value)

    with assert_raises(RuntimeError):
        # E: core dimension 's' has conflicting extents 5 and 6
        F.scaled_dot_product_attention(query, key, torch.ones((2, 4, 6, 6)))


if TYPE_CHECKING:

    def check_symbolic_attention[
        B: IntVar,
        H: IntVar,
        Tq: IntVar,
        Tkv: IntVar,
        D: IntVar,
        Dv: IntVar,
    ](
        query: Tensor[[B, H, Tq, D]],
        key: Tensor[[B, H, Tkv, D]],
        value: Tensor[[B, H, Tkv, Dv]],
    ) -> None:
        assert_type(
            F.scaled_dot_product_attention(query, key, value),
            Tensor[[B, H, Tq, Dv]],
        )
