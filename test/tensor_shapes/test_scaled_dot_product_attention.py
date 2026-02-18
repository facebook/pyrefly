# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test scaled_dot_product_attention shape inference"""

from typing import assert_type, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor
    from torch.nn import functional as F


def test_self_attention[B, NHead, T, D](
    q: Tensor[B, NHead, T, D],
    k: Tensor[B, NHead, T, D],
    v: Tensor[B, NHead, T, D],
):
    # Self-attention: all same shape, output matches input
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert_type(out, Tensor[B, NHead, T, D])


def test_cross_attention[B, H, Tq, Tkv, D](
    query: Tensor[B, H, Tq, D],
    key: Tensor[B, H, Tkv, D],
    value: Tensor[B, H, Tkv, D],
):
    # Cross-attention: query has different sequence length
    out = F.scaled_dot_product_attention(query, key, value)
    assert_type(out, Tensor[B, H, Tq, D])


def test_different_value_dim[B, H, Tq, Tkv, D, Dv](
    query: Tensor[B, H, Tq, D],
    key: Tensor[B, H, Tkv, D],
    value: Tensor[B, H, Tkv, Dv],
):
    # Most general: value has different feature dimension
    out = F.scaled_dot_product_attention(query, key, value)
    assert_type(out, Tensor[B, H, Tq, Dv])


def test_symbolic_arith[B, H, T, E](
    q: Tensor[B, H, T, (E // H)],
    k: Tensor[B, H, T, (E // H)],
    v: Tensor[B, H, T, (E // H)],
):
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert_type(out, Tensor[B, H, T, (E // H)])
