# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import assert_type, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor
    from torch_shapes import Dim


def test_view[B, T, D, NHead](
    x: Tensor[B, T, D],
    bsz: Dim[B],
    seqlen: Dim[T],
    n_head: Dim[NHead],
    head_dim: Dim[D // NHead],
) -> None:
    # Test view operation
    result = x.view(bsz, seqlen, n_head, head_dim)
    assert_type(result, Tensor[B, T, NHead, (D // NHead)])


def test_transpose[B, T, NHead, HeadDim](
    q: Tensor[B, T, NHead, HeadDim],
) -> None:
    # Test transpose operation
    result = q.transpose(1, 2)
    assert_type(result, Tensor[B, NHead, T, HeadDim])


def test_split[B, T, D, NLocalHeads, NHead](
    x: Tensor[B, T, (NHead + 2 * NLocalHeads) * (D // NHead)],
    dim: Dim[D],
    kv_size: Dim[NLocalHeads * (D // NHead)],
) -> None:
    # Test split with tuple (required for meta-shape inference)
    q, k, v = x.split((dim, kv_size, kv_size), dim=-1)
    assert_type(q, Tensor[B, T, D])
    assert_type(k, Tensor[B, T, (NLocalHeads * (D // NHead))])


def test_flatten[B, T, NHeads, HeadDim](
    x: Tensor[B, T, NHeads, HeadDim // 2, 2],
) -> None:
    # Test flatten operation
    result = x.flatten(3)
    # Result is Tensor[B, T, NHeads, ((HeadDim // 2) * 2)]
    # Note: algebraic equivalence to HeadDim is Issue 7
    assert_type(result, Tensor[B, T, NHeads, ((HeadDim // 2) * 2)])


def test_stack[SeqLen, HeadDim](
    real: Tensor[SeqLen, HeadDim // 2],
    imag: Tensor[SeqLen, HeadDim // 2],
) -> None:
    # Test stack with tuple (required for meta-shape inference)
    result = torch.stack((real, imag), dim=-1)
    assert_type(result, Tensor[SeqLen, HeadDim // 2, 2])
