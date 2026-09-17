# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Scalar shape arguments typed as `Int`-bounded type parameters.

These arguments are ordinary type parameters bounded by `Int`, and their runtime
arguments are typed with the parameter directly rather than through `Int[...]`.
A literal argument therefore keeps its value, while a runtime `builtins.int` has
no dimension to keep and lowers to the gradual size — widening only the extents
that the argument feeds, never the rank or the untouched axes.
"""

from typing import assert_type, TYPE_CHECKING

import torch
import torch.nn.functional as F
from shape_extensions import IntVar

if TYPE_CHECKING:
    from torch import Tensor


def test_topk_literal_and_gradual_k[B: IntVar](logits: Tensor[[B, 32]], k: int) -> None:
    literal_values, literal_indices = torch.topk(logits, 5)
    assert_type(literal_values, Tensor[[B, 5]])
    assert_type(literal_indices, Tensor[[B, 5]])

    values, indices = torch.topk(logits, k)
    # Rank and the symbolic batch axis survive; only the top-k extent is gradual.
    assert_type(values, Tensor[[B, int]])
    assert_type(indices, Tensor[[B, int]])


def test_topk_method_gradual_k[B: IntVar](logits: Tensor[[B, 32]], k: int) -> None:
    values, _ = logits.topk(k, dim=0)
    assert_type(values, Tensor[[int, 32]])


def test_narrow_literal_and_gradual_length[B: IntVar](
    x: Tensor[[B, 32, 8]], length: int
) -> None:
    assert_type(torch.narrow(x, 1, 0, 4), Tensor[[B, 4, 8]])
    assert_type(torch.narrow(x, 1, 0, length), Tensor[[B, int, 8]])
    assert_type(x.narrow(1, 0, length), Tensor[[B, int, 8]])


def test_multinomial_literal_and_gradual_samples[B: IntVar](
    weights: Tensor[[B, 32]], num_samples: int
) -> None:
    assert_type(torch.multinomial(weights, 3), Tensor[[B, 3]])
    assert_type(torch.multinomial(weights, num_samples), Tensor[[B, int]])
    assert_type(weights.multinomial(num_samples), Tensor[[B, int]])


def test_repeat_interleave_gradual_repeats[B: IntVar](
    x: Tensor[[B, 32]], repeats: int
) -> None:
    assert_type(torch.repeat_interleave(x, 2, dim=1), Tensor[[B, 64]])
    assert_type(torch.repeat_interleave(x, repeats, dim=1), Tensor[[B, int]])


def test_repeat_interleave_gradual_output_size[B: IntVar](
    x: Tensor[[B, 32]], output_size: int
) -> None:
    assert_type(
        torch.repeat_interleave(x, torch.ones(32), dim=1, output_size=output_size),
        Tensor[[B, int]],
    )


def test_adaptive_pool_gradual_output_size[B: IntVar](
    x: Tensor[[B, 64, 56, 56]], out: int
) -> None:
    # A scalar output size is shared across both spatial axes, so both go gradual
    # while the batch and channel axes stay exactly as declared.
    assert_type(F.adaptive_avg_pool2d(x, out), Tensor[[B, 64, int, int]])
    assert_type(F.adaptive_max_pool2d(x, out), Tensor[[B, 64, int, int]])
    # A tuple output size widens only the axes it names.
    assert_type(F.adaptive_avg_pool2d(x, (out, 7)), Tensor[[B, 64, int, 7]])
