# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type, TYPE_CHECKING

import torch
from shape_extensions import assert_raises, assert_shape, Int, IntTuple, IntVar
from torch import Tensor


def test_multinomial_shapes() -> None:
    assert_shape(torch.multinomial(torch.ones(5), 3).shape, (3,))
    assert_shape(torch.ones((4, 5)).multinomial(3).shape, (4, 3))
    assert_shape(
        torch.multinomial(torch.ones((2, 5)), 6, replacement=True).shape,
        (2, 6),
    )


def test_multinomial_rejects_invalid_ranks() -> None:
    vector = torch.ones(5)
    assert_shape(vector.multinomial(1).shape, (1,))

    scalar = torch.tensor(1.0)
    with assert_raises(RuntimeError):
        torch.multinomial(scalar, 1)  # E: multinomial expects 1D or 2D input

    cube = torch.ones((2, 3, 4))
    with assert_raises(RuntimeError):
        cube.multinomial(1)  # E: multinomial expects 1D or 2D input


def test_multinomial_rejects_invalid_sample_counts() -> None:
    probabilities = torch.ones(5)
    assert_shape(probabilities.multinomial(5).shape, (5,))

    with assert_raises(RuntimeError):
        # E: multinomial num_samples must be positive
        torch.multinomial(probabilities, 0)

    with assert_raises(RuntimeError):
        # E: multinomial sample count exceeds category count
        probabilities.multinomial(6)


if TYPE_CHECKING:

    def check_symbolic[B: IntVar, N: IntVar, K: IntVar](
        probabilities: Tensor[[B, N]], num_samples: Int[K]
    ) -> None:
        assert_type(torch.multinomial(probabilities, num_samples), Tensor[[B, K]])

    def check_gradual_boundaries(
        probabilities: Tensor[[2, 5]],
        num_samples: int,
        replacement: bool,
        bare: Tensor,
    ) -> None:
        assert_type(probabilities.multinomial(num_samples), Tensor[[2, int]])
        assert_type(
            probabilities.multinomial(3, replacement=replacement), Tensor[[2, 3]]
        )
        assert_type(torch.multinomial(bare, 2), Tensor[IntTuple])

    def check_invalid_sample_count_type[T, S: str](
        probabilities: Tensor[[4, 32]], unconstrained: T, string: S
    ) -> None:
        # E: `T` is not assignable to upper bound `Int[int]` of type variable `NumSamples`
        torch.multinomial(probabilities, unconstrained)
        # E: `S` is not assignable to upper bound `Int[int]` of type variable `NumSamples`
        torch.multinomial(probabilities, string)
