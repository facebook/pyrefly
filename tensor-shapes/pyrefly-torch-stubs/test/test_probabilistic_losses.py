# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type, TYPE_CHECKING

import torch
import torch.nn.functional as F
from shape_extensions import assert_raises, assert_shape, IntTuple, IntVar
from torch import Tensor


def test_binary_cross_entropy_shapes() -> None:
    probabilities = torch.rand((2, 3))
    targets = torch.rand((2, 3))
    logits = torch.randn((2, 3))

    assert_shape(F.binary_cross_entropy(probabilities, targets).shape, ())
    assert_shape(
        F.binary_cross_entropy(probabilities, targets, reduction="none").shape,
        (2, 3),
    )
    assert_shape(F.binary_cross_entropy_with_logits(logits, targets).shape, ())
    assert_shape(
        F.binary_cross_entropy_with_logits(logits, targets, reduction="none").shape,
        (2, 3),
    )


def test_broadcasting_probabilistic_loss_shapes() -> None:
    input = torch.randn((2, 1))
    target = torch.rand((1, 3))

    assert_shape(F.kl_div(input, target, reduction="none").shape, (2, 3))
    assert_shape(F.kl_div(input, target, reduction="batchmean").shape, ())
    assert_shape(F.poisson_nll_loss(input, target, reduction="none").shape, (2, 3))
    assert_shape(F.poisson_nll_loss(input, target).shape, ())


def test_binary_cross_entropy_rejects_different_shapes() -> None:
    input = torch.rand((2, 3))
    target = torch.rand((1, 3))
    assert_shape(F.binary_cross_entropy(input, input).shape, ())

    with assert_raises(ValueError):
        # E: is not assignable to parameter `target`
        F.binary_cross_entropy(input, target, reduction="none")

    with assert_raises(ValueError):
        # E: is not assignable to parameter `target`
        F.binary_cross_entropy_with_logits(input, target, reduction="none")


if TYPE_CHECKING:

    def check_symbolic_probabilistic_losses[N: IntVar, M: IntVar](
        input: Tensor[[N, 1]], target: Tensor[[1, M]]
    ) -> None:
        assert_type(F.kl_div(input, target, reduction="none"), Tensor[[N, M]])
        assert_type(F.poisson_nll_loss(input, target, reduction="none"), Tensor[[N, M]])

    def check_gradual_probabilistic_loss_controls(
        input: Tensor[[2, 3]],
        target: Tensor,
        reduction: str,
        reduce: bool | None,
    ) -> None:
        assert_type(F.kl_div(input, target, reduction="none"), Tensor[IntTuple])
        assert_type(F.kl_div(input, target, reduction="batchmean"), Tensor[[]])
        assert_type(F.kl_div(input, input, reduction=reduction), Tensor[IntTuple])
        assert_type(F.poisson_nll_loss(input, input, reduce=reduce), Tensor[IntTuple])
