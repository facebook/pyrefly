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


def test_classification_loss_shapes() -> None:
    matrix = torch.randn((3, 10))
    vector_target = torch.zeros(3, dtype=torch.long)
    assert_shape(F.cross_entropy(matrix, vector_target).shape, ())
    assert_shape(F.cross_entropy(matrix, vector_target, reduction="none").shape, (3,))
    assert_shape(F.nll_loss(matrix, vector_target, reduction="none").shape, (3,))

    tensor = torch.randn((3, 10, 8, 6))
    tensor_target = torch.zeros((3, 8, 6), dtype=torch.long)
    assert_shape(
        F.cross_entropy(input=tensor, target=tensor_target, reduction="none").shape,
        (3, 8, 6),
    )
    assert_shape(F.nll_loss(tensor, tensor_target, reduction="none").shape, (3, 8, 6))

    vector = torch.randn(10)
    scalar_target = torch.tensor(1)
    assert_shape(F.cross_entropy(vector, scalar_target, reduction="none").shape, ())
    assert_shape(F.nll_loss(vector, scalar_target, reduction="none").shape, ())


def test_classification_losses_reject_scalar_input() -> None:
    matrix = torch.randn((2, 3))
    target = torch.zeros(2, dtype=torch.long)
    assert_shape(F.cross_entropy(matrix, target).shape, ())

    scalar = torch.randn(())
    scalar_target = torch.tensor(0)
    with assert_raises(RuntimeError):
        F.cross_entropy(  # E: classification loss requires a class dimension
            scalar, scalar_target
        )

    with assert_raises(ValueError):
        F.nll_loss(  # E: classification loss requires a class dimension
            scalar, scalar_target
        )


if TYPE_CHECKING:

    def check_symbolic_classification_losses[N: IntVar, C: IntVar](
        input: Tensor[[N, C]], target: Tensor[[N]]
    ) -> None:
        assert_type(F.cross_entropy(input, target, reduction="none"), Tensor[[N]])
        assert_type(F.nll_loss(input, target, reduce=False), Tensor[[N]])

    def check_gradual_classification_loss_controls(
        input: Tensor[IntTuple],
        target: Tensor[[2]],
        reduction: str,
        reduce: bool | None,
    ) -> None:
        assert_type(F.cross_entropy(input, target, reduction="none"), Tensor[IntTuple])
        # TODO: BUG: A known reduction should produce a scalar for unknown rank.
        assert_type(F.cross_entropy(input, target), Tensor)
        assert_type(
            F.cross_entropy(input, target, reduction=reduction), Tensor[IntTuple]
        )
        assert_type(F.nll_loss(input, target, reduce=reduce), Tensor[IntTuple])
