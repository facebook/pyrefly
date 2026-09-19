# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import warnings
from typing import assert_type, TYPE_CHECKING

import torch
import torch.nn.functional as F
from shape_extensions import assert_raises, assert_shape, IntTuple, IntVar
from torch import Tensor


def test_regression_loss_shapes() -> None:
    matrix = torch.randn((2, 3))
    target = torch.randn((2, 3))
    broadcast_target = torch.randn((1, 3))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for loss in (F.mse_loss, F.l1_loss, F.smooth_l1_loss, F.huber_loss):
            assert_shape(loss(matrix, target).shape, ())
            assert_shape(loss(matrix, broadcast_target, reduction="none").shape, (2, 3))


def test_legacy_reduction_precedence() -> None:
    matrix = torch.randn((2, 3))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        assert_shape(
            F.l1_loss(matrix, matrix, reduce=False, reduction="sum").shape,
            (2, 3),
        )
        assert_shape(
            F.l1_loss(
                matrix,
                matrix,
                size_average=False,
                reduce=True,
                reduction="none",
            ).shape,
            (),
        )


def test_regression_losses_reject_invalid_inputs() -> None:
    matrix = torch.randn((2, 3))
    assert_shape(F.mse_loss(matrix, matrix).shape, ())

    with assert_raises(ValueError):
        F.l1_loss(matrix, matrix, reduction=1)  # E: not a valid `Flag[str]` value

    with assert_raises(ValueError):
        # E: loss reduction must be
        F.huber_loss(matrix, matrix, reduction="invalid")

    incompatible = torch.randn((4, 5))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for loss in (F.mse_loss, F.l1_loss, F.smooth_l1_loss, F.huber_loss):
            with assert_raises(RuntimeError):
                # E: Cannot broadcast dimension
                loss(matrix, incompatible, reduction="none")


if TYPE_CHECKING:

    def check_symbolic_regression_losses[N: IntVar, M: IntVar](
        input: Tensor[[N, 1]], target: Tensor[[1, M]]
    ) -> None:
        assert_type(F.mse_loss(input, target, reduction="none"), Tensor[[N, M]])
        assert_type(F.l1_loss(input, target, reduction="none"), Tensor[[N, M]])
        assert_type(F.smooth_l1_loss(input, target, reduction="none"), Tensor[[N, M]])
        assert_type(F.huber_loss(input, target, reduction="none"), Tensor[[N, M]])

    def check_gradual_regression_loss_controls(
        input: Tensor[[2, 3]],
        target: Tensor,
        reduction: str,
        reduce: bool | None,
        size_average: bool | None,
    ) -> None:
        assert_type(F.l1_loss(input, target, reduction="none"), Tensor[IntTuple])
        assert_type(F.l1_loss(input, target), Tensor[[]])
        assert_type(F.l1_loss(input, input, reduction=reduction), Tensor[IntTuple])
        assert_type(F.l1_loss(input, input, reduce=reduce), Tensor[IntTuple])
        assert_type(
            F.l1_loss(input, input, size_average=size_average), Tensor[IntTuple]
        )
