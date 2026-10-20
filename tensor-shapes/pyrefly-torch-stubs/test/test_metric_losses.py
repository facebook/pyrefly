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


def test_cosine_embedding_loss_shapes() -> None:
    vector = torch.randn(10)
    scalar_target = torch.tensor(1)
    assert_shape(
        F.cosine_embedding_loss(vector, vector, scalar_target, reduction="none").shape,
        (),
    )

    matrix = torch.randn((3, 10))
    broadcast_input = torch.randn((1, 10))
    target = torch.ones(3)
    assert_shape(
        F.cosine_embedding_loss(matrix, matrix, target, reduction="none").shape,
        (3,),
    )
    assert_shape(
        F.cosine_embedding_loss(
            matrix, broadcast_input, target, reduction="none"
        ).shape,
        (3,),
    )
    assert_shape(F.cosine_embedding_loss(matrix, matrix, target).shape, ())


def test_ranking_loss_shapes() -> None:
    left = torch.randn((2, 1))
    right = torch.randn((1, 3))
    target = torch.ones((2, 3))
    assert_shape(
        F.margin_ranking_loss(left, right, target, reduction="none").shape,
        (2, 3),
    )
    assert_shape(F.hinge_embedding_loss(left, target, reduction="none").shape, (2, 3))

    anchor = torch.randn((2, 1))
    positive = torch.randn((2, 3))
    negative = torch.randn((2, 4))
    assert_shape(
        F.triplet_margin_loss(
            anchor=anchor,
            positive=positive,
            negative=negative,
            reduction="none",
        ).shape,
        (2,),
    )


def test_cosine_embedding_loss_rejects_invalid_shapes() -> None:
    vector = torch.randn(3)
    matrix = torch.randn((2, 3))
    rank_four = torch.randn((2, 3, 4, 5))
    scalar_target = torch.tensor(1)
    vector_target = torch.ones(2)
    assert_shape(
        F.cosine_embedding_loss(vector, vector, scalar_target).shape,
        (),
    )

    with assert_raises(RuntimeError):
        F.cosine_embedding_loss(  # E: requires 1D or 2D inputs
            rank_four, rank_four, vector_target, reduction="none"
        )
    with assert_raises(RuntimeError):
        F.cosine_embedding_loss(  # E: requires a scalar target for 1D inputs
            vector, vector, vector_target, reduction="none"
        )
    with assert_raises(RuntimeError):
        F.cosine_embedding_loss(  # E: requires a 1D target for 2D inputs
            matrix, matrix, scalar_target, reduction="none"
        )
    with assert_raises(RuntimeError):
        F.cosine_embedding_loss(  # E: Cannot broadcast dimension
            matrix, matrix, torch.ones(3), reduction="none"
        )


def test_triplet_margin_loss_scalar_inputs() -> None:
    scalar = torch.randn(())
    assert_shape(
        F.triplet_margin_loss(scalar, scalar, scalar, reduction="none").shape,
        (),
    )


if TYPE_CHECKING:

    def check_symbolic_metric_losses[N: IntVar, C: IntVar, M: IntVar](
        matrix: Tensor[[N, C]],
        broadcast_matrix: Tensor[[1, C]],
        vector_target: Tensor[[N]],
        left: Tensor[[N, 1]],
        right: Tensor[[1, M]],
        matrix_target: Tensor[[N, M]],
    ) -> None:
        assert_type(
            F.cosine_embedding_loss(
                matrix, broadcast_matrix, vector_target, reduction="none"
            ),
            Tensor[[N]],
        )
        assert_type(
            F.margin_ranking_loss(left, right, matrix_target, reduction="none"),
            Tensor[[N, M]],
        )
        assert_type(
            F.hinge_embedding_loss(left, matrix_target, reduction="none"),
            Tensor[[N, M]],
        )
        assert_type(
            F.triplet_margin_loss(matrix, matrix, matrix, reduction="none"),
            Tensor[[N]],
        )

    def check_gradual_metric_loss_controls(
        input: Tensor[IntTuple],
        target: Tensor,
        reduction: str,
        reduce: bool | None,
    ) -> None:
        assert_type(
            F.cosine_embedding_loss(input, target, target, reduce=False),
            Tensor[IntTuple],
        )
        assert_type(
            F.cosine_embedding_loss(input, target, target, reduction=reduction),
            Tensor[IntTuple],
        )
        assert_type(
            F.triplet_margin_loss(input, target, target, reduce=reduce),
            Tensor[IntTuple],
        )
