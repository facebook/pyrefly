# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type, TYPE_CHECKING

import torch
from shape_extensions import assert_shape, IntTuple, IntVar
from torch import Tensor
from torch.distributions import Normal
from torch.distributions.transforms import (
    AbsTransform,
    AffineTransform,
    CatTransform,
    ComposeTransform,
    CorrCholeskyTransform,
    CumulativeDistributionTransform,
    ExpTransform,
    identity_transform,
    IndependentTransform,
    LowerCholeskyTransform,
    PositiveDefiniteTransform,
    PowerTransform,
    ReshapeTransform,
    SigmoidTransform,
    SoftmaxTransform,
    SoftplusTransform,
    StackTransform,
    StickBreakingTransform,
    TanhTransform,
)


def test_elementwise_transform_shapes() -> None:
    x = torch.randn(2, 3)

    assert_shape(AbsTransform()(x).shape, (2, 3))
    for transform in (
        ExpTransform(),
        SigmoidTransform(),
        SoftplusTransform(),
        TanhTransform(),
    ):
        y = transform(x)
        assert_shape(y.shape, (2, 3))
        assert_shape(transform.log_abs_det_jacobian(x, y).shape, (2, 3))


def test_vector_and_matrix_transform_shapes() -> None:
    vector = torch.randn((2, 3))
    assert_shape(SoftmaxTransform()(vector).shape, (2, 3))

    simplex = StickBreakingTransform()(vector)
    assert_shape(simplex.shape, (2, 4))
    assert_shape(
        StickBreakingTransform().log_abs_det_jacobian(vector, simplex).shape, (2,)
    )

    matrix = torch.eye(2).expand(3, 2, 2)
    assert_shape(LowerCholeskyTransform()(matrix).shape, (3, 2, 2))
    assert_shape(PositiveDefiniteTransform()(matrix).shape, (3, 2, 2))


def test_broadcast_transform_shapes() -> None:
    value = torch.ones((1, 3))
    assert_shape(
        AffineTransform(torch.zeros((2, 1)), torch.ones((1, 3)))(value).shape, (2, 3)
    )
    assert_shape(PowerTransform(torch.ones((2, 1)))(value).shape, (2, 3))
    normal = Normal(torch.zeros((2, 1)), torch.ones((2, 1)))
    assert_shape(CumulativeDistributionTransform(normal)(value).shape, (2, 3))


def test_composed_transform_shapes() -> None:
    value = torch.randn((2, 3))
    assert_shape(
        ComposeTransform([StickBreakingTransform(), SoftmaxTransform()])(value).shape,
        IntTuple,
        runtime=(2, 4),
    )
    assert_shape(
        CatTransform([ExpTransform(), SoftplusTransform()], dim=-1, lengths=[1, 2])(
            value
        ).shape,
        IntTuple,
        runtime=(2, 3),
    )
    assert_shape(
        StackTransform([ExpTransform(), SigmoidTransform()])(value).shape,
        IntTuple,
        runtime=(2, 3),
    )
    assert_shape(
        IndependentTransform(StickBreakingTransform(), 1)(value).shape,
        IntTuple,
        runtime=(2, 4),
    )
    assert_shape(identity_transform(value).shape, (2, 3))


def test_rank_changing_transform_shapes() -> None:
    matrix = torch.ones((4, 2, 3))
    assert_shape(ReshapeTransform((2, 3), (6,))(matrix).shape, (4, 6))
    assert_shape(ReshapeTransform((2, 3), (3, 2))(matrix).shape, (4, 3, 2))
    assert_shape(
        CorrCholeskyTransform()(torch.randn((2, 3))).shape,
        (2, int, int),
        runtime=(2, 3, 3),
    )


if TYPE_CHECKING:

    def check_symbolic_elementwise_transform_shapes[N: IntVar, M: IntVar](
        x: Tensor[[N, M]],
    ) -> None:
        assert_type(AbsTransform()(x), Tensor[[N, M]])
        for transform in (
            ExpTransform(),
            SigmoidTransform(),
            SoftplusTransform(),
            TanhTransform(),
        ):
            y = transform(x)
            assert_type(y, Tensor[[N, M]])
            assert_type(transform.log_abs_det_jacobian(x, y), Tensor[[N, M]])

    def check_vector_and_matrix_transforms[B: IntVar, N: IntVar](
        vector: Tensor[[B, N]], matrix: Tensor[[B, N, N]]
    ) -> None:
        assert_type(SoftmaxTransform()(vector), Tensor[[B, N]])
        simplex = StickBreakingTransform()(vector)
        assert_type(simplex, Tensor[[B, N + 1]])
        assert_type(
            StickBreakingTransform().log_abs_det_jacobian(vector, simplex), Tensor[[B]]
        )
        assert_type(LowerCholeskyTransform()(matrix), Tensor[[B, N, N]])
        assert_type(PositiveDefiniteTransform()(matrix), Tensor[[B, N, N]])

    def check_broadcast_transforms[B: IntVar, N: IntVar](
        value: Tensor[[1, N]], bound: Tensor[[B, 1]]
    ) -> None:
        assert_type(AffineTransform(bound, 1.0)(value), Tensor[[B, N]])
        assert_type(PowerTransform(bound)(value), Tensor[[B, N]])
        normal = Normal(bound, bound)
        assert_type(CumulativeDistributionTransform(normal)(value), Tensor[[B, N]])

    def check_composed_transforms[B: IntVar, N: IntVar](value: Tensor[[B, N]]) -> None:
        assert_type(identity_transform(value), Tensor[[B, N]])
        assert_type(
            ComposeTransform([StickBreakingTransform()])(value), Tensor[IntTuple]
        )
        assert_type(
            IndependentTransform(StickBreakingTransform(), 1)(value), Tensor[IntTuple]
        )

    def check_rank_changing_transforms[B: IntVar](
        matrix: Tensor[[B, 2, 3]], vector: Tensor[[B, 3]]
    ) -> None:
        assert_type(ReshapeTransform((2, 3), (6,))(matrix), Tensor[[B, 6]])
        assert_type(CorrCholeskyTransform()(vector), Tensor[[B, int, int]])
