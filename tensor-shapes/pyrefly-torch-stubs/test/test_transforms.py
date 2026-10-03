# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type, TYPE_CHECKING

import torch
from shape_extensions import assert_shape, IntVar
from torch import Tensor
from torch.distributions.transforms import (
    AbsTransform,
    ExpTransform,
    LowerCholeskyTransform,
    PositiveDefiniteTransform,
    SigmoidTransform,
    SoftmaxTransform,
    SoftplusTransform,
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
