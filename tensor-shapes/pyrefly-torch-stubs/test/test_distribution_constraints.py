# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type, TYPE_CHECKING

import torch
from shape_extensions import assert_shape, IntTuple, IntVar
from torch import Tensor
from torch.distributions import constraints


def test_elementwise_constraint() -> None:
    value = torch.tensor([[0.0, 1.0, 2.0], [1.0, 0.0, 1.0]])

    assert_shape(constraints.boolean.check(value).shape, (2, 3))
    assert constraints.boolean.check(value).tolist() == [
        [True, True, False],
        [True, True, True],
    ]


def test_elementwise_bounds() -> None:
    value = torch.tensor([[-1.0, 0.0, 0.5, 1.0, 2.0]])

    assert_shape(constraints.positive.check(value).shape, (1, 5))
    assert constraints.positive.check(value).tolist() == [
        [False, False, True, True, True]
    ]
    assert_shape(constraints.nonnegative.check(value).shape, (1, 5))
    assert constraints.nonnegative.check(value).tolist() == [
        [False, True, True, True, True]
    ]
    assert_shape(constraints.positive_integer.check(value).shape, (1, 5))
    assert constraints.positive_integer.check(value).tolist() == [
        [False, False, False, True, True]
    ]
    assert_shape(constraints.nonnegative_integer.check(value).shape, (1, 5))
    assert constraints.nonnegative_integer.check(value).tolist() == [
        [False, True, False, True, True]
    ]
    assert_shape(constraints.unit_interval.check(value).shape, (1, 5))
    assert constraints.unit_interval.check(value).tolist() == [
        [False, True, True, True, False]
    ]


def test_threshold_factories_broadcast_bounds() -> None:
    value = torch.tensor([[0.0, 1.0, 2.0]])
    lower_bound = torch.tensor([[0.5], [1.5]])

    assert_shape(constraints.greater_than(lower_bound).check(value).shape, (2, 3))
    assert constraints.greater_than(lower_bound).check(value).tolist() == [
        [False, True, True],
        [False, False, True],
    ]
    assert_shape(constraints.greater_than_eq(lower_bound).check(value).shape, (2, 3))
    assert constraints.greater_than_eq(lower_bound).check(value).tolist() == [
        [False, True, True],
        [False, False, True],
    ]
    assert_shape(constraints.less_than(lower_bound).check(value).shape, (2, 3))
    assert constraints.less_than(lower_bound).check(value).tolist() == [
        [True, False, False],
        [True, True, False],
    ]
    assert_shape(constraints.greater_than_eq(1.0).check(value).shape, (1, 3))
    assert constraints.greater_than_eq(1.0).check(value).tolist() == [
        [False, True, True]
    ]


def test_vector_constraints() -> None:
    value = torch.tensor([[1.0, 0.0, 0.0], [0.2, 0.3, 0.5]])

    assert_shape(constraints.one_hot.check(value).shape, (2,))
    assert constraints.one_hot.check(value).tolist() == [True, False]
    assert_shape(constraints.simplex.check(value).shape, (2,))
    assert constraints.simplex.check(value).tolist() == [True, True]


def test_matrix_constraints() -> None:
    square = torch.tensor([[[1.0, 2.0], [2.0, 3.0]]])
    rectangular = torch.ones((3, 2))

    assert_shape(constraints.square.check(square).shape, (1,))
    assert constraints.square.check(square).tolist() == [True]
    assert_shape(constraints.symmetric.check(square).shape, (1,))
    assert constraints.symmetric.check(square).tolist() == [True]
    assert_shape(constraints.square.check(rectangular).shape, ())
    assert not constraints.square.check(rectangular).item()
    assert not constraints.symmetric.check(rectangular).item()


def test_matrix_constraint_specializations() -> None:
    matrices = torch.stack((torch.eye(2), torch.zeros((2, 2))))

    assert_shape(constraints.lower_triangular.check(matrices).shape, (2,))
    assert constraints.lower_triangular.check(matrices).tolist() == [True, True]
    assert_shape(constraints.lower_cholesky.check(matrices).shape, (2,))
    assert constraints.lower_cholesky.check(matrices).tolist() == [True, False]
    assert_shape(constraints.corr_cholesky.check(matrices).shape, (2,))
    assert constraints.corr_cholesky.check(matrices).tolist() == [True, False]
    assert_shape(constraints.positive_semidefinite.check(matrices).shape, (2,))
    assert constraints.positive_semidefinite.check(matrices).tolist() == [True, True]
    assert_shape(constraints.positive_definite.check(matrices).shape, (2,))
    assert constraints.positive_definite.check(matrices).tolist() == [True, False]


if TYPE_CHECKING:

    def check_constraint_shapes[S: IntTuple, M: IntVar, N: IntVar](
        elementwise: Tensor[S],
        vector: Tensor[[*S, N]],
        matrix: Tensor[[*S, M, N]],
    ) -> None:
        assert_type(constraints.boolean.check(elementwise), Tensor[S])
        assert_type(constraints.positive.check(elementwise), Tensor[S])
        assert_type(constraints.nonnegative.check(elementwise), Tensor[S])
        assert_type(constraints.positive_integer.check(elementwise), Tensor[S])
        assert_type(constraints.nonnegative_integer.check(elementwise), Tensor[S])
        assert_type(constraints.unit_interval.check(elementwise), Tensor[S])
        assert_type(constraints.one_hot.check(vector), Tensor[S])
        assert_type(constraints.simplex.check(vector), Tensor[S])
        assert_type(constraints.square.check(matrix), Tensor[S])
        assert_type(constraints.symmetric.check(matrix), Tensor[S])
        assert_type(constraints.lower_triangular.check(matrix), Tensor[S])
        assert_type(constraints.lower_cholesky.check(matrix), Tensor[S])
        assert_type(constraints.corr_cholesky.check(matrix), Tensor[S])
        assert_type(constraints.positive_semidefinite.check(matrix), Tensor[S])
        assert_type(constraints.positive_definite.check(matrix), Tensor[S])

    def check_threshold_shapes[B: IntVar, N: IntVar](
        value: Tensor[[1, N]], bound: Tensor[[B, 1]]
    ) -> None:
        assert_type(constraints.greater_than(bound).check(value), Tensor[[B, N]])
        assert_type(constraints.greater_than_eq(bound).check(value), Tensor[[B, N]])
        assert_type(constraints.less_than(bound).check(value), Tensor[[B, N]])
        assert_type(constraints.greater_than(1.0).check(value), Tensor[[1, N]])
