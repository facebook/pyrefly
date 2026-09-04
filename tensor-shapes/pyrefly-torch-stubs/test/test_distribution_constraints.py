# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type, TYPE_CHECKING

import torch
from shape_extensions import assert_raises, assert_shape, IntTuple, IntVar
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


def test_interval_factories_broadcast_bounds() -> None:
    value = torch.tensor([[0.0, 1.0, 2.0]])
    lower = torch.tensor([[0.0], [1.0]])
    upper = torch.tensor([[1.0], [2.0]])

    assert_shape(constraints.integer_interval(lower, upper).check(value).shape, (2, 3))
    assert constraints.integer_interval(lower, upper).check(value).tolist() == [
        [True, True, False],
        [False, True, True],
    ]
    assert_shape(
        constraints.half_open_interval(lower, upper).check(value).shape, (2, 3)
    )
    assert constraints.half_open_interval(lower, upper).check(value).tolist() == [
        [True, False, False],
        [False, True, False],
    ]
    assert_shape(constraints.integer_interval(0, 2).check(value).shape, (1, 3))
    assert_shape(constraints.half_open_interval(0.0, 2.0).check(value).shape, (1, 3))


def test_multinomial_constraint_bounds() -> None:
    counts = torch.tensor([[2.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
    bound = torch.tensor([2.0, 3.0])

    assert_shape(constraints.multinomial(bound).check(counts).shape, (2,))
    assert constraints.multinomial(bound).check(counts).tolist() == [False, True]
    assert_shape(constraints.multinomial(3).check(counts).shape, (2,))
    assert constraints.multinomial(3).check(counts).tolist() == [True, True]


def test_concatenated_constraint() -> None:
    value = torch.tensor([[0.0, 1.0, -1.0], [1.0, 2.0, 3.0]])
    constraint = constraints.cat(
        [constraints.boolean, constraints.positive], dim=1, lengths=[1, 2]
    )

    assert_shape(constraint.check(value).shape, IntTuple, runtime=(2, 3))
    assert constraint.check(value).tolist() == [
        [True, True, False],
        [True, True, True],
    ]
    assert constraint.lengths == [1, 2]


def test_stacked_constraint() -> None:
    value = torch.tensor(
        [
            [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]],
            [[2.0, 3.0, 4.0], [5.0, 0.0, -1.0]],
        ]
    )
    constraint = constraints.stack([constraints.boolean, constraints.positive])

    assert_shape(constraint.check(value).shape, IntTuple, runtime=(2, 2, 3))
    assert constraint.check(value).tolist() == [
        [[True, True, True], [True, True, True]],
        [[True, True, True], [True, False, False]],
    ]
    assert constraint.dim == 0


def test_independent_constraint_event_dimensions() -> None:
    value = torch.ones((2, 3, 4))
    value[1, 0, 0] = -1
    constraint = constraints.independent(constraints.positive, 2)

    assert_shape(constraint.check(value).shape, IntTuple, runtime=(2,))
    assert constraint.check(value).tolist() == [True, False]
    assert constraint.reinterpreted_batch_ndims == 2


def test_real_vector_reduces_last_dimension() -> None:
    value = torch.tensor([[1.0, float("nan")], [2.0, 3.0]])

    assert_shape(constraints.real_vector.check(value).shape, (2,))
    assert constraints.real_vector.check(value).tolist() == [False, True]


def test_mixture_constraint_checks_every_component() -> None:
    component = constraints.greater_than(torch.tensor([0.5, 1.5]))
    mixture = constraints.MixtureSameFamilyConstraint(component)
    value = torch.tensor([[1.0, 2.0], [0.0, 3.0]])

    assert_shape(mixture.check(value).shape, IntTuple, runtime=(2, 2))
    assert mixture.check(value).tolist() == [[False, True], [False, True]]
    assert mixture.base_constraint is component


def test_dependent_constraint() -> None:
    dependent = constraints.dependent(is_discrete=True, event_dim=1)

    assert dependent.is_discrete
    assert dependent.event_dim == 1
    assert constraints.is_dependent(dependent)
    assert not constraints.is_dependent(constraints.boolean)
    with assert_raises(ValueError):
        dependent.check(torch.ones(3))
    with assert_raises(NotImplementedError):
        _ = constraints.dependent.event_dim


def test_dependent_property() -> None:
    class WithSupport:
        @constraints.dependent_property(is_discrete=False, event_dim=0)
        def support(self) -> constraints.Constraint:
            return constraints.positive

        @constraints.dependent_property
        def unspecified_support(self) -> constraints.Constraint:
            return constraints.boolean

    value = WithSupport()
    assert constraints.is_dependent(WithSupport.support)
    assert WithSupport.support.event_dim == 0
    assert value.support is constraints.positive
    assert value.unspecified_support is constraints.boolean
    with assert_raises(NotImplementedError):
        _ = WithSupport.unspecified_support.event_dim


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
        assert_type(constraints.real_vector.check(vector), Tensor[S])
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

    def check_bounded_constraint_shapes[B: IntVar, N: IntVar](
        values: Tensor[[1, N]],
        lower: Tensor[[B, 1]],
        upper: Tensor[[1, N]],
        counts: Tensor[[B, N]],
        count_bound: Tensor[[B]],
    ) -> None:
        assert_type(
            constraints.integer_interval(lower, upper).check(values),
            Tensor[[B, N]],
        )
        assert_type(
            constraints.half_open_interval(lower, upper).check(values),
            Tensor[[B, N]],
        )
        assert_type(constraints.multinomial(count_bound).check(counts), Tensor[[B]])
        assert_type(constraints.multinomial(3).check(counts), Tensor[[B]])

    def check_composed_constraints(value: Tensor[[2, 3]]) -> None:
        concat = constraints.cat(
            [constraints.boolean, constraints.positive], lengths=[1, 2]
        )
        stacked = constraints.stack((constraints.boolean, constraints.positive))

        assert_type(concat.check(value), Tensor[IntTuple])
        assert_type(stacked.check(value), Tensor[IntTuple])
        assert_type(concat.cseq, list[constraints.Constraint])
        assert_type(stacked.cseq, list[constraints.Constraint])

    def check_independent_constraints(value: Tensor[[2, 3, 4]]) -> None:
        independent = constraints.independent(constraints.positive, 2)
        mixture = constraints.MixtureSameFamilyConstraint(constraints.positive)

        assert_type(independent.check(value), Tensor[IntTuple])
        assert_type(mixture.check(value), Tensor[IntTuple])

    def check_dependent_metadata() -> None:
        dependent = constraints.dependent(is_discrete=True, event_dim=1)
        assert_type(dependent.is_discrete, bool)
        assert_type(dependent.event_dim, int)
