# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type, TYPE_CHECKING

import torch
from shape_extensions import assert_shape, Elements, IntTuple, IntVar
from torch import Tensor


def test_var_mean_shapes() -> None:
    matrix = torch.randn((3, 4))
    variance, mean = torch.var_mean(matrix)
    assert_shape(variance.shape, ())
    assert_shape(mean.shape, ())

    tensor = torch.randn((2, 3, 4))
    variance, mean = torch.var_mean(tensor, dim=(0, -1))
    assert_shape(variance.shape, (3,))
    assert_shape(mean.shape, (3,))

    variance, mean = torch.var_mean(tensor, dim=1, keepdim=True)
    assert_shape(variance.shape, (2, 1, 4))
    assert_shape(mean.shape, (2, 1, 4))


def test_std_mean_shapes() -> None:
    matrix = torch.randn((3, 4))
    deviation, mean = torch.std_mean(matrix)
    assert_shape(deviation.shape, ())
    assert_shape(mean.shape, ())

    tensor = torch.randn((2, 3, 4))
    deviation, mean = torch.std_mean(tensor, dim=-1, keepdim=True)
    assert_shape(deviation.shape, (2, 3, 1))
    assert_shape(mean.shape, (2, 3, 1))


def test_statistical_reduction_positional_arguments() -> None:
    matrix = torch.randn((2, 3))

    variance, variance_mean = torch.var_mean(matrix, False)
    deviation, deviation_mean = torch.std_mean(matrix, False)
    for result in (variance, variance_mean, deviation, deviation_mean):
        assert_shape(result.shape, ())

    variance, variance_mean = torch.var_mean(matrix, 1)
    deviation, deviation_mean = torch.std_mean(matrix, 1)
    for result in (variance, variance_mean, deviation, deviation_mean):
        assert_shape(result.shape, (2,))


def test_reduction_axis_forms() -> None:
    tensor = torch.randn((2, 3, 4))
    assert_shape(torch.sum(tensor, dim=(0, -1)).shape, (3,))
    assert_shape(tensor.mean(dim=(-1, -3), keepdim=True).shape, (1, 3, 1))
    assert_shape(torch.sum(tensor, dim=()).shape, ())
    assert_shape(tensor.sum(dim=(), keepdim=True).shape, (1, 1, 1))
    assert_shape(torch.sum(tensor, dim=None, keepdim=True).shape, (1, 1, 1))

    scalar = torch.tensor(1)
    assert_shape(torch.sum(scalar, dim=0).shape, ())
    assert_shape(scalar.sum(dim=-1, keepdim=True).shape, ())


def test_boolean_reduction_shapes() -> None:
    tensor = torch.ones((2, 3, 4), dtype=torch.bool)
    assert_shape(torch.all(input=tensor, dim=(0, 2)).shape, (3,))
    assert_shape(torch.any(input=tensor, dim=(0, -1), keepdim=True).shape, (1, 3, 1))


def test_min_max_shapes() -> None:
    tensor = torch.randn((2, 3, 4))
    maximum = torch.max(input=tensor, dim=1)
    assert_shape(maximum.values.shape, (2, 4))
    assert_shape(maximum.indices.shape, (2, 4))
    assert_shape(torch.max(input=tensor, other=tensor).shape, (2, 3, 4))

    left = torch.randn((2, 1))
    right = torch.randn(3)
    assert_shape(torch.max(left, right).shape, (2, 3))
    assert_shape(torch.min(left, right).shape, (2, 3))


if TYPE_CHECKING:

    def check_min_max_named_returns[N: IntVar](tensor: Tensor[[2, N, 4]]) -> None:
        maximum = torch.max(tensor, dim=1)
        assert_type(maximum, torch.return_types.max[[2, 4]])
        assert_type(maximum.values, Tensor[[2, 4]])
        assert_type(maximum.indices, Tensor[[2, 4]])

        minimum = tensor.min(dim=-1, keepdim=True)
        assert_type(minimum, torch.return_types.min[[2, N, 1]])
        assert_type(minimum.values, Tensor[[2, N, 1]])
        assert_type(minimum.indices, Tensor[[2, N, 1]])

    def check_unknown_rank_reduction(tensor: Tensor) -> None:
        assert_type(torch.sum(input=tensor, dim=0), Tensor[IntTuple])
        assert_type(torch.sum(input=tensor, dim=-1), Tensor[IntTuple])

    def check_symbolic_rank_reduction[Batch: IntTuple, N: IntVar](
        tensor: Tensor[[*Elements[Batch], N]], keepdim: bool
    ) -> None:
        assert_type(torch.sum(tensor, dim=-1), Tensor[Batch])
        assert_type(tensor.mean(dim=-1, keepdim=True), Tensor[[*Elements[Batch], 1]])
        assert_type(tensor.std(dim=-1), Tensor[Batch])
        assert_type(tensor.mean(dim=-1, keepdim=keepdim), Tensor[IntTuple])

    def check_gradual_reduction_controls[N: IntVar](
        tensor: Tensor[[N, 3, 4]],
        dim: int,
        dims: tuple[int, ...],
        keepdim: bool,
    ) -> None:
        assert_type(torch.sum(tensor, dim=-1), Tensor[[N, 3]])
        assert_type(torch.sum(tensor, dim=dim), Tensor[IntTuple])
        assert_type(tensor.std(dim=dims), Tensor[IntTuple])
        assert_type(torch.mean(tensor, dim=1, keepdim=keepdim), Tensor[IntTuple])
