# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type, TYPE_CHECKING

import torch
from shape_extensions import assert_raises, assert_shape, Int, IntTuple, IntVar
from torch import Tensor


def test_median_shapes() -> None:
    matrix = torch.randn((3, 4))
    assert_shape(torch.median(matrix).shape, ())

    result = torch.median(matrix, dim=0)
    assert_shape(result.values.shape, (4,))
    assert_shape(result.indices.shape, (4,))

    tensor = torch.randn((2, 3, 4))
    values, indices = tensor.median(dim=-2, keepdim=True)
    assert_shape(values.shape, (2, 1, 4))
    assert_shape(indices.shape, (2, 1, 4))


def test_logsumexp_shapes() -> None:
    tensor = torch.randn((2, 3, 4))
    assert_shape(torch.logsumexp(tensor, dim=(0, 2)).shape, (3,))
    assert_shape(tensor.logsumexp(dim=-1).shape, (2, 3))
    assert_shape(torch.logsumexp(tensor, dim=1, keepdim=True).shape, (2, 1, 4))


def test_count_nonzero_shapes() -> None:
    tensor = torch.randn((2, 3, 4))
    assert_shape(torch.count_nonzero(tensor).shape, ())
    assert_shape(tensor.count_nonzero(dim=0).shape, (3, 4))
    assert_shape(torch.count_nonzero(tensor, dim=(0, -1)).shape, (3,))
    assert_shape(tensor.count_nonzero(dim=()).shape, ())


def test_aminmax_shapes() -> None:
    matrix = torch.randn((3, 4))
    result = torch.aminmax(matrix)
    assert_shape(result.min.shape, ())
    assert_shape(result.max.shape, ())

    tensor = torch.randn((2, 3, 4))
    minimum, maximum = tensor.aminmax(dim=-2, keepdim=True)
    assert_shape(minimum.shape, (2, 1, 4))
    assert_shape(maximum.shape, (2, 1, 4))


def test_reductions_reject_invalid_dimensions() -> None:
    tensor = torch.randn((2, 3, 4))
    assert_shape(torch.sum(tensor, dim=1).shape, (2, 4))

    with assert_raises(IndexError):
        torch.sum(tensor, dim=999)  # E: dimension out of range

    with assert_raises(RuntimeError):
        tensor.std(dim=(0, -3))  # E: duplicate dimension

    with assert_raises(RuntimeError):
        torch.sum(torch.ones(()), dim=(0, -1))  # E: duplicate dimension


if TYPE_CHECKING:

    def check_symbolic_reductions[N: IntVar](
        tensor: Tensor[[2, N, 4]], dim: int, keepdim: bool
    ) -> None:
        median = torch.median(tensor, dim=1)
        assert_type(median, torch.return_types.median[[2, 4]])
        assert_type(median.values, Tensor[[2, 4]])
        assert_type(median.indices, Tensor[[2, 4]])
        assert_type(torch.logsumexp(tensor, dim=(0, 2)), Tensor[[N]])
        assert_type(torch.count_nonzero(tensor, dim=-1), Tensor[[2, N]])
        extrema = torch.aminmax(tensor, dim=1)
        assert_type(extrema, torch.return_types.aminmax[[2, 4]])
        assert_type(extrema.min, Tensor[[2, 4]])
        assert_type(extrema.max, Tensor[[2, 4]])

        assert_type(torch.logsumexp(tensor, dim=dim), Tensor[IntTuple])
        assert_type(torch.median(tensor, dim=dim)[0], Tensor[IntTuple])
        assert_type(torch.aminmax(tensor, dim=dim)[0], Tensor[IntTuple])
        assert_type(torch.logsumexp(tensor, dim=1, keepdim=keepdim), Tensor[IntTuple])

        tensor.aminmax().values  # E: no attribute `values`

    def check_symbolic_dimension[N: IntVar](
        tensor: Tensor[[2, N, 4]], dim: Int[N]
    ) -> None:
        assert_type(torch.count_nonzero(tensor, dim=dim), Tensor[IntTuple])
