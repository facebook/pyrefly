# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type, TYPE_CHECKING

import torch
from shape_extensions import assert_raises, assert_shape, IntTuple
from torch import Tensor


def test_mode_shapes() -> None:
    matrix = torch.randn((4, 5))
    result = torch.mode(matrix, dim=1)
    assert_shape(result.values.shape, (4,))
    assert_shape(result.indices.shape, (4,))

    tensor = torch.randn((2, 3, 4))
    values, indices = tensor.mode(dim=-2, keepdim=True)
    assert_shape(values.shape, (2, 1, 4))
    assert_shape(indices.shape, (2, 1, 4))


def test_sort_shapes() -> None:
    vector = torch.randn(5)
    result = torch.sort(vector)
    assert_shape(result.values.shape, (5,))
    assert_shape(result.indices.shape, (5,))

    tensor = torch.randn((2, 5, 3))
    values, indices = tensor.sort(dim=1, descending=True, stable=True)
    assert_shape(values.shape, (2, 5, 3))
    assert_shape(indices.shape, (2, 5, 3))


def test_kthvalue_shapes() -> None:
    vector = torch.randn(10)
    result = torch.kthvalue(vector, k=3)
    assert_shape(result.values.shape, ())
    assert_shape(result.indices.shape, ())

    matrix = torch.randn((4, 5))
    values, indices = matrix.kthvalue(k=2, dim=1, keepdim=True)
    assert_shape(values.shape, (4, 1))
    assert_shape(indices.shape, (4, 1))

    scalar = torch.randn(())
    values, indices = scalar.kthvalue(k=1)
    assert_shape(values.shape, ())
    assert_shape(indices.shape, ())


def test_ordering_rejects_invalid_arguments() -> None:
    matrix = torch.randn((2, 3))
    values, _ = torch.mode(matrix, dim=1)
    assert_shape(values.shape, (2,))

    with assert_raises(IndexError):
        torch.mode(matrix, dim=2)  # E: dimension out of range

    with assert_raises(IndexError):
        # TODO: BUG: Reject out-of-range sort dimensions statically.
        matrix.sort(dim=-3)

    with assert_raises(RuntimeError):
        # TODO: BUG: Reject non-positive kthvalue indices statically.
        torch.kthvalue(matrix, k=0, dim=1)

    with assert_raises(RuntimeError):
        # TODO: BUG: Reject kthvalue indices larger than the dimension statically.
        matrix.kthvalue(k=4, dim=1)


if TYPE_CHECKING:

    def check_ordering_shapes[Shape: IntTuple](tensor: Tensor[Shape], dim: int) -> None:
        assert_type(torch.sort(tensor, dim), torch.return_types.sort[Shape])
        assert_type(tensor.sort(dim).values, Tensor[Shape])
        assert_type(torch.mode(tensor, dim=dim), torch.return_types.mode[IntTuple])
        assert_type(torch.mode(tensor, dim=dim).indices, Tensor[IntTuple])
        assert_type(tensor.kthvalue(1, dim=dim), torch.return_types.kthvalue[IntTuple])
        assert_type(tensor.kthvalue(1, dim=dim).values, Tensor[IntTuple])

        tensor.sort(dim).value  # E: no attribute `value`
