# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import assert_type

import torch
from shape_extensions import assert_shape, IntTuple
from torch import Tensor


def test_unique_default() -> None:
    result = torch.unique(torch.tensor([1, 1, 2]))
    assert_type(result, Tensor[[int]])
    assert_shape(result.shape, (int,), runtime=(2,))


def test_unique_return_inverse() -> None:
    values, inverse = torch.unique(torch.tensor([[1, 1], [2, 1]]), return_inverse=True)
    assert_type(values, Tensor[[int]])
    assert_type(inverse, Tensor[[2, 2]])
    assert_shape(inverse.shape, (2, 2))


def test_unique_return_counts() -> None:
    values, counts = torch.unique(torch.tensor([1, 1, 2]), return_counts=True)
    assert_type(values, Tensor[[int]])
    assert_type(counts, Tensor[[int]])
    assert_shape(counts.shape, (int,), runtime=(2,))


def test_unique_both_flags() -> None:
    values, inverse, counts = torch.unique(
        torch.tensor([1, 1, 2]), return_inverse=True, return_counts=True
    )
    assert_type(values, Tensor[[int]])
    assert_type(inverse, Tensor[[3]])
    assert_type(counts, Tensor[[int]])
    assert_shape(values.shape, (int,), runtime=(2,))


def test_unique_dim() -> None:
    input = torch.tensor([[1, 1], [1, 1], [2, 1]])
    values = torch.unique(input, dim=0)
    assert_type(values, Tensor)
    assert_shape(values.shape, IntTuple, runtime=(2, 2))

    values, inverse, counts = torch.unique(
        input, return_inverse=True, return_counts=True, dim=0
    )
    assert_type(values, Tensor)
    assert_type(inverse, Tensor[[int]])
    assert_type(counts, Tensor[[int]])
    assert_shape(values.shape, IntTuple, runtime=(2, 2))
    assert_shape(inverse.shape, (int,), runtime=(3,))


def test_unique_dynamic_flags() -> None:
    return_inverse: bool = bool(torch.tensor(1))
    return_counts: bool = bool(torch.tensor(0))
    result = torch.unique(
        torch.tensor([1, 1, 2]),
        return_inverse=return_inverse,
        return_counts=return_counts,
    )
    assert_type(
        result,
        Tensor | tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor],
    )
    values = result[0] if isinstance(result, tuple) else result
    assert_shape(values.shape, IntTuple, runtime=(2,))
