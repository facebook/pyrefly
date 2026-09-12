# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test the named tuples in `torch.return_types` keep their shapes."""

from typing import assert_type, TYPE_CHECKING

import torch
from shape_extensions import IntTuple

if TYPE_CHECKING:
    from torch import Tensor


def test_sort() -> None:
    x: Tensor[[4, 3]] = torch.randn(4, 3)
    result = x.sort(dim=0)
    assert_type(result, torch.return_types.sort[[4, 3]])
    assert_type(result.values, Tensor[[4, 3]])
    assert_type(result.indices, Tensor[[4, 3]])
    assert_type(torch.sort(x, dim=0), torch.return_types.sort[[4, 3]])
    assert_type(torch.sort(x, dim=0).values, Tensor[[4, 3]])
    assert_type(torch.sort(x, dim=0).indices, Tensor[[4, 3]])


def test_sort_is_a_tuple() -> None:
    x: Tensor[[4, 3]] = torch.randn(4, 3)
    result = x.sort(dim=0)
    assert_type(result[0], Tensor[[4, 3]])
    assert_type(result[1], Tensor[[4, 3]])
    values, indices = result
    assert_type(values, Tensor[[4, 3]])
    assert_type(indices, Tensor[[4, 3]])


def test_cummax() -> None:
    x: Tensor[[4, 3]] = torch.randn(4, 3)
    assert_type(x.cummax(0), torch.return_types.cummax[[4, 3]])
    assert_type(x.cummax(0).values, Tensor[[4, 3]])
    assert_type(x.cummax(0).indices, Tensor[[4, 3]])
    assert_type(torch.cummax(x, 0), torch.return_types.cummax[[4, 3]])
    assert_type(torch.cummax(x, 0).values, Tensor[[4, 3]])
    assert_type(torch.cummax(x, 0).indices, Tensor[[4, 3]])


def test_cummin() -> None:
    x: Tensor[[4, 3]] = torch.randn(4, 3)
    assert_type(x.cummin(1), torch.return_types.cummin[[4, 3]])
    assert_type(x.cummin(1).values, Tensor[[4, 3]])
    assert_type(x.cummin(1).indices, Tensor[[4, 3]])
    assert_type(torch.cummin(x, 1), torch.return_types.cummin[[4, 3]])
    assert_type(torch.cummin(x, 1).values, Tensor[[4, 3]])
    assert_type(torch.cummin(x, 1).indices, Tensor[[4, 3]])


def test_max() -> None:
    x: Tensor[[4, 3]] = torch.randn(4, 3)
    assert_type(x.max(dim=1), torch.return_types.max[[4]])
    assert_type(x.max(dim=1).values, Tensor[[4]])
    assert_type(x.max(dim=1).indices, Tensor[[4]])
    assert_type(torch.max(x, dim=1), torch.return_types.max[[4]])
    assert_type(torch.max(x, dim=1).values, Tensor[[4]])
    assert_type(torch.max(x, dim=1).indices, Tensor[[4]])
    assert_type(torch.max(x, dim=0, keepdim=True), torch.return_types.max[[1, 3]])


def test_min() -> None:
    x: Tensor[[4, 3]] = torch.randn(4, 3)
    assert_type(x.min(dim=1), torch.return_types.min[[4]])
    assert_type(x.min(dim=1).values, Tensor[[4]])
    assert_type(x.min(dim=1).indices, Tensor[[4]])
    assert_type(torch.min(x, dim=1), torch.return_types.min[[4]])
    assert_type(torch.min(x, dim=1).values, Tensor[[4]])
    assert_type(torch.min(x, dim=1).indices, Tensor[[4]])


def test_topk() -> None:
    x: Tensor[[4, 3]] = torch.randn(4, 3)
    assert_type(x.topk(2, dim=0), torch.return_types.topk[[2, 3]])
    assert_type(x.topk(2, dim=0).values, Tensor[[2, 3]])
    assert_type(x.topk(2, dim=0).indices, Tensor[[2, 3]])
    assert_type(torch.topk(x, 2, dim=0), torch.return_types.topk[[2, 3]])
    assert_type(torch.topk(x, 2, dim=0).values, Tensor[[2, 3]])
    assert_type(torch.topk(x, 2, dim=0).indices, Tensor[[2, 3]])


def test_kthvalue() -> None:
    x: Tensor[[4, 3]] = torch.randn(4, 3)
    assert_type(x.kthvalue(2, dim=1), torch.return_types.kthvalue[[4]])
    assert_type(x.kthvalue(2, dim=1).values, Tensor[[4]])
    assert_type(x.kthvalue(2, dim=1).indices, Tensor[[4]])
    assert_type(torch.kthvalue(x, 2, dim=1), torch.return_types.kthvalue[[4]])
    assert_type(torch.kthvalue(x, 2, dim=1).values, Tensor[[4]])
    assert_type(torch.kthvalue(x, 2, dim=1).indices, Tensor[[4]])


def test_median() -> None:
    x: Tensor[[4, 3]] = torch.randn(4, 3)
    assert_type(x.median(dim=0), torch.return_types.median[[3]])
    assert_type(x.median(dim=0).values, Tensor[[3]])
    assert_type(x.median(dim=0).indices, Tensor[[3]])
    assert_type(torch.median(x, dim=0), torch.return_types.median[[3]])
    assert_type(torch.median(x, dim=0).values, Tensor[[3]])
    assert_type(torch.median(x, dim=0).indices, Tensor[[3]])


def test_mode() -> None:
    x: Tensor[[4, 3]] = torch.randn(4, 3)
    assert_type(x.mode(dim=0), torch.return_types.mode[[3]])
    assert_type(x.mode(dim=0).values, Tensor[[3]])
    assert_type(x.mode(dim=0).indices, Tensor[[3]])
    assert_type(torch.mode(x, dim=0), torch.return_types.mode[[3]])
    assert_type(torch.mode(x, dim=0).values, Tensor[[3]])
    assert_type(torch.mode(x, dim=0).indices, Tensor[[3]])


def test_aminmax() -> None:
    x: Tensor[[4, 3]] = torch.randn(4, 3)
    assert_type(x.aminmax(), torch.return_types.aminmax[[]])
    assert_type(x.aminmax().min, Tensor[[]])
    assert_type(x.aminmax().max, Tensor[[]])
    assert_type(torch.aminmax(x, dim=0), torch.return_types.aminmax[[3]])
    assert_type(torch.aminmax(x, dim=0).min, Tensor[[3]])
    assert_type(torch.aminmax(x, dim=0).max, Tensor[[3]])


def test_slogdet() -> None:
    m: Tensor[[2, 3, 3]] = torch.randn(2, 3, 3)
    assert_type(m.slogdet(), torch.return_types.slogdet[[2]])
    assert_type(m.slogdet().sign, Tensor[[2]])
    assert_type(m.slogdet().logabsdet, Tensor[[2]])
    assert_type(torch.slogdet(m), torch.return_types.slogdet[[2]])
    assert_type(torch.slogdet(m).sign, Tensor[[2]])
    assert_type(torch.slogdet(m).logabsdet, Tensor[[2]])


def test_linalg_slogdet() -> None:
    m: Tensor[[2, 3, 3]] = torch.randn(2, 3, 3)
    assert_type(torch.linalg.slogdet(m), torch.return_types.linalg_slogdet[[2]])
    assert_type(torch.linalg.slogdet(m).sign, Tensor[[2]])
    assert_type(torch.linalg.slogdet(m).logabsdet, Tensor[[2]])


def test_slogdet_shapeless_input(m: Tensor) -> None:
    assert_type(m.slogdet(), torch.return_types.slogdet[IntTuple])
    assert_type(torch.slogdet(m).sign, Tensor[IntTuple])
    assert_type(torch.linalg.slogdet(m), torch.return_types.linalg_slogdet[IntTuple])
