# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Phase 1.3: Tensor creation operations tests
from typing import Any, assert_type, TYPE_CHECKING

import torch
from shape_extensions import IntTuple
from torch import Tensor


def test_tensor_data_constructors() -> None:
    assert_type(torch.tensor(1), Tensor[[]])
    assert_type(torch.tensor([1, 2, 3]), Tensor[[3]])
    assert_type(torch.tensor([[1, 2], [3, 4]]), Tensor[[2, 2]])
    assert_type(torch.tensor([[], []]), Tensor[[2, 0]])
    assert_type(torch.tensor([1, 2], dtype=torch.float32), Tensor[[2]])
    assert_type(
        torch.tensor([1, 2], device="cpu", requires_grad=False, pin_memory=False),
        Tensor[[2]],
    )
    assert_type(torch.Tensor([1, 2, 3]), Tensor[[3]])
    assert_type(torch.Tensor([[1, 2], [3, 4]]), Tensor[[2, 2]])


def test_tensor_size_constructor() -> None:
    assert_type(torch.Tensor(), Tensor[[0]])
    assert_type(torch.Tensor(device="cpu"), Tensor[[0]])
    assert_type(torch.Tensor(2), Tensor[[2]])
    assert_type(torch.Tensor(2, device="cpu"), Tensor[[2]])
    assert_type(torch.Tensor(2, 3), Tensor[[2, 3]])
    assert_type(torch.Tensor([1, 2], device="cpu"), Tensor[[2]])


def check_tensor_constructor_compatibility(
    tensor: Tensor[[2, 3]], raw: list[int], dynamic: Any, size: int
) -> None:
    assert_type(torch.tensor(tensor), Tensor[[2, 3]])
    assert_type(torch.tensor(tensor, pin_memory=False), Tensor[[2, 3]])
    assert_type(torch.Tensor(tensor), Tensor[[2, 3]])
    assert_type(torch.Tensor(size), Tensor[IntTuple[int]])
    assert_type(torch.Tensor(size, size), Tensor[IntTuple[int, int]])
    assert_type(torch.tensor(raw), Tensor[IntTuple])
    assert_type(torch.tensor(dynamic), Tensor[Any])


if TYPE_CHECKING:
    assert_type(torch.tensor([[1], [2, 3]]), Tensor[IntTuple])
    assert_type(torch.Tensor([[1], [2, 3]]), Tensor[IntTuple])
    assert_type(torch.Tensor(True), Tensor[IntTuple])
    assert_type(torch.Tensor(data=True), Tensor[IntTuple])
    assert_type(torch.Tensor(1.0), Tensor[IntTuple])
    assert_type(torch.Tensor(data=2), Tensor[IntTuple])
    assert_type(torch.Tensor([1 + 2j]), Tensor[IntTuple])

# ==== Triangular Operations (preserve shape) ====


# Test: torch.tril
def test_tril():
    x: Tensor[[3, 3]] = torch.randn(3, 3)
    # Lower triangular preserves shape
    result = torch.tril(x)
    assert_type(result, Tensor[[3, 3]])


def test_tril_rectangular():
    x: Tensor[[4, 5]] = torch.randn(4, 5)
    result = torch.tril(x)
    assert_type(result, Tensor[[4, 5]])


def test_tril_with_diagonal():
    x: Tensor[[3, 3]] = torch.randn(3, 3)
    # With diagonal offset
    result = torch.tril(x, diagonal=1)
    assert_type(result, Tensor[[3, 3]])


def test_tril_method():
    x: Tensor[[4, 4]] = torch.randn(4, 4)
    result = x.tril()
    assert_type(result, Tensor[[4, 4]])


# Test: torch.triu
def test_triu():
    x: Tensor[[3, 3]] = torch.randn(3, 3)
    # Upper triangular preserves shape
    result = torch.triu(x)
    assert_type(result, Tensor[[3, 3]])


def test_triu_rectangular():
    x: Tensor[[5, 4]] = torch.randn(5, 4)
    result = torch.triu(x)
    assert_type(result, Tensor[[5, 4]])


def test_triu_with_diagonal():
    x: Tensor[[3, 3]] = torch.randn(3, 3)
    result = torch.triu(x, diagonal=-1)
    assert_type(result, Tensor[[3, 3]])


def test_triu_method():
    x: Tensor[[4, 4]] = torch.randn(4, 4)
    result = x.triu()
    assert_type(result, Tensor[[4, 4]])


# ==== Triangular Indices ====


# Test: torch.tril_indices
def test_tril_indices():
    result = torch.tril_indices(3, 3)
    assert_type(result.shape, IntTuple[2, int])


def test_tril_indices_rectangular():
    result = torch.tril_indices(4, 5)
    assert_type(result.shape, IntTuple[2, int])


def test_tril_indices_with_offset():
    result = torch.tril_indices(3, 3, offset=1)
    assert_type(result.shape, IntTuple[2, int])


# Test: torch.triu_indices
def test_triu_indices():
    result = torch.triu_indices(3, 3)
    assert_type(result.shape, IntTuple[2, int])


def test_triu_indices_rectangular():
    result = torch.triu_indices(5, 4)
    assert_type(result.shape, IntTuple[2, int])


def test_triu_indices_with_offset():
    result = torch.triu_indices(3, 3, offset=-1)
    assert_type(result.shape, IntTuple[2, int])


def check_tri_indices_runtime_parameters(row: int, col: int, offset: int) -> None:
    lower = torch.tril_indices(row=row, col=col, offset=offset)
    upper = torch.triu_indices(row, col, offset)
    assert_type(lower.shape, IntTuple[2, int])
    assert_type(upper.shape, IntTuple[2, int])
