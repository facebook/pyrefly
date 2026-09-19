# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Phase 1.1: Missing shape operations tests
from typing import assert_type, reveal_type

import torch
from torch import Tensor


# Test: Tensor.unfold (sliding window view)
def test_unfold_basic():
    x: Tensor[[8]] = torch.randn(8)
    # unfold with size=3, step=1: (8-3)/1 + 1 = 6 windows of size 3
    # Output shape: [6, 3]
    result = x.unfold(dimension=0, size=3, step=1)
    assert_type(result, Tensor[[6, 3]])


def test_unfold_2d():
    x: Tensor[[4, 6]] = torch.randn(4, 6)
    # unfold dimension 1 with size=2, step=2: (6-2)/2 + 1 = 3 windows
    # Output shape: [4, 3, 2]
    result = x.unfold(dimension=1, size=2, step=2)
    assert_type(result, Tensor[[4, 3, 2]])


def test_unfold_method():
    x: Tensor[[10]] = torch.randn(10)
    # unfold with size=4, step=2: (10-4)/2 + 1 = 4 windows
    result = x.unfold(dimension=0, size=4, step=2)
    assert_type(result, Tensor[[4, 4]])


def test_unfold_3d():
    x: Tensor[[2, 5, 8]] = torch.randn(2, 5, 8)
    # unfold dimension 2 with size=3, step=1: (8-3)/1 + 1 = 6 windows
    # Output shape: [2, 5, 6, 3]
    result = x.unfold(dimension=2, size=3, step=1)
    assert_type(result, Tensor[[2, 5, 6, 3]])


def test_unfold_negative_dimension():
    x: Tensor[[8, 5]] = torch.randn(8, 5)
    result = x.unfold(dimension=-2, size=3, step=2)
    assert_type(result, Tensor[[3, 5, 3]])


def test_unfold_zero_size():
    x: Tensor[[5]] = torch.randn(5)
    reveal_type(x.unfold(dimension=0, size=0, step=2))  # revealed type: Tensor[[3, 0]]
    empty = torch.randn(0)
    reveal_type(
        empty.unfold(dimension=0, size=0, step=2)
    )  # revealed type: Tensor[[1, 0]]


def test_unfold_scalar():
    scalar: Tensor[[]] = torch.tensor(1)
    reveal_type(scalar.unfold(0, 0, 2))  # revealed type: Tensor[[0]]
    reveal_type(scalar.unfold(-1, 0, 2))  # revealed type: Tensor[[0]]
    assert_type(scalar.unfold(0, 1, 2), Tensor[[1]])
    assert_type(scalar.unfold(-1, 1, 2), Tensor[[1]])


def test_unfold_shapeless(x: Tensor):
    assert_type(x.unfold(dimension=0, size=3, step=1), Tensor)
