# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Phase 1.1: Missing shape operations tests
from typing import assert_type

import torch
from torch import Tensor


# Test: torch.unbind (removes a dimension)
def test_unbind_dim0():
    x: Tensor[3, 4] = torch.randn(3, 4)
    # unbind along dim 0 removes first dimension
    # Returns tuple of 3 tensors, each of shape [4]
    # Note: Type checking tuple elements is limited, so we just verify the call works
    _ = torch.unbind(x, dim=0)


def test_unbind_dim1():
    x: Tensor[3, 4] = torch.randn(3, 4)
    # unbind along dim 1 removes second dimension
    _ = torch.unbind(x, dim=1)


def test_unbind_method():
    x: Tensor[2, 3, 4] = torch.randn(2, 3, 4)
    _ = x.unbind(dim=1)


# Test: torch.movedim (moves dimensions to new positions)
def test_movedim_single():
    x: Tensor[2, 3, 4] = torch.randn(2, 3, 4)
    # Move dimension 0 to position 2: [2, 3, 4] -> [3, 4, 2]
    result = torch.movedim(x, source=0, destination=2)
    assert_type(result, Tensor[3, 4, 2])


def test_movedim_multiple():
    x: Tensor[2, 3, 4, 5] = torch.randn(2, 3, 4, 5)
    # Move dims 0 and 1 to positions 2 and 3: [2, 3, 4, 5] -> [4, 5, 2, 3]
    result = torch.movedim(x, source=(0, 1), destination=(2, 3))
    assert_type(result, Tensor[4, 5, 2, 3])


def test_movedim_negative():
    x: Tensor[2, 3, 4] = torch.randn(2, 3, 4)
    # Move last dimension to first: [2, 3, 4] -> [4, 2, 3]
    result = torch.movedim(x, source=-1, destination=0)
    assert_type(result, Tensor[4, 2, 3])


def test_movedim_method():
    x: Tensor[2, 3, 4] = torch.randn(2, 3, 4)
    result = x.movedim(source=1, destination=0)
    assert_type(result, Tensor[3, 2, 4])


# Test: torch.moveaxis (alias for movedim)
def test_moveaxis():
    x: Tensor[2, 3, 4] = torch.randn(2, 3, 4)
    result = torch.moveaxis(x, source=0, destination=2)
    assert_type(result, Tensor[3, 4, 2])


# Test: torch.unfold (sliding window view)
def test_unfold_basic():
    x: Tensor[8] = torch.randn(8)
    # unfold with size=3, step=1: (8-3)/1 + 1 = 6 windows of size 3
    # Output shape: [6, 3]
    result = torch.unfold(x, dimension=0, size=3, step=1)
    assert_type(result, Tensor[6, 3])


def test_unfold_2d():
    x: Tensor[4, 6] = torch.randn(4, 6)
    # unfold dimension 1 with size=2, step=2: (6-2)/2 + 1 = 3 windows
    # Output shape: [4, 3, 2]
    result = torch.unfold(x, dimension=1, size=2, step=2)
    assert_type(result, Tensor[4, 3, 2])


def test_unfold_method():
    x: Tensor[10] = torch.randn(10)
    # unfold with size=4, step=2: (10-4)/2 + 1 = 4 windows
    result = x.unfold(dimension=0, size=4, step=2)
    assert_type(result, Tensor[4, 4])


def test_unfold_3d():
    x: Tensor[2, 5, 8] = torch.randn(2, 5, 8)
    # unfold dimension 2 with size=3, step=1: (8-3)/1 + 1 = 6 windows
    # Output shape: [2, 5, 6, 3]
    result = x.unfold(dimension=2, size=3, step=1)
    assert_type(result, Tensor[2, 5, 6, 3])
