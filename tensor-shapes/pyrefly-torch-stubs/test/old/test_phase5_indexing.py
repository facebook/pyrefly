# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Phase 5: Advanced indexing & conditional operations tests
from typing import assert_type

import torch
from shape_extensions import IntTuple
from torch import Tensor

# ==== torch.where ====


def test_where_2d():
    """Conditional element-wise selection"""
    condition: Tensor[[3, 4]] = torch.ones(3, 4)
    x: Tensor[[3, 4]] = torch.randn(3, 4)
    y: Tensor[[3, 4]] = torch.randn(3, 4)
    result = torch.where(condition, x, y)
    assert_type(result, Tensor[[3, 4]])


def test_where_indices():
    condition: Tensor[[3, 4]] = torch.ones(3, 4)
    assert_type(torch.where(condition), tuple[Tensor, ...])


def test_where_broadcasting():
    """where with broadcasting"""
    condition: Tensor[[3, 1]] = torch.ones(3, 1)
    x: Tensor[[3, 4]] = torch.randn(3, 4)
    y: Tensor[[3, 4]] = torch.randn(3, 4)
    result = torch.where(condition, x, y)
    assert_type(result, Tensor[[3, 4]])


def test_where_generic_shape[XShape: IntTuple](
    condition: Tensor, x: Tensor[XShape], y: Tensor
):
    assert_type(torch.where(condition, x, y), Tensor)


# ==== torch.index_put ====


def test_index_put():
    """Put values at multi-dimensional indices"""
    x: Tensor[[3, 4]] = torch.randn(3, 4)
    idx1: Tensor[[2]] = torch.randn(2)
    idx2: Tensor[[2]] = torch.randn(2)
    values: Tensor[[2]] = torch.randn(2)
    result = torch.index_put(x, (idx1, idx2), values)
    # Preserves shape: (3, 4)
    assert_type(result, Tensor[[3, 4]])


def test_index_put_method():
    """Put values at indices as method"""
    x: Tensor[[5, 5]] = torch.randn(5, 5)
    indices: Tensor[[3]] = torch.randn(3)
    values: Tensor[[3]] = torch.randn(3)
    result = x.index_put((indices,), values)
    # Preserves shape: (5, 5)
    assert_type(result, Tensor[[5, 5]])


def test_projection_bare_tensor_fallback(condition: Tensor, x: Tensor, y: Tensor):
    assert_type(torch.where(condition, x, y), Tensor)
