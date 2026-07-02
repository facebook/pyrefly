# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Phase 5: Advanced indexing & conditional operations tests
from typing import assert_type

import torch
from torch import Tensor

# ==== torch.where ====


def test_where_2d():
    """Conditional element-wise selection"""
    condition: Tensor[3, 4] = torch.ones(3, 4)
    x: Tensor[3, 4] = torch.randn(3, 4)
    y: Tensor[3, 4] = torch.randn(3, 4)
    result = torch.where(condition, x, y)
    # Returns same shape as inputs: (3, 4)
    assert_type(result, Tensor[3, 4])


def test_where_broadcasting():
    """where with broadcasting"""
    condition: Tensor[3, 1] = torch.ones(3, 1)
    x: Tensor[3, 4] = torch.randn(3, 4)
    y: Tensor[3, 4] = torch.randn(3, 4)
    result = torch.where(condition, x, y)
    # Returns shape of x: (3, 4)
    assert_type(result, Tensor[3, 4])


# ==== torch.masked_fill ====


def test_masked_fill():
    """Fill masked elements"""
    x: Tensor[3, 4] = torch.randn(3, 4)
    mask: Tensor[3, 4] = torch.ones(3, 4)
    result = torch.masked_fill(x, mask, 0.0)
    # Preserves shape: (3, 4)
    assert_type(result, Tensor[3, 4])


def test_masked_fill_method():
    """Fill masked elements as method"""
    x: Tensor[4, 5] = torch.randn(4, 5)
    mask: Tensor[4, 5] = torch.ones(4, 5)
    result = x.masked_fill(mask, -1.0)
    # Preserves shape: (4, 5)
    assert_type(result, Tensor[4, 5])


def test_masked_fill_inplace():
    """Fill masked elements in-place"""
    x: Tensor[2, 3] = torch.randn(2, 3)
    mask: Tensor[2, 3] = torch.ones(2, 3)
    result = x.masked_fill_(mask, 0.0)
    # Preserves shape: (2, 3)
    assert_type(result, Tensor[2, 3])


# ==== torch.masked_scatter ====


def test_masked_scatter():
    """Scatter into masked positions"""
    x: Tensor[3, 4] = torch.randn(3, 4)
    mask: Tensor[3, 4] = torch.ones(3, 4)
    source: Tensor[12] = torch.randn(12)
    result = torch.masked_scatter(x, mask, source)
    # Preserves shape: (3, 4)
    assert_type(result, Tensor[3, 4])


def test_masked_scatter_method():
    """Scatter into masked positions as method"""
    x: Tensor[2, 5] = torch.randn(2, 5)
    mask: Tensor[2, 5] = torch.ones(2, 5)
    source: Tensor[10] = torch.randn(10)
    result = x.masked_scatter(mask, source)
    # Preserves shape: (2, 5)
    assert_type(result, Tensor[2, 5])


# ==== torch.index_add ====


def test_index_add():
    """Add values at indices"""
    x: Tensor[3, 5] = torch.randn(3, 5)
    index: Tensor[2] = torch.randn(2)
    source: Tensor[2, 5] = torch.randn(2, 5)
    result = torch.index_add(x, 0, index, source)
    # Preserves shape: (3, 5)
    assert_type(result, Tensor[3, 5])


def test_index_add_method():
    """Add values at indices as method"""
    x: Tensor[4, 3] = torch.randn(4, 3)
    index: Tensor[2] = torch.randn(2)
    source: Tensor[4, 2] = torch.randn(4, 2)
    result = x.index_add(1, index, source)
    # Preserves shape: (4, 3)
    assert_type(result, Tensor[4, 3])


def test_index_add_inplace():
    """Add values at indices in-place"""
    x: Tensor[5, 4] = torch.randn(5, 4)
    index: Tensor[3] = torch.randn(3)
    source: Tensor[3, 4] = torch.randn(3, 4)
    result = x.index_add_(0, index, source)
    # Preserves shape: (5, 4)
    assert_type(result, Tensor[5, 4])


# ==== torch.index_copy ====


def test_index_copy():
    """Copy values to indices"""
    x: Tensor[3, 5] = torch.randn(3, 5)
    index: Tensor[2] = torch.randn(2)
    source: Tensor[2, 5] = torch.randn(2, 5)
    result = torch.index_copy(x, 0, index, source)
    # Preserves shape: (3, 5)
    assert_type(result, Tensor[3, 5])


def test_index_copy_method():
    """Copy values to indices as method"""
    x: Tensor[4, 3] = torch.randn(4, 3)
    index: Tensor[2] = torch.randn(2)
    source: Tensor[4, 2] = torch.randn(4, 2)
    result = x.index_copy(1, index, source)
    # Preserves shape: (4, 3)
    assert_type(result, Tensor[4, 3])


# ==== torch.index_put ====


def test_index_put():
    """Put values at multi-dimensional indices"""
    x: Tensor[3, 4] = torch.randn(3, 4)
    idx1: Tensor[2] = torch.randn(2)
    idx2: Tensor[2] = torch.randn(2)
    values: Tensor[2] = torch.randn(2)
    result = torch.index_put(x, (idx1, idx2), values)
    # Preserves shape: (3, 4)
    assert_type(result, Tensor[3, 4])


def test_index_put_method():
    """Put values at indices as method"""
    x: Tensor[5, 5] = torch.randn(5, 5)
    indices: Tensor[3] = torch.randn(3)
    values: Tensor[3] = torch.randn(3)
    result = x.index_put((indices,), values)
    # Preserves shape: (5, 5)
    assert_type(result, Tensor[5, 5])


# ==== torch.index_fill ====


def test_index_fill():
    """Fill indices with value"""
    x: Tensor[3, 5] = torch.randn(3, 5)
    index: Tensor[2] = torch.randn(2)
    result = torch.index_fill(x, 0, index, 0.0)
    # Preserves shape: (3, 5)
    assert_type(result, Tensor[3, 5])


def test_index_fill_method():
    """Fill indices with value as method"""
    x: Tensor[4, 3] = torch.randn(4, 3)
    index: Tensor[2] = torch.randn(2)
    result = x.index_fill(1, index, 1.0)
    # Preserves shape: (4, 3)
    assert_type(result, Tensor[4, 3])


# ==== torch.take ====


def test_take_1d():
    """Take elements at flat indices"""
    x: Tensor[3, 4] = torch.randn(3, 4)
    index: Tensor[5] = torch.randn(5)
    result = torch.take(x, index)
    # Output shape matches index: (5,)
    assert_type(result, Tensor[5])


def test_take_2d_index():
    """Take with 2D index"""
    x: Tensor[10] = torch.randn(10)
    index: Tensor[2, 3] = torch.randn(2, 3)
    result = torch.take(x, index)
    # Output shape matches index: (2, 3)
    assert_type(result, Tensor[2, 3])


def test_take_method():
    """Take as method"""
    x: Tensor[4, 5] = torch.randn(4, 5)
    index: Tensor[8] = torch.randn(8)
    result = x.take(index)
    # Output shape matches index: (8,)
    assert_type(result, Tensor[8])


# ==== torch.take_along_dim ====


def test_take_along_dim_2d():
    """Take along dimension with index tensor"""
    x: Tensor[3, 4] = torch.randn(3, 4)
    indices: Tensor[3, 2] = torch.randn(3, 2)
    result = torch.take_along_dim(x, indices, dim=1)
    # Output shape matches indices: (3, 2)
    assert_type(result, Tensor[3, 2])


def test_take_along_dim_3d():
    """Take along dimension with 3D tensors"""
    x: Tensor[2, 3, 4] = torch.randn(2, 3, 4)
    indices: Tensor[2, 3, 2] = torch.randn(2, 3, 2)
    result = torch.take_along_dim(x, indices, dim=2)
    # Output shape matches indices: (2, 3, 2)
    assert_type(result, Tensor[2, 3, 2])


def test_take_along_dim_method():
    """Take along dimension as method"""
    x: Tensor[5, 6] = torch.randn(5, 6)
    indices: Tensor[5, 3] = torch.randn(5, 3)
    result = x.take_along_dim(indices, dim=1)
    # Output shape matches indices: (5, 3)
    assert_type(result, Tensor[5, 3])


# ==== torch.put ====


def test_put():
    """Put values at flat indices"""
    x: Tensor[3, 4] = torch.randn(3, 4)
    index: Tensor[3] = torch.randn(3)
    source: Tensor[3] = torch.randn(3)
    result = torch.put(x, index, source)
    # Preserves shape: (3, 4)
    assert_type(result, Tensor[3, 4])


def test_put_method():
    """Put values at indices as method"""
    x: Tensor[5, 5] = torch.randn(5, 5)
    index: Tensor[4] = torch.randn(4)
    source: Tensor[4] = torch.randn(4)
    result = x.put(index, source)
    # Preserves shape: (5, 5)
    assert_type(result, Tensor[5, 5])


def test_put_inplace():
    """Put values at indices in-place"""
    x: Tensor[4, 4] = torch.randn(4, 4)
    index: Tensor[3] = torch.randn(3)
    source: Tensor[3] = torch.randn(3)
    result = x.put_(index, source)
    # Preserves shape: (4, 4)
    assert_type(result, Tensor[4, 4])
