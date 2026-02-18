# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Phase 1.2: Missing reduction operations tests
from typing import assert_type

import torch
from torch import Tensor

# ==== Reduction Operations (using ReduceMetaShape) ====


# Test: torch.median
def test_median_no_dim():
    x: Tensor[3, 4] = torch.randn(3, 4)
    # Reduce all dimensions to scalar
    result = torch.median(x)
    assert_type(result, Tensor[()])  # Scalar tensor (0-d)


def test_median_with_dim():
    x: Tensor[3, 4, 5] = torch.randn(3, 4, 5)
    # Reduce along dim 1: [3, 4, 5] -> [3, 5] (returns tuple)
    values, indices = torch.median(x, dim=1)
    assert_type(values, Tensor[3, 5])
    assert_type(indices, Tensor[3, 5])


def test_median_with_keepdim():
    x: Tensor[3, 4, 5] = torch.randn(3, 4, 5)
    # Reduce with keepdim: [3, 4, 5] -> [3, 1, 5] (returns tuple)
    values, indices = torch.median(x, dim=1, keepdim=True)
    assert_type(values, Tensor[3, 1, 5])
    assert_type(indices, Tensor[3, 1, 5])


def test_median_method():
    x: Tensor[2, 3] = torch.randn(2, 3)
    values, indices = x.median(dim=0)
    assert_type(values, Tensor[3])
    assert_type(indices, Tensor[3])


# Test: torch.logsumexp
def test_logsumexp_no_dim():
    x: Tensor[3, 4] = torch.randn(3, 4)
    result = torch.logsumexp(x)
    assert_type(result, Tensor[()])  # Scalar


def test_logsumexp_with_dim():
    x: Tensor[3, 4, 5] = torch.randn(3, 4, 5)
    # Reduce along dim 2: [3, 4, 5] -> [3, 4]
    result = torch.logsumexp(x, dim=2)
    assert_type(result, Tensor[3, 4])


def test_logsumexp_method():
    x: Tensor[2, 3, 4] = torch.randn(2, 3, 4)
    result = x.logsumexp(dim=1, keepdim=True)
    assert_type(result, Tensor[2, 1, 4])


# Test: torch.count_nonzero
def test_count_nonzero_no_dim():
    x: Tensor[3, 4] = torch.randn(3, 4)
    result = torch.count_nonzero(x)
    assert_type(result, Tensor[()])  # Scalar


def test_count_nonzero_with_dim():
    x: Tensor[3, 4, 5] = torch.randn(3, 4, 5)
    # Count along dim 0: [3, 4, 5] -> [4, 5]
    result = torch.count_nonzero(x, dim=0)
    assert_type(result, Tensor[4, 5])


def test_count_nonzero_method():
    x: Tensor[2, 3] = torch.randn(2, 3)
    result = x.count_nonzero(dim=1)
    assert_type(result, Tensor[2])


# Test: torch.aminmax (returns tuple of (min, max))
def test_aminmax_no_dim():
    x: Tensor[3, 4] = torch.randn(3, 4)
    min_val, max_val = torch.aminmax(x)
    # Returns tuple of (min, max), both scalars when no dim specified
    assert_type(min_val, Tensor[()])
    assert_type(max_val, Tensor[()])


def test_aminmax_with_dim():
    x: Tensor[3, 4, 5] = torch.randn(3, 4, 5)
    # Reduce along dim 1: [3, 4, 5] -> [3, 5]
    min_vals, max_vals = torch.aminmax(x, dim=1)
    # Returns tuple of (min_values, max_values), both have reduced shape
    assert_type(min_vals, Tensor[3, 5])
    assert_type(max_vals, Tensor[3, 5])


def test_aminmax_method():
    x: Tensor[2, 3] = torch.randn(2, 3)
    min_vals, max_vals = x.aminmax(dim=0, keepdim=True)
    # keepdim=True: [2, 3] -> [1, 3]
    assert_type(min_vals, Tensor[1, 3])
    assert_type(max_vals, Tensor[1, 3])


# ==== Cumulative Operations (preserve shape) ====


# Test: torch.cumsum
def test_cumsum_basic():
    x: Tensor[4] = torch.randn(4)
    # Cumulative sum preserves shape
    result = torch.cumsum(x, dim=0)
    assert_type(result, Tensor[4])


def test_cumsum_2d():
    x: Tensor[3, 4] = torch.randn(3, 4)
    # Cumsum along dim 0: shape [3, 4] -> [3, 4]
    result = torch.cumsum(x, dim=0)
    assert_type(result, Tensor[3, 4])


def test_cumsum_3d():
    x: Tensor[2, 3, 4] = torch.randn(2, 3, 4)
    # Cumsum along dim 2: shape preserved
    result = torch.cumsum(x, dim=2)
    assert_type(result, Tensor[2, 3, 4])


def test_cumsum_method():
    x: Tensor[3, 4] = torch.randn(3, 4)
    result = x.cumsum(dim=1)
    assert_type(result, Tensor[3, 4])


# Test: torch.cumprod
def test_cumprod_basic():
    x: Tensor[5] = torch.randn(5)
    result = torch.cumprod(x, dim=0)
    assert_type(result, Tensor[5])


def test_cumprod_2d():
    x: Tensor[3, 4] = torch.randn(3, 4)
    result = torch.cumprod(x, dim=1)
    assert_type(result, Tensor[3, 4])


def test_cumprod_method():
    x: Tensor[2, 3, 4] = torch.randn(2, 3, 4)
    result = x.cumprod(dim=0)
    assert_type(result, Tensor[2, 3, 4])


# Test: torch.cummax (returns tuple)
def test_cummax_basic():
    x: Tensor[4] = torch.randn(4)
    values, indices = torch.cummax(x, dim=0)
    assert_type(values, Tensor[4])
    assert_type(indices, Tensor[4])


def test_cummax_2d():
    x: Tensor[3, 4] = torch.randn(3, 4)
    values, indices = torch.cummax(x, dim=0)
    assert_type(values, Tensor[3, 4])
    assert_type(indices, Tensor[3, 4])


def test_cummax_method():
    x: Tensor[2, 3] = torch.randn(2, 3)
    values, indices = x.cummax(dim=1)
    assert_type(values, Tensor[2, 3])
    assert_type(indices, Tensor[2, 3])


# Test: torch.cummin (returns tuple)
def test_cummin_basic():
    x: Tensor[4] = torch.randn(4)
    values, indices = torch.cummin(x, dim=0)
    assert_type(values, Tensor[4])
    assert_type(indices, Tensor[4])


def test_cummin_2d():
    x: Tensor[3, 4] = torch.randn(3, 4)
    values, indices = torch.cummin(x, dim=1)
    assert_type(values, Tensor[3, 4])
    assert_type(indices, Tensor[3, 4])


def test_cummin_method():
    x: Tensor[2, 3, 4] = torch.randn(2, 3, 4)
    values, indices = x.cummin(dim=2)
    assert_type(values, Tensor[2, 3, 4])
    assert_type(indices, Tensor[2, 3, 4])


# ==== Tier 2: Additional Tuple-Returning Reductions ====


# Test: torch.mode (always returns tuple)
def test_mode_basic():
    x: Tensor[4, 5] = torch.randn(4, 5)
    values, indices = torch.mode(x, dim=1)
    # Reduce along dim 1: [4, 5] -> [4]
    assert_type(values, Tensor[4])
    assert_type(indices, Tensor[4])


def test_mode_keepdim():
    x: Tensor[3, 4, 5] = torch.randn(3, 4, 5)
    values, indices = torch.mode(x, dim=1, keepdim=True)
    # Reduce with keepdim: [3, 4, 5] -> [3, 1, 5]
    assert_type(values, Tensor[3, 1, 5])
    assert_type(indices, Tensor[3, 1, 5])


def test_mode_method():
    x: Tensor[2, 3] = torch.randn(2, 3)
    values, indices = x.mode(dim=0)
    assert_type(values, Tensor[3])
    assert_type(indices, Tensor[3])


# Test: torch.topk (always returns tuple, changes dimension size to k)
def test_topk_basic():
    x: Tensor[10] = torch.randn(10)
    values, indices = torch.topk(x, k=3)
    # Returns top 3 values along last dim: [10] -> [3]
    assert_type(values, Tensor[3])
    assert_type(indices, Tensor[3])


def test_topk_2d():
    x: Tensor[4, 5] = torch.randn(4, 5)
    values, indices = torch.topk(x, k=2, dim=1)
    # Top 2 along dim 1: [4, 5] -> [4, 2]
    assert_type(values, Tensor[4, 2])
    assert_type(indices, Tensor[4, 2])


def test_topk_method():
    x: Tensor[3, 6] = torch.randn(3, 6)
    values, indices = x.topk(k=4, dim=1)
    # Top 4 along dim 1: [3, 6] -> [3, 4]
    assert_type(values, Tensor[3, 4])
    assert_type(indices, Tensor[3, 4])


# Test: torch.sort (always returns tuple, preserves shape)
def test_sort_basic():
    x: Tensor[5] = torch.randn(5)
    values, indices = torch.sort(x)
    # Sort along last dim: [5] -> [5] (preserves shape)
    assert_type(values, Tensor[5])
    assert_type(indices, Tensor[5])


def test_sort_2d():
    x: Tensor[3, 4] = torch.randn(3, 4)
    values, indices = torch.sort(x, dim=0)
    # Sort along dim 0: [3, 4] -> [3, 4] (preserves shape)
    assert_type(values, Tensor[3, 4])
    assert_type(indices, Tensor[3, 4])


def test_sort_method():
    x: Tensor[2, 5, 3] = torch.randn(2, 5, 3)
    values, indices = x.sort(dim=1, descending=True)
    # Sort along dim 1: [2, 5, 3] -> [2, 5, 3] (preserves shape)
    assert_type(values, Tensor[2, 5, 3])
    assert_type(indices, Tensor[2, 5, 3])


# Test: torch.kthvalue (always returns tuple)
def test_kthvalue_basic():
    x: Tensor[10] = torch.randn(10)
    values, indices = torch.kthvalue(x, k=3)
    # 3rd smallest along last dim: [10] -> []
    assert_type(values, Tensor[()])
    assert_type(indices, Tensor[()])


def test_kthvalue_2d():
    x: Tensor[4, 5] = torch.randn(4, 5)
    values, indices = torch.kthvalue(x, k=2, dim=1)
    # 2nd smallest along dim 1: [4, 5] -> [4]
    assert_type(values, Tensor[4])
    assert_type(indices, Tensor[4])


def test_kthvalue_method():
    x: Tensor[3, 6] = torch.randn(3, 6)
    values, indices = x.kthvalue(k=4, dim=1)
    # Kth value along dim 1: [3, 6] -> [3]
    assert_type(values, Tensor[3])
    assert_type(indices, Tensor[3])


# ==== Norm Operations ====


# Test: torch.norm
def test_norm_no_dim():
    x: Tensor[3, 4] = torch.randn(3, 4)
    # Compute norm over all dimensions -> scalar
    result = torch.norm(x)
    assert_type(result, Tensor[()])  # Scalar


def test_norm_with_dim():
    x: Tensor[3, 4, 5] = torch.randn(3, 4, 5)
    # Compute norm along dim 1: [3, 4, 5] -> [3, 5]
    result = torch.norm(x, dim=1)
    assert_type(result, Tensor[3, 5])


def test_norm_with_keepdim():
    x: Tensor[3, 4, 5] = torch.randn(3, 4, 5)
    # Compute norm with keepdim: [3, 4, 5] -> [3, 1, 5]
    result = torch.norm(x, dim=1, keepdim=True)
    assert_type(result, Tensor[3, 1, 5])


def test_norm_multiple_dims():
    x: Tensor[3, 4, 5] = torch.randn(3, 4, 5)
    # Compute norm along dims (1, 2): [3, 4, 5] -> [3]
    result = torch.norm(x, dim=(1, 2))
    assert_type(result, Tensor[3])


def test_norm_method():
    x: Tensor[2, 3, 4] = torch.randn(2, 3, 4)
    result = x.norm(dim=2)
    assert_type(result, Tensor[2, 3])


# Test: torch.dist
def test_dist_basic():
    x: Tensor[3, 4] = torch.randn(3, 4)
    y: Tensor[3, 4] = torch.randn(3, 4)
    # dist always returns scalar
    result = torch.dist(x, y)
    assert_type(result, Tensor[()])  # Scalar


def test_dist_1d():
    x: Tensor[5] = torch.randn(5)
    y: Tensor[5] = torch.randn(5)
    result = torch.dist(x, y)
    assert_type(result, Tensor[()])  # Scalar


def test_dist_method():
    x: Tensor[2, 3, 4] = torch.randn(2, 3, 4)
    y: Tensor[2, 3, 4] = torch.randn(2, 3, 4)
    result = x.dist(y)
    assert_type(result, Tensor[()])  # Scalar


# ==== Tier 3: Additional Statistical Operations ====


# Test: torch.var_mean (returns tuple)
def test_var_mean_no_dim():
    x: Tensor[3, 4] = torch.randn(3, 4)
    var, mean = torch.var_mean(x)
    # Both scalars when no dim
    assert_type(var, Tensor[()])
    assert_type(mean, Tensor[()])


def test_var_mean_with_dim():
    x: Tensor[3, 4, 5] = torch.randn(3, 4, 5)
    var, mean = torch.var_mean(x, dim=1)
    # Reduce along dim 1: [3, 4, 5] -> [3, 5]
    assert_type(var, Tensor[3, 5])
    assert_type(mean, Tensor[3, 5])


# Test: torch.std_mean (returns tuple)
def test_std_mean_no_dim():
    x: Tensor[2, 3] = torch.randn(2, 3)
    std, mean = torch.std_mean(x)
    # Both scalars when no dim
    assert_type(std, Tensor[()])
    assert_type(mean, Tensor[()])


def test_std_mean_with_dim():
    x: Tensor[4, 5, 6] = torch.randn(4, 5, 6)
    std, mean = torch.std_mean(x, dim=2, keepdim=True)
    # Reduce with keepdim: [4, 5, 6] -> [4, 5, 1]
    assert_type(std, Tensor[4, 5, 1])
    assert_type(mean, Tensor[4, 5, 1])
