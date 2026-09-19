# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Phase 1.2: Missing reduction operations tests
from typing import assert_type

import torch
from shape_extensions import Elements, IntTuple, IntVar
from torch import Tensor

# ==== Tier 2: Additional Tuple-Returning Reductions ====


# Test: torch.mode (always returns tuple)
def test_mode_basic():
    x: Tensor[[4, 5]] = torch.randn(4, 5)
    values, indices = torch.mode(x, dim=1)
    # Reduce along dim 1: [4, 5] -> [4]
    assert_type(values, Tensor[[4]])
    assert_type(indices, Tensor[[4]])


def test_mode_keepdim():
    x: Tensor[[3, 4, 5]] = torch.randn(3, 4, 5)
    values, indices = torch.mode(x, dim=1, keepdim=True)
    # Reduce with keepdim: [3, 4, 5] -> [3, 1, 5]
    assert_type(values, Tensor[[3, 1, 5]])
    assert_type(indices, Tensor[[3, 1, 5]])


def test_mode_method():
    x: Tensor[[2, 3]] = torch.randn(2, 3)
    values, indices = x.mode(dim=0)
    assert_type(values, Tensor[[3]])
    assert_type(indices, Tensor[[3]])


# Test: torch.sort (always returns tuple, preserves shape)
def test_sort_basic():
    x: Tensor[[5]] = torch.randn(5)
    values, indices = torch.sort(x)
    # Sort along last dim: [5] -> [5] (preserves shape)
    assert_type(values, Tensor[[5]])
    assert_type(indices, Tensor[[5]])


def test_sort_2d():
    x: Tensor[[3, 4]] = torch.randn(3, 4)
    values, indices = torch.sort(x, dim=0)
    # Sort along dim 0: [3, 4] -> [3, 4] (preserves shape)
    assert_type(values, Tensor[[3, 4]])
    assert_type(indices, Tensor[[3, 4]])


def test_sort_method():
    x: Tensor[[2, 5, 3]] = torch.randn(2, 5, 3)
    values, indices = x.sort(dim=1, descending=True)
    # Sort along dim 1: [2, 5, 3] -> [2, 5, 3] (preserves shape)
    assert_type(values, Tensor[[2, 5, 3]])
    assert_type(indices, Tensor[[2, 5, 3]])


# Test: torch.kthvalue (always returns tuple)
def test_kthvalue_basic():
    x: Tensor[[10]] = torch.randn(10)
    values, indices = torch.kthvalue(x, k=3)
    # 3rd smallest along last dim: [10] -> []
    assert_type(values, Tensor[[]])
    assert_type(indices, Tensor[[]])


def test_kthvalue_2d():
    x: Tensor[[4, 5]] = torch.randn(4, 5)
    values, indices = torch.kthvalue(x, k=2, dim=1)
    # 2nd smallest along dim 1: [4, 5] -> [4]
    assert_type(values, Tensor[[4]])
    assert_type(indices, Tensor[[4]])


def test_kthvalue_method():
    x: Tensor[[3, 6]] = torch.randn(3, 6)
    values, indices = x.kthvalue(k=4, dim=1)
    # Kth value along dim 1: [3, 6] -> [3]
    assert_type(values, Tensor[[3]])
    assert_type(indices, Tensor[[3]])


# ==== Norm Operations ====


# Test: torch.norm
def test_norm_no_dim():
    x: Tensor[[3, 4]] = torch.randn(3, 4)
    # Compute norm over all dimensions -> scalar
    result = torch.norm(x)  # noqa: CITRINE(torchfix_deprecated_symbol_call)
    assert_type(result, Tensor[[]])  # Scalar


def test_norm_with_dim():
    x: Tensor[[3, 4, 5]] = torch.randn(3, 4, 5)
    # Compute norm along dim 1: [3, 4, 5] -> [3, 5]
    result = torch.norm(  # noqa: CITRINE(torchfix_deprecated_symbol_call)
        x, dim=1
    )
    assert_type(result, Tensor[[3, 5]])


def test_norm_with_keepdim():
    x: Tensor[[3, 4, 5]] = torch.randn(3, 4, 5)
    # Compute norm with keepdim: [3, 4, 5] -> [3, 1, 5]
    result = torch.norm(  # noqa: CITRINE(torchfix_deprecated_symbol_call)
        x, dim=1, keepdim=True
    )
    assert_type(result, Tensor[[3, 1, 5]])


def test_norm_multiple_dims():
    x: Tensor[[3, 4, 5]] = torch.randn(3, 4, 5)
    # Compute norm along dims (1, 2): [3, 4, 5] -> [3]
    result = torch.norm(  # noqa: CITRINE(torchfix_deprecated_symbol_call)
        x, dim=(1, 2)
    )
    assert_type(result, Tensor[[3]])


def test_norm_method():
    x: Tensor[[2, 3, 4]] = torch.randn(2, 3, 4)
    result = x.norm(dim=2)
    assert_type(result, Tensor[[2, 3]])


# Test: torch.dist
def test_dist_basic():
    x: Tensor[[3, 4]] = torch.randn(3, 4)
    y: Tensor[[3, 4]] = torch.randn(3, 4)
    # dist always returns scalar
    result = torch.dist(x, y)
    assert_type(result, Tensor[[]])  # Scalar


def test_dist_1d():
    x: Tensor[[5]] = torch.randn(5)
    y: Tensor[[5]] = torch.randn(5)
    result = torch.dist(x, y)
    assert_type(result, Tensor[[]])  # Scalar


def test_dist_method():
    x: Tensor[[2, 3, 4]] = torch.randn(2, 3, 4)
    y: Tensor[[2, 3, 4]] = torch.randn(2, 3, 4)
    result = x.dist(y)
    assert_type(result, Tensor[[]])  # Scalar


# ==== Tier 3: Additional Statistical Operations ====


# Test: torch.var_mean (returns tuple)
def test_var_mean_no_dim():
    x: Tensor[[3, 4]] = torch.randn(3, 4)
    var, mean = torch.var_mean(x)
    # Both scalars when no dim
    assert_type(var, Tensor[[]])
    assert_type(mean, Tensor[[]])


def test_var_mean_with_dim():
    x: Tensor[[3, 4, 5]] = torch.randn(3, 4, 5)
    var, mean = torch.var_mean(x, dim=1)
    # Reduce along dim 1: [3, 4, 5] -> [3, 5]
    assert_type(var, Tensor[[3, 5]])
    assert_type(mean, Tensor[[3, 5]])


# Test: torch.std_mean (returns tuple)
def test_std_mean_no_dim():
    x: Tensor[[2, 3]] = torch.randn(2, 3)
    std, mean = torch.std_mean(x)
    # Both scalars when no dim
    assert_type(std, Tensor[[]])
    assert_type(mean, Tensor[[]])


def test_std_mean_with_dim():
    x: Tensor[[4, 5, 6]] = torch.randn(4, 5, 6)
    std, mean = torch.std_mean(x, dim=2, keepdim=True)
    # Reduce with keepdim: [4, 5, 6] -> [4, 5, 1]
    assert_type(std, Tensor[[4, 5, 1]])
    assert_type(mean, Tensor[[4, 5, 1]])


def test_statistical_reduction_positional_unbiased():
    x: Tensor[[2, 3]] = torch.randn(2, 3)
    var, var_mean = torch.var_mean(x, False)
    std, std_mean = torch.std_mean(x, False)
    assert_type(var, Tensor[[]])
    assert_type(var_mean, Tensor[[]])
    assert_type(std, Tensor[[]])
    assert_type(std_mean, Tensor[[]])


def test_statistical_reduction_positional_dim():
    x: Tensor[[2, 3]] = torch.randn(2, 3)
    var, var_mean = torch.var_mean(x, 1)
    std, std_mean = torch.std_mean(x, 1)
    assert_type(var, Tensor[[2]])
    assert_type(var_mean, Tensor[[2]])
    assert_type(std, Tensor[[2]])
    assert_type(std_mean, Tensor[[2]])


def test_statistical_reduction_explicit_none_keepdim():
    x: Tensor[[2, 3]] = torch.randn(2, 3)
    var, var_mean = torch.var_mean(x, dim=None, keepdim=True)
    std, std_mean = torch.std_mean(x, dim=None, keepdim=True)
    assert_type(var, Tensor[[1, 1]])
    assert_type(var_mean, Tensor[[1, 1]])
    assert_type(std, Tensor[[1, 1]])
    assert_type(std_mean, Tensor[[1, 1]])


def test_reduction_tuple_and_negative_dims() -> None:
    x: Tensor[[2, 3, 4]] = torch.randn(2, 3, 4)

    assert_type(torch.sum(x, dim=(0, -1)), Tensor[[3]])
    assert_type(x.mean(dim=(-1, -3), keepdim=True), Tensor[[1, 3, 1]])
    var, mean = torch.var_mean(x, dim=(0, 2))
    assert_type(var, Tensor[[3]])
    assert_type(mean, Tensor[[3]])


def test_reduction_scalar_and_empty_dim_tuple() -> None:
    scalar: Tensor[[]] = torch.tensor(1)
    x: Tensor[[2, 3, 4]] = torch.randn(2, 3, 4)

    assert_type(torch.sum(scalar, dim=0), Tensor[[]])
    assert_type(scalar.sum(dim=-1, keepdim=True), Tensor[[]])
    assert_type(torch.sum(x, dim=()), Tensor[[]])
    assert_type(x.sum(dim=(), keepdim=True), Tensor[[1, 1, 1]])
    assert_type(torch.sum(x, dim=None, keepdim=True), Tensor[[1, 1, 1]])


def test_standalone_reduction_keyword_input() -> None:
    x: Tensor[[2, 3, 4]] = torch.randn(2, 3, 4)

    assert_type(torch.sum(input=x, dim=1), Tensor[[2, 4]])
    assert_type(torch.logsumexp(input=x, dim=(0, 2)), Tensor[[3]])
    values, indices = torch.max(input=x, dim=1)
    assert_type(values, Tensor[[2, 4]])
    assert_type(indices, Tensor[[2, 4]])
    assert_type(torch.max(input=x, other=x), Tensor[[2, 3, 4]])


def test_elementwise_min_max_broadcast() -> None:
    left: Tensor[[2, 1]] = torch.randn(2, 1)
    right: Tensor[[3]] = torch.randn(3)

    assert_type(torch.max(left, right), Tensor[[2, 3]])
    assert_type(torch.min(left, right), Tensor[[2, 3]])


def test_all_any_tuple_dims() -> None:
    x: Tensor[[2, 3, 4]] = torch.randn(2, 3, 4)

    assert_type(torch.all(input=x, dim=(0, 2)), Tensor[[3]])
    assert_type(torch.any(input=x, dim=(0, -1), keepdim=True), Tensor[[1, 3, 1]])


def check_unknown_rank_reduction(x: Tensor) -> None:
    assert_type(torch.sum(input=x, dim=0), Tensor[IntTuple])
    assert_type(torch.sum(input=x, dim=-1), Tensor[IntTuple])


def check_symbolic_rank_last_dim_reduction[Bs: IntTuple, N: IntVar](
    x: Tensor[[*Elements[Bs], N]],
    keepdim: bool,
) -> None:
    assert_type(torch.sum(x, dim=-1), Tensor[Bs])
    assert_type(x.mean(dim=-1, keepdim=True), Tensor[[*Elements[Bs], 1]])
    assert_type(x.mean(dim=-1), Tensor[Bs])
    assert_type(x.std(dim=-1, keepdim=True), Tensor[[*Elements[Bs], 1]])
    assert_type(x.std(dim=-1), Tensor[Bs])
    assert_type(x.mean(dim=-1, keepdim=keepdim), Tensor[IntTuple])


def check_symbolic_and_gradual_reductions[N: IntVar](
    x: Tensor[[N, 3, 4]],
    dim: int,
    dims: tuple[int, ...],
    keepdim: bool,
) -> None:
    assert_type(torch.sum(x, dim=-1), Tensor[[N, 3]])
    assert_type(torch.sum(x, dim=dim), Tensor[IntTuple])
    assert_type(x.std(dim=dims), Tensor[IntTuple])
    assert_type(torch.mean(x, dim=1, keepdim=keepdim), Tensor[IntTuple])
