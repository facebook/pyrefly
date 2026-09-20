# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Phase 6: Specialized operations tests (FFT, Loss, Padding, Random, Properties)
from typing import assert_type, Literal

import torch
import torch.fft
import torch.nn
from shape_extensions import Int, IntVar
from torch import Tensor

# ==== FFT Operations ====


# ==== Random Sampling Operations ====


def test_bernoulli():
    """Bernoulli sampling"""
    x: Tensor[[3, 4]] = torch.randn(3, 4)
    result = torch.bernoulli(x)
    # Preserves shape
    assert_type(result, Tensor[[3, 4]])


def test_bernoulli_method():
    """Bernoulli sampling as method"""
    x: Tensor[[2, 5]] = torch.randn(2, 5)
    result = x.bernoulli()
    # Preserves shape
    assert_type(result, Tensor[[2, 5]])


def test_bernoulli_inplace():
    """Bernoulli sampling in-place"""
    x: Tensor[[4, 3]] = torch.randn(4, 3)
    result = x.bernoulli_()
    # Preserves shape
    assert_type(result, Tensor[[4, 3]])


def test_normal_inplace():
    """Normal distribution sampling in-place"""
    x: Tensor[[3, 4]] = torch.randn(3, 4)
    result = x.normal_()
    # Preserves shape
    assert_type(result, Tensor[[3, 4]])


def test_poisson():
    """Poisson sampling"""
    x: Tensor[[2, 3]] = torch.randn(2, 3)
    result = torch.poisson(x)
    # Preserves shape
    assert_type(result, Tensor[[2, 3]])


def test_random_inplace():
    """Random integers in-place"""
    x: Tensor[[3, 3]] = torch.randn(3, 3)
    result = x.random_()
    # Preserves shape
    assert_type(result, Tensor[[3, 3]])


def test_uniform_inplace():
    """Uniform distribution in-place"""
    x: Tensor[[4, 5]] = torch.randn(4, 5)
    result = x.uniform_()
    # Preserves shape
    assert_type(result, Tensor[[4, 5]])


# ==== Tensor Property Operations ====


def test_numel():
    """Number of elements"""
    x: Tensor[[3, 4, 5]] = torch.randn(3, 4, 5)
    result = torch.numel(x)
    # Returns int (symbolic multiplication of dimensions)
    assert_type(result, Literal[60])


# ==== Tier 3: torch.normal Overloads ====


def test_normal_tensor_tensor():
    """Normal with both tensor parameters"""
    mean: Tensor[[3, 4]] = torch.randn(3, 4)
    std: Tensor[[3, 4]] = torch.randn(3, 4)
    result = torch.normal(mean, std)
    assert_type(result, Tensor[[3, 4]])


def test_normal_tensor_tensor_mean_shape():
    mean: Tensor[[2, 3]] = torch.randn(2, 3)
    std: Tensor[[6]] = torch.randn(6)
    assert_type(torch.normal(mean, std), Tensor[[2, 3]])


def test_normal_tensor_scalar():
    """Normal with tensor mean, scalar std"""
    mean: Tensor[[2, 5]] = torch.randn(2, 5)
    result = torch.normal(mean, 0.5)
    assert_type(result, Tensor[[2, 5]])


def test_normal_scalar_tensor():
    """Normal with scalar mean, tensor std"""
    std: Tensor[[4, 3]] = torch.randn(4, 3)
    result = torch.normal(0.0, std)
    assert_type(result, Tensor[[4, 3]])


def test_normal_scalar_scalar_size():
    """Normal with scalar mean/std and size parameter"""
    result = torch.normal(0.0, 1.0, size=(3, 4))
    assert_type(result, Tensor[[3, 4]])


def test_normal_scalar_scalar_shape[N: IntVar](n: Int[N], plain: int):
    assert_type(torch.normal(0.0, 1.0, size=()), Tensor[[]])
    assert_type(torch.normal(0.0, 1.0, size=(n, plain)), Tensor[[N, int]])
