# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Tests for literal int support in meta-shapes
# Demonstrates: size() literal tracking, numel() literals, dim() literals
from typing import assert_type, Literal

import torch
from torch import Tensor

# ==== tensor.size() -> tuple[Literal[...], ...] ====


def test_size_returns_literal_tuple():
    """size() returns tuple of literal ints"""
    x: Tensor[3, 4, 5] = torch.randn(3, 4, 5)
    size = x.size()
    # Should infer: tuple[Literal[3], Literal[4], Literal[5]]
    assert_type(size, tuple[Literal[3], Literal[4], Literal[5]])


def test_size_with_dim_returns_literal_int():
    """size(dim) returns single literal int"""
    x: Tensor[2, 7, 4] = torch.randn(2, 7, 4)
    dim0_size = x.size(0)
    dim1_size = x.size(1)
    dim2_size = x.size(2)
    # Should infer: Literal[2], Literal[7], Literal[4]
    assert_type(dim0_size, Literal[2])
    assert_type(dim1_size, Literal[7])
    assert_type(dim2_size, Literal[4])


def test_size_enables_reshape():
    """size() literal tracking enables reshape patterns"""
    x: Tensor[3, 4, 5] = torch.randn(3, 4, 5)
    size = x.size()
    # Type system tracks size as tuple[Literal[3], Literal[4], Literal[5]]
    assert_type(size, tuple[Literal[3], Literal[4], Literal[5]])
    # Note: torch.zeros(*size) would work but requires unpacking
    # This demonstrates the literal tracking capability


def test_size_partial_reshape():
    """size() enables partial dimension extraction"""
    x: Tensor[2, 3, 4, 5] = torch.randn(2, 3, 4, 5)
    # Get sizes for selective reshaping
    dim0 = x.size(0)
    dim3 = x.size(3)
    # Use literal sizes in computation
    # (This pattern enables dynamic reshaping with static verification)
    assert_type(dim0, Literal[2])
    assert_type(dim3, Literal[5])


# ==== tensor.numel() -> Literal[n] ====


def test_numel_returns_literal():
    """numel() returns literal int"""
    x: Tensor[3, 4, 5] = torch.randn(3, 4, 5)
    n = x.numel()
    # Should infer: Literal[60] (3*4*5=60)
    assert_type(n, Literal[60])


def test_numel_2d():
    """numel() for 2D tensor"""
    x: Tensor[10, 20] = torch.randn(10, 20)
    n = x.numel()
    # Should infer: Literal[200] (10*20=200)
    assert_type(n, Literal[200])


def test_numel_1d():
    """numel() for 1D tensor"""
    x: Tensor[7] = torch.randn(7)
    n = x.numel()
    # Should infer: Literal[7]
    assert_type(n, Literal[7])


def test_numel_enables_comparisons():
    """numel() literal enables static size checks"""
    x: Tensor[2, 3, 4] = torch.randn(2, 3, 4)
    n = x.numel()
    # Type system knows n is Literal[24]
    # This pattern enables size-based logic
    assert_type(n, Literal[24])


# ==== tensor.dim() / tensor.ndim() -> Literal[n] ====


def test_dim_returns_literal():
    """dim() returns literal int"""
    x: Tensor[3, 4, 5] = torch.randn(3, 4, 5)
    d = x.dim()
    # Should infer: Literal[3] (rank)
    assert_type(d, Literal[3])


def test_dim_2d():
    """dim() for 2D tensor"""
    x: Tensor[10, 20] = torch.randn(10, 20)
    d = x.dim()
    # Should infer: Literal[2]
    assert_type(d, Literal[2])


# def test_ndim_alias():
#    """ndim is alias for dim()"""
#    x: Tensor[2, 3, 4, 5] = torch.randn(2, 3, 4, 5)
#    d = x.ndim
#    # Should infer: Literal[4]
#    expected: Literal[4] = d


def test_nelement_returns_literal():
    """nelement() returns literal int (alias for numel)"""
    x: Tensor[5, 6] = torch.randn(5, 6)
    n = x.nelement()
    # Should infer: Literal[30] (5*6=30)
    assert_type(n, Literal[30])


# ==== Module-level torch.numel() returns int, same as method ====


def test_torch_numel_returns_int():
    """Module-level torch.numel() returns int, same as method version!"""
    x: Tensor[3, 4] = torch.randn(3, 4)
    result = torch.numel(x)
    # torch.numel() returns int, same as x.numel()
    assert_type(result, Literal[12])


# ==== Practical Benefits Demo ====


def test_size_tracking_through_variables():
    """size() literals propagate through variables"""
    x: Tensor[3, 4, 5] = torch.randn(3, 4, 5)
    size = x.size()

    # Type system tracks size as tuple[Literal[3], Literal[4], Literal[5]]
    # Can verify type at any point
    verified_size: tuple[Literal[3], Literal[4], Literal[5]] = size

    # Can pass to functions expecting precise sizes
    # (Future: could be used for torch.zeros(size))
    assert_type(verified_size, tuple[Literal[3], Literal[4], Literal[5]])


def test_dimension_literals_enable_checks():
    """Dimension literals enable compile-time checks"""
    x: Tensor[2, 3, 4] = torch.randn(2, 3, 4)

    # Get literal values
    numel_val = x.numel()  # Literal[24]
    dim_val = x.dim()  # Literal[3]
    size_val = x.size()  # tuple[Literal[2], Literal[3], Literal[4]]

    # Type system can verify relationships
    assert_type(numel_val, Literal[24])
    assert_type(dim_val, Literal[3])
    assert_type(size_val, tuple[Literal[2], Literal[3], Literal[4]])
