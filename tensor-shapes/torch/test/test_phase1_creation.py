# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Phase 1.3: Tensor creation operations tests
from typing import assert_type

import torch
from torch import Tensor

# ==== *_like Operations (preserve shape) ====


# Test: torch.zeros_like
def test_zeros_like():
    x: Tensor[3, 4] = torch.randn(3, 4)
    result = torch.zeros_like(x)
    assert_type(result, Tensor[3, 4])


def test_zeros_like_3d():
    x: Tensor[2, 3, 4] = torch.randn(2, 3, 4)
    result = torch.zeros_like(x)
    assert_type(result, Tensor[2, 3, 4])


# Test: torch.ones_like
def test_ones_like():
    x: Tensor[3, 4] = torch.randn(3, 4)
    result = torch.ones_like(x)
    assert_type(result, Tensor[3, 4])


def test_ones_like_1d():
    x: Tensor[5] = torch.randn(5)
    result = torch.ones_like(x)
    assert_type(result, Tensor[5])


# Test: torch.full_like
def test_full_like():
    x: Tensor[3, 4] = torch.randn(3, 4)
    result = torch.full_like(x, 2.5)
    assert_type(result, Tensor[3, 4])


# Test: torch.empty_like
def test_empty_like():
    x: Tensor[3, 4] = torch.randn(3, 4)
    result = torch.empty_like(x)
    assert_type(result, Tensor[3, 4])


# Test: torch.rand_like
def test_rand_like():
    x: Tensor[2, 3] = torch.randn(2, 3)
    result = torch.rand_like(x)
    assert_type(result, Tensor[2, 3])


# Test: torch.randn_like
def test_randn_like():
    x: Tensor[2, 3, 4] = torch.randn(2, 3, 4)
    result = torch.randn_like(x)
    assert_type(result, Tensor[2, 3, 4])


# ==== Diagonal Operations ====


# Test: torch.diag_embed
def test_diag_embed_1d():
    x: Tensor[3] = torch.randn(3)
    # Embeds into 3x3 diagonal matrix
    result = torch.diag_embed(x)
    assert_type(result, Tensor[3, 3])


def test_diag_embed_2d():
    x: Tensor[2, 3] = torch.randn(2, 3)
    # Embeds last dim (3) into 3x3, output: [2, 3, 3]
    result = torch.diag_embed(x)
    assert_type(result, Tensor[2, 3, 3])


def test_diag_embed_method():
    x: Tensor[4] = torch.randn(4)
    result = x.diag_embed()
    assert_type(result, Tensor[4, 4])


# ==== Triangular Operations (preserve shape) ====


# Test: torch.tril
def test_tril():
    x: Tensor[3, 3] = torch.randn(3, 3)
    # Lower triangular preserves shape
    result = torch.tril(x)
    assert_type(result, Tensor[3, 3])


def test_tril_rectangular():
    x: Tensor[4, 5] = torch.randn(4, 5)
    result = torch.tril(x)
    assert_type(result, Tensor[4, 5])


def test_tril_with_diagonal():
    x: Tensor[3, 3] = torch.randn(3, 3)
    # With diagonal offset
    result = torch.tril(x, diagonal=1)
    assert_type(result, Tensor[3, 3])


def test_tril_method():
    x: Tensor[4, 4] = torch.randn(4, 4)
    result = x.tril()
    assert_type(result, Tensor[4, 4])


# Test: torch.triu
def test_triu():
    x: Tensor[3, 3] = torch.randn(3, 3)
    # Upper triangular preserves shape
    result = torch.triu(x)
    assert_type(result, Tensor[3, 3])


def test_triu_rectangular():
    x: Tensor[5, 4] = torch.randn(5, 4)
    result = torch.triu(x)
    assert_type(result, Tensor[5, 4])


def test_triu_with_diagonal():
    x: Tensor[3, 3] = torch.randn(3, 3)
    result = torch.triu(x, diagonal=-1)
    assert_type(result, Tensor[3, 3])


def test_triu_method():
    x: Tensor[4, 4] = torch.randn(4, 4)
    result = x.triu()
    assert_type(result, Tensor[4, 4])


# ==== Triangular Indices ====


# Test: torch.tril_indices
def test_tril_indices():
    # Returns [2, num_indices] where num_indices is calculated
    _ = torch.tril_indices(3, 3)
    # We expect [2, ?] but can't assert exact second dim
    # Just verify the call works


def test_tril_indices_rectangular():
    _ = torch.tril_indices(4, 5)


def test_tril_indices_with_offset():
    _ = torch.tril_indices(3, 3, offset=1)


# Test: torch.triu_indices
def test_triu_indices():
    _ = torch.triu_indices(3, 3)


def test_triu_indices_rectangular():
    _ = torch.triu_indices(5, 4)


def test_triu_indices_with_offset():
    _ = torch.triu_indices(3, 3, offset=-1)
