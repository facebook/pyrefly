# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Phase 1.4: Basic linear algebra operations tests
from typing import assert_type

import torch
from shape_extensions import IntVar
from torch import Tensor

# ==== torch.matmul ====


# Test: matmul - 1D @ 1D → scalar (dot product)
def test_matmul_1d_1d():
    a: Tensor[[3]] = torch.randn(3)
    b: Tensor[[3]] = torch.randn(3)
    result = torch.matmul(a, b)
    assert_type(result, Tensor[[]])  # Scalar (0-d tensor)


# Test: matmul - 1D @ 2D → 1D (vector @ matrix)
def test_matmul_1d_2d():
    a: Tensor[[3]] = torch.randn(3)
    b: Tensor[[3, 4]] = torch.randn(3, 4)
    result = torch.matmul(a, b)
    assert_type(result, Tensor[[4]])


# Test: matmul - 2D @ 1D → 1D (matrix @ vector)
def test_matmul_2d_1d():
    a: Tensor[[3, 4]] = torch.randn(3, 4)
    b: Tensor[[4]] = torch.randn(4)
    result = torch.matmul(a, b)
    assert_type(result, Tensor[[3]])


# Test: matmul - 2D @ 2D → 2D (matrix @ matrix)
def test_matmul_2d_2d():
    a: Tensor[[3, 4]] = torch.randn(3, 4)
    b: Tensor[[4, 5]] = torch.randn(4, 5)
    result = torch.matmul(a, b)
    assert_type(result, Tensor[[3, 5]])


# Test: matmul - batched 3D @ 3D
def test_matmul_3d_3d():
    a: Tensor[[2, 3, 4]] = torch.randn(2, 3, 4)
    b: Tensor[[2, 4, 5]] = torch.randn(2, 4, 5)
    result = torch.matmul(a, b)
    assert_type(result, Tensor[[2, 3, 5]])


def test_matmul_mixed_and_broadcast_batch_ranks():
    vector: Tensor[[4]] = torch.randn(4)
    matrix: Tensor[[3, 4]] = torch.randn(3, 4)
    batched: Tensor[[2, 4, 5]] = torch.randn(2, 4, 5)
    assert_type(matrix @ batched, Tensor[[2, 3, 5]])
    assert_type(vector @ batched, Tensor[[2, 5]])

    contractible: Tensor[[5, 4]] = torch.randn(5, 4)
    assert_type(batched.matmul(contractible), Tensor[[2, 4, 4]])
    right_vector: Tensor[[5]] = torch.randn(5)
    assert_type(batched.matmul(right_vector), Tensor[[2, 4]])

    left: Tensor[[7, 1, 3, 4]] = torch.randn(7, 1, 3, 4)
    right: Tensor[[5, 4, 6]] = torch.randn(5, 4, 6)
    assert_type(torch.matmul(left, right), Tensor[[7, 5, 3, 6]])


def test_matmul_preserves_permissive_v1_batch_merge():
    left: Tensor[[2, 3, 4]] = torch.randn(2, 3, 4)
    right: Tensor[[5, 4, 6]] = torch.randn(5, 4, 6)
    assert_type(torch.matmul(left, right), Tensor[[2, 3, 6]])


# Test: matmul - Tensor method version
def test_matmul_method():
    a: Tensor[[3, 4]] = torch.randn(3, 4)
    b: Tensor[[4, 5]] = torch.randn(4, 5)
    result = a.matmul(b)
    assert_type(result, Tensor[[3, 5]])


# ==== torch.mm (2D only) ====


# Test: mm - 2D @ 2D
def test_mm_2d_2d():
    a: Tensor[[3, 4]] = torch.randn(3, 4)
    b: Tensor[[4, 5]] = torch.randn(4, 5)
    result = torch.mm(a, b)
    assert_type(result, Tensor[[3, 5]])


# Test: mm - rectangular matrices
def test_mm_rectangular():
    a: Tensor[[5, 3]] = torch.randn(5, 3)
    b: Tensor[[3, 7]] = torch.randn(3, 7)
    result = torch.mm(a, b)
    assert_type(result, Tensor[[5, 7]])


# Test: mm - Tensor method version
def test_mm_method():
    a: Tensor[[3, 4]] = torch.randn(3, 4)
    b: Tensor[[4, 5]] = torch.randn(4, 5)
    result = a.mm(b)
    assert_type(result, Tensor[[3, 5]])


# ==== torch.bmm (3D only) ====


# Test: bmm - 3D @ 3D
def test_bmm_3d_3d():
    a: Tensor[[10, 3, 4]] = torch.randn(10, 3, 4)
    b: Tensor[[10, 4, 5]] = torch.randn(10, 4, 5)
    result = torch.bmm(a, b)
    assert_type(result, Tensor[[10, 3, 5]])


# Test: bmm - larger batch size
def test_bmm_large_batch():
    a: Tensor[[32, 5, 6]] = torch.randn(32, 5, 6)
    b: Tensor[[32, 6, 7]] = torch.randn(32, 6, 7)
    result = torch.bmm(a, b)
    assert_type(result, Tensor[[32, 5, 7]])


# Test: bmm - Tensor method version
def test_bmm_method():
    a: Tensor[[10, 3, 4]] = torch.randn(10, 3, 4)
    b: Tensor[[10, 4, 5]] = torch.randn(10, 4, 5)
    result = a.bmm(b)
    assert_type(result, Tensor[[10, 3, 5]])


# ==== torch.mv (matrix-vector) ====


# Test: mv - 2D @ 1D → 1D
def test_mv_2d_1d():
    mat: Tensor[[3, 4]] = torch.randn(3, 4)
    vec: Tensor[[4]] = torch.randn(4)
    result = torch.mv(mat, vec)
    assert_type(result, Tensor[[3]])


# Test: mv - rectangular matrix
def test_mv_rectangular():
    mat: Tensor[[5, 3]] = torch.randn(5, 3)
    vec: Tensor[[3]] = torch.randn(3)
    result = torch.mv(mat, vec)
    assert_type(result, Tensor[[5]])


# Test: mv - Tensor method version
def test_mv_method():
    mat: Tensor[[3, 4]] = torch.randn(3, 4)
    vec: Tensor[[4]] = torch.randn(4)
    result = mat.mv(vec)
    assert_type(result, Tensor[[3]])


def test_mv_symbolic[M: IntVar, K: IntVar](
    mat: Tensor[[M, K]], vec: Tensor[[K]]
) -> None:
    assert_type(torch.mv(mat, vec), Tensor[[M]])
    assert_type(mat.mv(vec), Tensor[[M]])


def test_outer():
    left: Tensor[[3]] = torch.randn(3)
    right: Tensor[[5]] = torch.randn(5)
    assert_type(torch.outer(left, right), Tensor[[3, 5]])


def test_outer_symbolic[M: IntVar, N: IntVar](
    left: Tensor[[M]], right: Tensor[[N]]
) -> None:
    assert_type(torch.outer(left, right), Tensor[[M, N]])


def test_mv_outer_bare_tensors(left: Tensor, right: Tensor) -> None:
    assert_type(torch.mv(left, right), Tensor[[int]])
    assert_type(left.mv(right), Tensor[[int]])
    assert_type(torch.outer(left, right), Tensor[[int, int]])


# ==== torch.dot (dot product) ====


# Test: dot - 1D @ 1D → scalar
def test_dot_1d_1d():
    a: Tensor[[5]] = torch.randn(5)
    b: Tensor[[5]] = torch.randn(5)
    result = torch.dot(a, b)
    assert_type(result, Tensor[[]])  # Scalar (0-d tensor)


# Test: dot - longer vectors
def test_dot_long_vectors():
    a: Tensor[[100]] = torch.randn(100)
    b: Tensor[[100]] = torch.randn(100)
    result = torch.dot(a, b)
    assert_type(result, Tensor[[]])  # Scalar


# Test: dot - Tensor method version
def test_dot_method():
    a: Tensor[[5]] = torch.randn(5)
    b: Tensor[[5]] = torch.randn(5)
    result = a.dot(b)
    assert_type(result, Tensor[[]])  # Scalar


# Note: @ operator (__matmul__) tests omitted for now
# The @ operator requires special meta-shape handling that will be added in a future update
# All the direct method calls (torch.matmul, tensor.matmul) work correctly
