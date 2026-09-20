# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Phase 2: Arithmetic & Basic Operations tests
# All operations preserve input shape (use IdentityMetaShape)
from typing import assert_type

import torch
from torch import Tensor

# ==== Arithmetic Operations ====


# Test: add - element-wise addition
def test_add():
    a: Tensor[[2, 3]] = torch.randn(2, 3)
    b: Tensor[[2, 3]] = torch.randn(2, 3)
    result = torch.add(a, b)
    assert_type(result, Tensor[[2, 3]])


# Test: sub - element-wise subtraction
def test_sub():
    a: Tensor[[2, 3]] = torch.randn(2, 3)
    b: Tensor[[2, 3]] = torch.randn(2, 3)
    result = torch.sub(a, b)
    assert_type(result, Tensor[[2, 3]])


# Test: mul - element-wise multiplication
def test_mul():
    a: Tensor[[2, 3]] = torch.randn(2, 3)
    b: Tensor[[2, 3]] = torch.randn(2, 3)
    result = torch.mul(a, b)
    assert_type(result, Tensor[[2, 3]])


# Test: div - element-wise division
def test_div():
    a: Tensor[[2, 3]] = torch.randn(2, 3)
    b: Tensor[[2, 3]] = torch.randn(2, 3)
    result = torch.div(a, b)
    assert_type(result, Tensor[[2, 3]])


# Test: pow - element-wise power
def test_pow():
    a: Tensor[[2, 3]] = torch.randn(2, 3)
    result = torch.pow(a, 2.0)
    assert_type(result, Tensor[[2, 3]])


# Test: arithmetic operations on 3D tensors
def test_add_3d():
    a: Tensor[[2, 3, 4]] = torch.randn(2, 3, 4)
    b: Tensor[[2, 3, 4]] = torch.randn(2, 3, 4)
    result = torch.add(a, b)
    assert_type(result, Tensor[[2, 3, 4]])


# Test: arithmetic method version
def test_add_method():
    a: Tensor[[2, 3]] = torch.randn(2, 3)
    b: Tensor[[2, 3]] = torch.randn(2, 3)
    result = a.add(b)
    assert_type(result, Tensor[[2, 3]])


# ==== Comparison Operations ====


# Test: eq - element-wise equality
def test_eq():
    a: Tensor[[2, 3]] = torch.randn(2, 3)
    b: Tensor[[2, 3]] = torch.randn(2, 3)
    result = torch.eq(a, b)
    assert_type(result, Tensor[[2, 3]])


# Test: ne - element-wise inequality
def test_ne():
    a: Tensor[[2, 3]] = torch.randn(2, 3)
    b: Tensor[[2, 3]] = torch.randn(2, 3)
    result = torch.ne(a, b)
    assert_type(result, Tensor[[2, 3]])


# Test: lt - element-wise less than
def test_lt():
    a: Tensor[[2, 3]] = torch.randn(2, 3)
    b: Tensor[[2, 3]] = torch.randn(2, 3)
    result = torch.lt(a, b)
    assert_type(result, Tensor[[2, 3]])


# Test: le - element-wise less than or equal
def test_le():
    a: Tensor[[2, 3]] = torch.randn(2, 3)
    b: Tensor[[2, 3]] = torch.randn(2, 3)
    result = torch.le(a, b)
    assert_type(result, Tensor[[2, 3]])


# Test: gt - element-wise greater than
def test_gt():
    a: Tensor[[2, 3]] = torch.randn(2, 3)
    b: Tensor[[2, 3]] = torch.randn(2, 3)
    result = torch.gt(a, b)
    assert_type(result, Tensor[[2, 3]])


# Test: ge - element-wise greater than or equal
def test_ge():
    a: Tensor[[2, 3]] = torch.randn(2, 3)
    b: Tensor[[2, 3]] = torch.randn(2, 3)
    result = torch.ge(a, b)
    assert_type(result, Tensor[[2, 3]])


# Test: comparison method version
def test_eq_method():
    a: Tensor[[2, 3]] = torch.randn(2, 3)
    b: Tensor[[2, 3]] = torch.randn(2, 3)
    result = a.eq(b)
    assert_type(result, Tensor[[2, 3]])


def test_comparisons_with_scalars():
    a: Tensor[[2, 3]] = torch.randn(2, 3)
    assert_type(torch.ne(a, 0), Tensor[[2, 3]])
    assert_type(torch.lt(a, 0), Tensor[[2, 3]])
    assert_type(torch.le(a, 0), Tensor[[2, 3]])
    assert_type(torch.gt(a, 0), Tensor[[2, 3]])
    assert_type(torch.ge(a, 0), Tensor[[2, 3]])
    assert_type(a.ne(0), Tensor[[2, 3]])
    assert_type(a.lt(0), Tensor[[2, 3]])
    assert_type(a.le(0), Tensor[[2, 3]])
    assert_type(a.gt(0), Tensor[[2, 3]])
    assert_type(a.ge(0), Tensor[[2, 3]])


# ==== Logical Operations ====


# Test: logical_and - element-wise logical AND
def test_logical_and():
    a: Tensor[[2, 3]] = torch.randn(2, 3)
    b: Tensor[[2, 3]] = torch.randn(2, 3)
    result = torch.logical_and(a, b)
    assert_type(result, Tensor[[2, 3]])


# Test: logical_or - element-wise logical OR
def test_logical_or():
    a: Tensor[[2, 3]] = torch.randn(2, 3)
    b: Tensor[[2, 3]] = torch.randn(2, 3)
    result = torch.logical_or(a, b)
    assert_type(result, Tensor[[2, 3]])


# Test: logical method version
def test_logical_and_method():
    a: Tensor[[2, 3]] = torch.randn(2, 3)
    b: Tensor[[2, 3]] = torch.randn(2, 3)
    result = a.logical_and(b)
    assert_type(result, Tensor[[2, 3]])


# ==== Shape Preservation Verification ====


# Test: operations preserve shape on 4D tensors
def test_operations_4d():
    a: Tensor[[2, 3, 4, 5]] = torch.randn(2, 3, 4, 5)
    b: Tensor[[2, 3, 4, 5]] = torch.randn(2, 3, 4, 5)

    # Arithmetic
    add_result = torch.add(a, b)
    assert_type(add_result, Tensor[[2, 3, 4, 5]])

    # Comparison
    eq_result = torch.eq(a, b)
    assert_type(eq_result, Tensor[[2, 3, 4, 5]])


# Test: operations preserve shape on 1D tensors
def test_operations_1d():
    a: Tensor[[10]] = torch.randn(10)
    b: Tensor[[10]] = torch.randn(10)

    # Arithmetic
    mul_result = torch.mul(a, b)
    assert_type(mul_result, Tensor[[10]])

    # Comparison
    lt_result = torch.lt(a, b)
    assert_type(lt_result, Tensor[[10]])

    # Logical
    logical_and_result = torch.logical_and(a, b)
    assert_type(logical_and_result, Tensor[[10]])
