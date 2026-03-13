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
    a: Tensor[2, 3] = torch.randn(2, 3)
    b: Tensor[2, 3] = torch.randn(2, 3)
    result = torch.add(a, b)
    assert_type(result, Tensor[2, 3])


# Test: sub - element-wise subtraction
def test_sub():
    a: Tensor[2, 3] = torch.randn(2, 3)
    b: Tensor[2, 3] = torch.randn(2, 3)
    result = torch.sub(a, b)
    assert_type(result, Tensor[2, 3])


# Test: mul - element-wise multiplication
def test_mul():
    a: Tensor[2, 3] = torch.randn(2, 3)
    b: Tensor[2, 3] = torch.randn(2, 3)
    result = torch.mul(a, b)
    assert_type(result, Tensor[2, 3])


# Test: div - element-wise division
def test_div():
    a: Tensor[2, 3] = torch.randn(2, 3)
    b: Tensor[2, 3] = torch.randn(2, 3)
    result = torch.div(a, b)
    assert_type(result, Tensor[2, 3])


# Test: pow - element-wise power
def test_pow():
    a: Tensor[2, 3] = torch.randn(2, 3)
    result = torch.pow(a, 2.0)
    assert_type(result, Tensor[2, 3])


# Test: neg - element-wise negation
def test_neg():
    a: Tensor[2, 3] = torch.randn(2, 3)
    result = torch.neg(a)
    assert_type(result, Tensor[2, 3])


# Test: abs - element-wise absolute value
def test_abs():
    a: Tensor[2, 3] = torch.randn(2, 3)
    result = torch.abs(a)
    assert_type(result, Tensor[2, 3])


# Test: floor - element-wise floor
def test_floor():
    a: Tensor[2, 3] = torch.randn(2, 3)
    result = torch.floor(a)
    assert_type(result, Tensor[2, 3])


# Test: ceil - element-wise ceiling
def test_ceil():
    a: Tensor[2, 3] = torch.randn(2, 3)
    result = torch.ceil(a)
    assert_type(result, Tensor[2, 3])


# Test: round - element-wise rounding
def test_round():
    a: Tensor[2, 3] = torch.randn(2, 3)
    result = torch.round(a)
    assert_type(result, Tensor[2, 3])


# Test: arithmetic operations on 3D tensors
def test_add_3d():
    a: Tensor[2, 3, 4] = torch.randn(2, 3, 4)
    b: Tensor[2, 3, 4] = torch.randn(2, 3, 4)
    result = torch.add(a, b)
    assert_type(result, Tensor[2, 3, 4])


# Test: arithmetic method version
def test_add_method():
    a: Tensor[2, 3] = torch.randn(2, 3)
    b: Tensor[2, 3] = torch.randn(2, 3)
    result = a.add(b)
    assert_type(result, Tensor[2, 3])


# ==== Point-wise Mathematical Operations ====


# Test: sin - element-wise sine
def test_sin():
    a: Tensor[2, 3] = torch.randn(2, 3)
    result = torch.sin(a)
    assert_type(result, Tensor[2, 3])


# Test: cos - element-wise cosine
def test_cos():
    a: Tensor[2, 3] = torch.randn(2, 3)
    result = torch.cos(a)
    assert_type(result, Tensor[2, 3])


# Test: tan - element-wise tangent
def test_tan():
    a: Tensor[2, 3] = torch.randn(2, 3)
    result = torch.tan(a)
    assert_type(result, Tensor[2, 3])


# Test: exp - element-wise exponential
def test_exp():
    a: Tensor[2, 3] = torch.randn(2, 3)
    result = torch.exp(a)
    assert_type(result, Tensor[2, 3])


# Test: log - element-wise natural logarithm
def test_log():
    a: Tensor[2, 3] = torch.randn(2, 3)
    result = torch.log(a)
    assert_type(result, Tensor[2, 3])


# Test: sqrt - element-wise square root
def test_sqrt():
    a: Tensor[2, 3] = torch.randn(2, 3)
    result = torch.sqrt(a)
    assert_type(result, Tensor[2, 3])


# Test: tanh - element-wise hyperbolic tangent
def test_tanh():
    a: Tensor[2, 3] = torch.randn(2, 3)
    result = torch.tanh(a)
    assert_type(result, Tensor[2, 3])


# Test: math operations on 1D tensors
def test_sin_1d():
    a: Tensor[5] = torch.randn(5)
    result = torch.sin(a)
    assert_type(result, Tensor[5])


# Test: math method version
def test_sin_method():
    a: Tensor[2, 3] = torch.randn(2, 3)
    result = a.sin()
    assert_type(result, Tensor[2, 3])


# ==== Comparison Operations ====


# Test: eq - element-wise equality
def test_eq():
    a: Tensor[2, 3] = torch.randn(2, 3)
    b: Tensor[2, 3] = torch.randn(2, 3)
    result = torch.eq(a, b)
    assert_type(result, Tensor[2, 3])


# Test: ne - element-wise inequality
def test_ne():
    a: Tensor[2, 3] = torch.randn(2, 3)
    b: Tensor[2, 3] = torch.randn(2, 3)
    result = torch.ne(a, b)
    assert_type(result, Tensor[2, 3])


# Test: lt - element-wise less than
def test_lt():
    a: Tensor[2, 3] = torch.randn(2, 3)
    b: Tensor[2, 3] = torch.randn(2, 3)
    result = torch.lt(a, b)
    assert_type(result, Tensor[2, 3])


# Test: le - element-wise less than or equal
def test_le():
    a: Tensor[2, 3] = torch.randn(2, 3)
    b: Tensor[2, 3] = torch.randn(2, 3)
    result = torch.le(a, b)
    assert_type(result, Tensor[2, 3])


# Test: gt - element-wise greater than
def test_gt():
    a: Tensor[2, 3] = torch.randn(2, 3)
    b: Tensor[2, 3] = torch.randn(2, 3)
    result = torch.gt(a, b)
    assert_type(result, Tensor[2, 3])


# Test: ge - element-wise greater than or equal
def test_ge():
    a: Tensor[2, 3] = torch.randn(2, 3)
    b: Tensor[2, 3] = torch.randn(2, 3)
    result = torch.ge(a, b)
    assert_type(result, Tensor[2, 3])


# Test: comparison method version
def test_eq_method():
    a: Tensor[2, 3] = torch.randn(2, 3)
    b: Tensor[2, 3] = torch.randn(2, 3)
    result = a.eq(b)
    assert_type(result, Tensor[2, 3])


# ==== Logical Operations ====


# Test: logical_and - element-wise logical AND
def test_logical_and():
    a: Tensor[2, 3] = torch.randn(2, 3)
    b: Tensor[2, 3] = torch.randn(2, 3)
    result = torch.logical_and(a, b)
    assert_type(result, Tensor[2, 3])


# Test: logical_or - element-wise logical OR
def test_logical_or():
    a: Tensor[2, 3] = torch.randn(2, 3)
    b: Tensor[2, 3] = torch.randn(2, 3)
    result = torch.logical_or(a, b)
    assert_type(result, Tensor[2, 3])


# Test: logical_not - element-wise logical NOT
def test_logical_not():
    a: Tensor[2, 3] = torch.randn(2, 3)
    result = torch.logical_not(a)
    assert_type(result, Tensor[2, 3])


# Test: logical method version
def test_logical_and_method():
    a: Tensor[2, 3] = torch.randn(2, 3)
    b: Tensor[2, 3] = torch.randn(2, 3)
    result = a.logical_and(b)
    assert_type(result, Tensor[2, 3])


# ==== Activation Functions ====


# Test: relu - ReLU activation
def test_relu():
    a: Tensor[2, 3] = torch.randn(2, 3)
    result = torch.relu(a)
    assert_type(result, Tensor[2, 3])


# Test: relu on 3D tensor
def test_relu_3d():
    a: Tensor[2, 3, 4] = torch.randn(2, 3, 4)
    result = torch.relu(a)
    assert_type(result, Tensor[2, 3, 4])


# Test: relu method version
def test_relu_method():
    a: Tensor[2, 3] = torch.randn(2, 3)
    result = a.relu()
    assert_type(result, Tensor[2, 3])


# ==== Clamping Operations ====


# Test: clamp - clamp tensor values
def test_clamp():
    a: Tensor[2, 3] = torch.randn(2, 3)
    result = torch.clamp(a, min=-1.0, max=1.0)
    assert_type(result, Tensor[2, 3])


# Test: clip - alias for clamp
def test_clip():
    a: Tensor[2, 3] = torch.randn(2, 3)
    result = torch.clip(a, min=-1.0, max=1.0)
    assert_type(result, Tensor[2, 3])


# Test: clamp method version
def test_clamp_method():
    a: Tensor[2, 3] = torch.randn(2, 3)
    result = a.clamp(min=-1.0, max=1.0)
    assert_type(result, Tensor[2, 3])


# ==== Shape Preservation Verification ====


# Test: operations preserve shape on 4D tensors
def test_operations_4d():
    a: Tensor[2, 3, 4, 5] = torch.randn(2, 3, 4, 5)
    b: Tensor[2, 3, 4, 5] = torch.randn(2, 3, 4, 5)

    # Arithmetic
    add_result = torch.add(a, b)
    assert_type(add_result, Tensor[2, 3, 4, 5])

    # Math
    sin_result = torch.sin(a)
    assert_type(sin_result, Tensor[2, 3, 4, 5])

    # Comparison
    eq_result = torch.eq(a, b)
    assert_type(eq_result, Tensor[2, 3, 4, 5])

    # Activation
    relu_result = torch.relu(a)
    assert_type(relu_result, Tensor[2, 3, 4, 5])


# Test: operations preserve shape on 1D tensors
def test_operations_1d():
    a: Tensor[10] = torch.randn(10)
    b: Tensor[10] = torch.randn(10)

    # Arithmetic
    mul_result = torch.mul(a, b)
    assert_type(mul_result, Tensor[10])

    # Math
    exp_result = torch.exp(a)
    assert_type(exp_result, Tensor[10])

    # Comparison
    lt_result = torch.lt(a, b)
    assert_type(lt_result, Tensor[10])

    # Logical
    logical_and_result = torch.logical_and(a, b)
    assert_type(logical_and_result, Tensor[10])
