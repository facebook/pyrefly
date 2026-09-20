# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Phase 2: Arithmetic & Basic Operations tests
# All operations preserve input shape (use IdentityMetaShape)
from typing import assert_type

import torch
from torch import Tensor

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


# Test: operations preserve shape on 1D tensors
def test_operations_1d():
    a: Tensor[[10]] = torch.randn(10)
    b: Tensor[[10]] = torch.randn(10)

    # Logical
    logical_and_result = torch.logical_and(a, b)
    assert_type(logical_and_result, Tensor[[10]])
