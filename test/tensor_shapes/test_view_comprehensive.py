# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Comprehensive test suite for .view()/.reshape() with symbolic dimensions"""

from typing import assert_type, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

# ===== POSITIVE TESTS (should work) =====


def test_literal_flatten():
    """Flatten with literal dimensions"""
    x: Tensor[10, 20] = torch.randn(10, 20)
    y = x.view(-1)
    assert_type(y, Tensor[200])
    assert_type(y, Tensor[200])


def test_literal_partial_infer():
    """Partial -1 inference with literals"""
    x: Tensor[10, 20] = torch.randn(10, 20)
    y = x.view(2, -1)
    assert_type(y, Tensor[2, 100])
    assert_type(y, Tensor[2, 100])


def test_symbolic_flatten[N, M](x: Tensor[N, M]) -> Tensor[N * M]:
    """Flatten with symbolic dimensions"""
    y = x.view(-1)
    assert_type(y, Tensor[(N * M)])
    assert_type(y, Tensor[N * M])
    return y


# Test with concrete tensor
result_flatten = test_symbolic_flatten(torch.randn(10, 20))
assert_type(result_flatten, Tensor[(10 * 20)])


def test_symbolic_partial_infer[C](x: Tensor[C]) -> Tensor[1, C, 1, 1]:
    """Partial -1 inference with symbolic dimensions"""
    y = x.view(1, -1, 1, 1)
    assert_type(y, Tensor[1, C, 1, 1])
    assert_type(y, Tensor[1, C, 1, 1])
    return y


# Test with concrete tensor
result_partial = test_symbolic_partial_infer(torch.randn(64))
assert_type(result_partial, Tensor[1, 64, 1, 1])


def test_symbolic_division[N](x: Tensor[N]) -> Tensor[5, N // 5]:
    """Division with symbolic dimensions"""
    y = x.view(5, -1)
    assert_type(y, Tensor[5, (N // 5)])
    assert_type(y, Tensor[5, N // 5])
    return y


# Test with concrete tensor
result_division = test_symbolic_division(torch.randn(100))
assert_type(result_division, Tensor[5, (100 // 5)])


def test_no_inference():
    """No -1, explicit dimensions"""
    x: Tensor[10, 20] = torch.randn(10, 20)
    y = x.view(200)
    assert_type(y, Tensor[200])
    assert_type(y, Tensor[200])


def test_incompatible_symbolic_shape[N](x: Tensor[N]):
    """Test symbolic reshape with division"""
    # This works symbolically even if N isn't divisible by 3 at runtime
    y = x.view(3, -1)
    assert_type(y, Tensor[3, (N // 3)])


# Test with concrete tensor
test_incompatible_symbolic_shape(torch.randn(100))
