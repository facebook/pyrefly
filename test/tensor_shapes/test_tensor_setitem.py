# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Test Tensor.__setitem__ for in-place assignment

This verifies that tensor item assignment (e.g., tensor[index] = value) works correctly.
"""

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor


def test_simple_setitem():
    """Test basic tensor item assignment"""
    x: Tensor[10, 20] = torch.randn(10, 20)

    # Simple index assignment
    x[0] = 1.0
    x[1] = torch.ones(20)


def test_slice_setitem():
    """Test tensor slice assignment"""
    x: Tensor[10, 20] = torch.randn(10, 20)

    # Slice assignment
    x[0:5] = 0.0
    x[:, 0:10] = torch.zeros(10, 10)


def test_boolean_mask_setitem():
    """Test tensor assignment with boolean mask"""
    logits: Tensor[32, 100] = torch.randn(32, 100)
    v: Tensor[32, 1] = torch.randn(32, 1)

    # Boolean mask assignment (like in nanogpt.py)
    logits[logits < v[:, [-1]]] = -float("Inf")


def test_advanced_indexing_setitem():
    """Test advanced indexing assignment"""
    x: Tensor[10, 20, 30] = torch.randn(10, 20, 30)

    # Multi-dimensional indexing
    x[0, :, 5:10] = 1.0
    x[:5, 10:, :15] = torch.zeros(5, 10, 15)
