# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Regression test for adaptive pooling with symbolic output dimensions

Adaptive pooling should support both literal and symbolic dimensions in output_size.
Previously only worked with literal tuples.
"""

from typing import assert_type, TYPE_CHECKING

if TYPE_CHECKING:
    import torch.nn.functional as F
    from torch import Tensor
    from torch_shapes import Dim


def test_adaptive_pool_symbolic_batch[B](x: Tensor[B, 64, 56, 56]):
    """Adaptive pool with symbolic batch dimension and literal output size"""
    y = F.adaptive_avg_pool2d(x, (7, 7))
    # Batch dimension B is preserved, output spatial dims are 7x7
    assert_type(y, Tensor[B, 64, 7, 7])


def test_adaptive_pool_symbolic_output[B, S](x: Tensor[B, 64, 56, 56], s: Dim[S]):
    """Adaptive pool with symbolic output size dimensions"""
    # Use symbolic dimension s in output_size
    # This previously failed because tuple with symbolic dims wasn't handled
    y = F.adaptive_avg_pool2d(x, (s, s))

    # Output has symbolic spatial dimensions S (not Dim[S] in the type annotation)
    assert_type(y, Tensor[B, 64, S, S])


def test_adaptive_pool_mixed[B, H](x: Tensor[B, 64, 56, 56], h: Dim[H]):
    """Adaptive pool with mixed literal and symbolic output size"""
    # One literal, one symbolic dimension
    y = F.adaptive_avg_pool2d(x, (h, 7))

    # Output has one symbolic, one literal dimension
    assert_type(y, Tensor[B, 64, H, 7])
