# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Regression tests for repeat and expand with symbolic dimensions from size()

These operations must work with symbolic dimensions like Dim[N] returned from x.size().
Previously failed when iter_shape_dims() filtered out Type::Quantified dimensions.
"""

from typing import assert_type, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor


def test_repeat_symbolic[N](x: Tensor[N, 1]):
    """Repeat with symbolic dimension from size()"""
    # Get symbolic dimension from size()
    n = x.size(0)  # Returns Dim[N]

    # Repeat using symbolic dimension and literal
    # This previously failed with "repeat sizes length 1 doesn't match tensor rank 2"
    # because iter_shape_dims() filtered out the Quantified(N) type
    y = x.repeat(n, 3)

    # Should produce [N*N, 3]
    assert_type(y, Tensor[N * N, 3])


def test_expand_symbolic[N](x: Tensor[N, 1]):
    """Expand with symbolic dimension from size()"""
    # Get symbolic dimension
    n = x.size(0)  # Returns Dim[N]

    # Expand using symbolic dimension and literal
    # This previously failed with "expand target size length 1 doesn't match tensor rank 2"
    y = x.expand(n, 5)

    # Expands [N, 1] → [N, 5] (keeps dim 0, broadcasts dim 1)
    assert_type(y, Tensor[N, 5])


def test_expand_runtime_values[N, M](x: Tensor[N, M]):
    """Expand with multiple symbolic dimensions from size()"""
    n = x.size(0)
    m = x.size(1)

    # Use -1 to keep original dimension, and symbolic m for second dim
    y = x.expand(n, m)
    assert_type(y, Tensor[N, M])
