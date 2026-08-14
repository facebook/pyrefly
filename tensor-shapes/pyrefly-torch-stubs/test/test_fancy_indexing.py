# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import assert_type, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor


def test_tuple_indexing():
    """Test fancy indexing with tuples of integers"""
    x: Tensor[[2, 3, 4]] = torch.randn(2, 3, 4)

    # Tuple with single element
    y1 = x[:, (-1,), :]
    assert_type(y1, Tensor[[2, 1, 4]])

    # Tuple with multiple elements
    y2 = x[:, (0, 2), :]
    assert_type(y2, Tensor[[2, 2, 4]])

    # Tuple with all indices
    y3 = x[:, (0, 1, 2), :]
    assert_type(y3, Tensor[[2, 3, 4]])


def test_list_indexing_preserves_literal_length():
    """A list literal has a statically known length, so the indexed dimension is concrete."""
    x: Tensor[[2, 3, 4]] = torch.randn(2, 3, 4)

    # List with single element
    y1 = x[:, [-1], :]
    assert_type(y1, Tensor[[2, 1, 4]])

    # List with multiple elements
    y2 = x[:, [0, 2], :]
    assert_type(y2, Tensor[[2, 2, 4]])

    # List with all indices
    y3 = x[:, [0, 1, 2], :]
    assert_type(y3, Tensor[[2, 3, 4]])


def test_mixed_indexing():
    """Test mixing different index types"""
    x: Tensor[[10, 20, 30]] = torch.randn(10, 20, 30)

    # Mix slice, tuple, and slice
    y1 = x[:5, (1, 3, 5), :]
    assert_type(y1, Tensor[[5, 3, 30]])

    # Mix integer, tuple, and slice
    y2 = x[0, (1, 2), :]
    assert_type(y2, Tensor[[2, 30]])

    # Mix tuple, slice, integer
    y3 = x[(0, 1), :10, 5]
    assert_type(y3, Tensor[[2, 10]])


def test_comparison_with_integer_and_slice():
    """Compare fancy indexing with integer and slice indexing"""
    x: Tensor[[2, 3, 4]] = torch.randn(2, 3, 4)

    # Integer indexing removes dimension
    y1 = x[:, 0, :]
    assert_type(y1, Tensor[[2, 4]])

    # Slice indexing with bound updates dimension
    y2 = x[:, :1, :]
    assert_type(y2, Tensor[[2, 1, 4]])

    # Tuple indexing with 1 element keeps dimension
    y3 = x[:, (0,), :]
    assert_type(y3, Tensor[[2, 1, 4]])
