# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test view/reshape validation errors"""

from typing import assert_type, reveal_type, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor


def test_multiple_minus_ones():
    """Multiple -1s are rejected."""
    x: Tensor[[10, 20]] = torch.randn(10, 20)
    # E: can only specify one unknown dimension as -1
    y = x.view(-1, -1)
    assert_type(y, Tensor)


def test_incompatible_shape():
    """Incompatible shape with literal dimensions is rejected."""
    x: Tensor[[10, 20]] = torch.randn(10, 20)  # 200 elements
    # E: could not infer size for dimension -1
    y = x.view(3, -1)
    assert_type(y, Tensor)


def test_invalid_dimension_value():
    """Invalid dimension values like -2 and -3 are rejected."""
    x: Tensor[[100]] = torch.randn(100)
    # E: invalid negative dimension value (only -1 is allowed)
    y = x.view(-2, 10)
    assert_type(y, Tensor)


def test_zero_dimension_with_nonempty_input():
    """A zero target cannot hold a nonempty input."""
    x: Tensor[[100]] = torch.randn(100)
    # E: reshape target element count does not match the input
    y = x.view(0, 1)
    assert_type(y, Tensor)


def test_mismatched_element_count():
    """A fully specified target whose element count differs is rejected."""
    x: Tensor[[6]] = torch.randn(6)
    # E: reshape target element count does not match the input
    y = x.reshape(4, 2)
    assert_type(y, Tensor)
    # E: reshape target element count does not match the input
    torch.reshape(x, (2, 2))


def test_zero_sized_inference():
    empty = torch.empty(0, 3)
    # E: revealed type: Tensor[[0]]
    reveal_type(empty.reshape(-1))
    # E: could not infer size for dimension -1
    empty.reshape(0, -1)
