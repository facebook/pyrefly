# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test that .size() works on tensors with variadic shapes"""

from typing import assert_type, Literal

import torch
import torch.nn as nn
from shape_extensions import Elements, Int, IntTuple, IntVar
from torch import Tensor


def test_size_on_variadic_from_linear():
    """Linear returns Tensor[[*Elements[Bs], OUT]] which is variadic when Bs is unknown"""
    linear = nn.Linear(10, 20)
    x: Tensor = torch.randn(5, 10)  # unannotated shape

    # Linear forward returns variadic shape Tensor[[*Elements[Bs], OUT]]
    output = linear(x)

    # .size(-1) reads the known suffix; other indices need the unknown rank.
    last_dim = output.size(-1)
    assert_type(last_dim, Int[20])

    first_dim = output.size(0)
    assert_type(first_dim, Int[int])


def test_size_on_concrete_shape():
    """Verify that concrete shapes still work correctly"""
    x: Tensor[[5, 10]] = torch.randn(5, 10)

    # .size() returns tuple with specific dimensions
    # For concrete shapes, we return the actual tuple of dimensions
    # This should be revealed as tuple with actual shape info
    assert_type(x.size(), tuple[Literal[5], Literal[10]])

    # .size(i) returns the actual dimension
    assert_type(x.size(0), Literal[5])
    assert_type(x.size(1), Literal[10])


def test_size_last_dim_on_symbolic_rank[Batch: IntTuple, Out: IntVar](
    x: Tensor[[*Elements[Batch], Out]],
) -> None:
    """`size(-1)` reads the known suffix even though the total rank is symbolic."""
    assert_type(x.size(-1), Int[Out])
