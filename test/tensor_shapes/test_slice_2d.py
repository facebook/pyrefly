# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test that slicing a 2D tensor preserves dimensionality"""

from typing import assert_type, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor


def test_slice_first_dim():
    """Test slicing first dimension of 2D tensor"""
    x: Tensor[10, 20] = torch.randn(10, 20)

    # Slice first dimension - should remain 2D
    # Don't annotate the result, just let Pyrefly infer it
    y = x[:5]
    assert_type(y, Tensor[5, 20])

    # Another way to slice
    z = x[0:3]
    assert_type(z, Tensor[3, 20])

    # Index with integer - should reduce to 1D
    w = x[0]
    assert_type(w, Tensor[20])


def test_parameter_slice():
    """Test that Parameter preserves slicing shape"""
    import torch.nn as nn

    x: Tensor[10, 20] = torch.randn(10, 20)
    raw_param = nn.Parameter(x)
    assert_type(raw_param, Tensor[10, 20])
    param: Tensor[10, 20] = nn.Parameter(x)
    assert_type(param, Tensor[10, 20])

    # Slice and wrap in Parameter
    sliced = param[:5]
    assert_type(sliced, Tensor[5, 20])

    # Also verify we can assign to typed variable
    assert_type(param[:5], Tensor[5, 20])

    param2 = nn.Parameter(sliced)
    assert_type(param2, Tensor[5, 20])
