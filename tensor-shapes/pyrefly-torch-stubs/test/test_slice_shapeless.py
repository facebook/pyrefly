# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test to understand bare Tensor type"""

from typing import assert_type, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor


def test_bare_tensor_subscript(x: Tensor):
    """Test simple subscript on bare Tensor"""
    y = x[0]  # Single integer index
    assert_type(y, Tensor)


def test_bare_tensor_slice(x: Tensor):
    """Test slice on bare Tensor"""
    z = x[:]  # Single slice
    assert_type(z, Tensor)


def test_bare_tensor_subscripts(idx: Tensor):
    x = idx[:, -1, :]
    assert_type(x, Tensor)
