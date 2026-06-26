# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test matmul with mixed shaped and shapeless tensors"""

from typing import assert_type, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor


def test_shaped_matmul_shapeless[N](a: Tensor[N, N], b: Tensor):
    """Shaped tensor @ shapeless tensor"""
    result1 = a.matmul(b)
    assert_type(result1, Tensor)

    result2 = b.matmul(a)
    assert_type(result2, Tensor)
