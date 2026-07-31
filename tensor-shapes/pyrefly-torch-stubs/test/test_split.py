# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test to understand bare Tensor type"""

from typing import assert_type, TYPE_CHECKING

from shape_extensions import IntVar

if TYPE_CHECKING:
    from shape_extensions import Int
    from torch import Tensor


def test_split(x: Tensor[[4, 5, 18]]):
    y = x.split(6, dim=2)
    assert_type(y, tuple[Tensor[[4, 5, 6]], Tensor[[4, 5, 6]], Tensor[[4, 5, 6]]])
    a, b, c = y
    assert_type(a, Tensor[[4, 5, 6]])
    assert_type(b, Tensor[[4, 5, 6]])
    assert_type(c, Tensor[[4, 5, 6]])


def test_split_symbolic[B: IntVar, T: IntVar, N: IntVar](
    x: Tensor[[B, T, (3 * N)]], n: Int[N]
):
    y = x.split(n, dim=2)
    assert_type(y, tuple[Tensor[[B, T, N]], Tensor[[B, T, N]], Tensor[[B, T, N]]])


def test_split_mixed[B: IntVar, T: IntVar, N: IntVar](
    x: Tensor[[B, T, (3 * N)]],
):
    y = x.split(3, dim=2)
    assert_type(y, tuple[Tensor[[B, T, 3]], ...])
