# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test to understand bare Tensor type"""

from typing import assert_type, reveal_type, TYPE_CHECKING

import torch
from shape_extensions import IntVar

if TYPE_CHECKING:
    from shape_extensions import Int
    from torch import Tensor


def test_arange():
    assert_type(torch.arange(5), Tensor[[5]])
    assert_type(torch.arange(2, 7), Tensor[[5]])
    assert_type(torch.arange(0, 6, 2), Tensor[[3]])
    assert_type(torch.arange(2, 7, step=1), Tensor[[5]])


def test_arange_nondivisible_ranges():
    assert_type(torch.arange(0, 5, 2), Tensor[[3]])
    assert_type(torch.arange(5, 0, -2), Tensor[[3]])
    assert_type(torch.arange(6, 0, -2), Tensor[[3]])
    reveal_type(torch.arange(4, 4, 2))  # revealed type: Tensor[[0]]
    reveal_type(torch.arange(4, 4, -2))  # revealed type: Tensor[[0]]


def test_arange_keyword_options():
    assert_type(torch.arange(5, dtype=None, device=None), Tensor[[5]])
    assert_type(torch.arange(0, 6, 2, dtype=None, device=None), Tensor[[3]])


def test_arange_symbolic[N: IntVar](t: Int[N]):
    x = torch.arange(0, t)
    assert_type(x, Tensor[[N]])


def test_arange_single_arg[N: IntVar](t: Int[N]):
    x = torch.arange(t)
    assert_type(x, Tensor[[N]])


# A symbolic bound keeps a truncating extent, which is exact when the step divides the range.
def test_arange_symbolic_bound_literal_step[N: IntVar](t: Int[N]):
    assert_type(torch.arange(0, t, 2), Tensor[[N // 2]])
    assert_type(torch.arange(1, t), Tensor[[N - 1]])
    assert_type(torch.arange(t, 10), Tensor[[10 - N]])


# A symbolic step becomes an unknown Flag value, so its result is gradual.
def test_arange_symbolic_step[N: IntVar, M: IntVar](t: Int[N], step: Int[M]):
    assert_type(torch.arange(0, t, step), Tensor[[int]])


def test_arange_gradual(end: int, start: int, step: int):
    assert_type(torch.arange(end), Tensor[[int]])
    assert_type(torch.arange(start, end), Tensor[[int]])
    assert_type(torch.arange(start, end, step), Tensor[[int]])


def test_arange_arithmetic_overflow():
    assert_type(torch.arange(-9223372036854775808, 9223372036854775807), Tensor[[int]])
    assert_type(torch.arange(1, 0, -9223372036854775808), Tensor[[int]])
