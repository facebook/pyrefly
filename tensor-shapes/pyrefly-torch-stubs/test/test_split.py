# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test to understand bare Tensor type"""

from typing import assert_type, TYPE_CHECKING

import torch
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


def test_split_nondivisible(x: Tensor[[4, 5, 17]]):
    assert_type(
        x.split(6, dim=2),
        tuple[Tensor[[4, 5, 6]], Tensor[[4, 5, 6]], Tensor[[4, 5, 5]]],
    )
    assert_type(
        torch.split(x, 6, dim=2),
        tuple[Tensor[[4, 5, 6]], Tensor[[4, 5, 6]], Tensor[[4, 5, 5]]],
    )


def test_split_negative_dim(x: Tensor[[4, 5, 17]]):
    assert_type(
        x.split(6, dim=-1),
        tuple[Tensor[[4, 5, 6]], Tensor[[4, 5, 6]], Tensor[[4, 5, 5]]],
    )
    assert_type(
        torch.split(x, (1, 4), dim=-2),
        tuple[Tensor[[4, 1, 17]], Tensor[[4, 4, 17]]],
    )


def test_split_sections(x: Tensor[[4, 9, 5]]):
    assert_type(
        x.split((2, 3, 4), dim=1),
        tuple[Tensor[[4, 2, 5]], Tensor[[4, 3, 5]], Tensor[[4, 4, 5]]],
    )
    assert_type(
        torch.split(x, (2, 3, 4), dim=1),
        tuple[Tensor[[4, 2, 5]], Tensor[[4, 3, 5]], Tensor[[4, 4, 5]]],
    )


# A list cannot carry its element values to the type level, so the documented list
# spelling stays valid but only recovers a gradual tuple of tensors.
def test_split_list_sections(x: Tensor[[4, 9, 5]]):
    assert_type(x.split([2, 3, 4], dim=1), tuple[Tensor, ...])
    assert_type(torch.split(x, [2, 3, 4], dim=1), tuple[Tensor, ...])


def test_split_symbolic[B: IntVar, T: IntVar, N: IntVar](
    x: Tensor[[B, T, (3 * N)]], n: Int[N]
):
    y = x.split(n, dim=2)
    assert_type(y, tuple[Tensor[[B, T, N]], Tensor[[B, T, N]], Tensor[[B, T, N]]])
    assert_type(
        torch.split(x, n, dim=2),
        tuple[Tensor[[B, T, N]], Tensor[[B, T, N]], Tensor[[B, T, N]]],
    )


def test_split_symbolic_sections[B: IntVar, N: IntVar](
    x: Tensor[[B, (3 * N)]], n: Int[N]
):
    assert_type(
        x.split((n, n, n), dim=1),
        tuple[Tensor[[B, N]], Tensor[[B, N]], Tensor[[B, N]]],
    )
    assert_type(
        torch.split(x, (n, n, n), dim=1),
        tuple[Tensor[[B, N]], Tensor[[B, N]], Tensor[[B, N]]],
    )


def test_split_mixed[B: IntVar, T: IntVar, N: IntVar](
    x: Tensor[[B, T, (3 * N)]],
):
    y = x.split(3, dim=2)
    assert_type(y, tuple[Tensor[[B, T, 3]], ...])
    assert_type(torch.split(x, 3, dim=2), tuple[Tensor[[B, T, 3]], ...])


def test_split_symbolic_remainder[B: IntVar, T: IntVar, N: IntVar](
    x: Tensor[[B, T, N]],
):
    assert_type(x.split(3, dim=2), tuple[Tensor[[B, T, int]], ...])
    assert_type(torch.split(x, 3, dim=2), tuple[Tensor[[B, T, int]], ...])


def test_split_gradual_size(x: Tensor[[4, 5, 18]], split_size: int):
    assert_type(x.split(split_size, dim=2), tuple[Tensor[[4, 5, int]], ...])
    assert_type(torch.split(x, split_size, dim=2), tuple[Tensor[[4, 5, int]], ...])
