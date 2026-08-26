# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test item() rank validation."""

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor


def test_item_on_1d_tensor() -> None:
    x: Tensor[[10]] = torch.randn(10)
    x.item()  # E: not assignable


def test_item_on_2d_tensor() -> None:
    x: Tensor[[5, 7]] = torch.randn(5, 7)
    x.item()  # E: not assignable


def test_item_on_one_element_vector() -> None:
    # Known limitation: at runtime item() accepts any tensor holding exactly one
    # element, but the rank-zero `Tensor[[]]` receiver only recognizes scalars, so
    # this valid call is rejected.
    x: Tensor[[1]] = torch.randn(1)
    x.item()  # E: not assignable
