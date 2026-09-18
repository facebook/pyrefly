# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type, TYPE_CHECKING

import torch
from shape_extensions import assert_raises, assert_shape
from torch import Tensor


def test_item_scalar() -> None:
    scalar = torch.tensor(2.0)
    assert_shape(scalar.shape, ())
    assert_type(scalar.item(), float | int)
    assert scalar.item() == 2.0


def test_item_rejects_multiple_elements() -> None:
    vector = torch.ones(2)
    assert_shape(vector.shape, (2,))

    with assert_raises(RuntimeError):
        vector.item()  # E: not assignable

    matrix = torch.ones((5, 7))
    assert_shape(matrix.shape, (5, 7))

    with assert_raises(RuntimeError):
        matrix.item()  # E: not assignable


def test_item_accepts_single_element_non_scalar() -> None:
    vector = torch.ones(1)
    assert_shape(vector.shape, (1,))
    # TODO: BUG: Accept any tensor with exactly one element, regardless of rank.
    vector.item()  # E: not assignable


if TYPE_CHECKING:

    def check_gradual_tensor(x: Tensor) -> None:
        assert_type(x.item(), float | int)
