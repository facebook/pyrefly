# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type

import torch
from shape_extensions import assert_shape, IntTuple
from torch import Tensor


def tensor_identity[Shape: IntTuple](tensor: Tensor[Shape]) -> Tensor[Shape]:
    return tensor


def tuple_identity[*Elements](values: tuple[*Elements]) -> tuple[*Elements]:
    return values


class Box[Value]:
    __slots__ = ("value",)

    def __init__(self, value: Value) -> None:
        self.value = value


def test_generic_round_trips() -> None:
    tensor = tensor_identity(torch.randn((10, 20)))
    assert_type(tensor, Tensor[[10, 20]])
    assert_shape(tensor.shape, (10, 20))
    values = tuple_identity((Box(0), Box("value")))
    assert_type(values, tuple[Box[int], Box[str]])
    assert values[0].value == 0
