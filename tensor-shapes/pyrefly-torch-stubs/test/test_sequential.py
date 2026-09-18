# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type, TYPE_CHECKING

import torch
import torch.nn as nn
from shape_extensions import assert_shape, IntVar
from torch import Tensor


class DoubleWidth(nn.Module):
    def forward[N: IntVar, M: IntVar](
        self, tensor: Tensor[[N, M]]
    ) -> Tensor[[N, 2 * M]]:
        return torch.cat((tensor, tensor), dim=1)


def test_sequential_builtin_modules() -> None:
    model = nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3, padding=1),
        nn.BatchNorm2d(8),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )
    assert_shape(model(torch.randn((2, 3, 16, 20))).shape, (2, 8, 8, 10))


def test_sequential_shape_changing_modules() -> None:
    model = nn.Sequential(
        nn.ReflectionPad2d(1),
        nn.Conv2d(3, 4, kernel_size=3),
        nn.Upsample(scale_factor=2),
    )
    assert_shape(model(torch.randn((2, 3, 8, 10))).shape, (2, 4, 16, 20))


def test_sequential_custom_module() -> None:
    model = nn.Sequential(DoubleWidth(), nn.ReLU())
    assert_shape(model(torch.randn((3, 5))).shape, (3, 10))


if TYPE_CHECKING:

    def check_symbolic_sequential[N: IntVar, M: IntVar](
        tensor: Tensor[[N, M]], module: DoubleWidth
    ) -> None:
        assert_type(nn.Sequential(module, nn.ReLU())(tensor), Tensor[[N, 2 * M]])
