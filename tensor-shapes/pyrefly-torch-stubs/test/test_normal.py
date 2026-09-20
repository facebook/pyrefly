# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type, TYPE_CHECKING

import torch
from shape_extensions import assert_raises, assert_shape, Int, IntTuple, IntVar
from torch import Tensor


def test_normal_tensor_parameter_shapes() -> None:
    mean = torch.zeros((2, 3))
    std = torch.ones((2, 3))
    assert_shape(torch.normal(mean, std).shape, (2, 3))
    assert_shape(torch.normal(mean, 0.5).shape, (2, 3))
    assert_shape(torch.normal(0.0, std).shape, (2, 3))

    broadcast_mean = torch.zeros((2, 1))
    broadcast_std = torch.ones((1, 3))
    assert_shape(torch.normal(broadcast_mean, broadcast_std).shape, (2, 3))


def test_normal_size_shapes() -> None:
    assert_shape(torch.normal(0.0, 1.0, size=(3, 4)).shape, (3, 4))
    assert_shape(torch.normal(0.0, 1.0, size=()).shape, ())


def test_normal_rejects_different_element_counts() -> None:
    mean = torch.zeros((2, 3))
    assert_shape(torch.normal(mean, torch.ones((2, 3))).shape, (2, 3))

    with assert_raises(RuntimeError):
        # E: Cannot broadcast dimension
        torch.normal(mean, torch.ones(5))


if TYPE_CHECKING:

    def check_symbolic_normal[N: IntVar, M: IntVar](
        mean: Tensor[[N, M]], std: Tensor, n: Int[N], size: tuple[int, ...]
    ) -> None:
        assert_type(torch.normal(mean, std), Tensor)
        assert_type(torch.normal(mean, 0.5), Tensor[[N, M]])
        assert_type(torch.normal(0.0, mean), Tensor[[N, M]])
        assert_type(torch.normal(0.0, 1.0, size=(n, 3)), Tensor[[N, 3]])
        assert_type(torch.normal(0.0, 1.0, size=size), Tensor[IntTuple])

    def check_symbolic_normal_broadcast[N: IntVar, M: IntVar](
        mean: Tensor[[N, 1]], std: Tensor[[1, M]]
    ) -> None:
        assert_type(torch.normal(mean, std), Tensor[[N, M]])
