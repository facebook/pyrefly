# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type, TYPE_CHECKING

import torch
from shape_extensions import assert_shape, IntVar
from torch import Tensor


def test_unary_math_shapes() -> None:
    x = torch.ones((2, 3))

    for result in (
        torch.abs(x),
        torch.neg(x),
        torch.floor(x),
        torch.ceil(x),
        torch.round(x),
        torch.sin(x),
        torch.cos(x),
        torch.tan(x),
        torch.exp(x),
        torch.log(x),
        torch.sqrt(x),
        torch.tanh(x),
    ):
        assert_shape(result.shape, (2, 3))

    assert_shape(x.sin().shape, (2, 3))


def test_logical_activation_and_clamp_shapes() -> None:
    x = torch.ones((2, 3, 4))

    assert_shape(torch.logical_not(x).shape, (2, 3, 4))
    assert_shape(torch.relu(x).shape, (2, 3, 4))
    assert_shape(x.relu().shape, (2, 3, 4))
    assert_shape(torch.clamp(x, min=-1.0, max=1.0).shape, (2, 3, 4))
    assert_shape(torch.clip(x, min=-1.0, max=1.0).shape, (2, 3, 4))
    assert_shape(x.clamp(min=-1.0, max=1.0).shape, (2, 3, 4))


if TYPE_CHECKING:

    def check_symbolic_unary_shapes[N: IntVar, M: IntVar](x: Tensor[[N, M]]) -> None:
        assert_type(torch.abs(x), Tensor[[N, M]])
        assert_type(torch.neg(x), Tensor[[N, M]])
        assert_type(torch.sin(x), Tensor[[N, M]])
        assert_type(x.sin(), Tensor[[N, M]])
        assert_type(torch.logical_not(x), Tensor[[N, M]])
        assert_type(torch.relu(x), Tensor[[N, M]])
        assert_type(x.relu(), Tensor[[N, M]])
        assert_type(torch.clamp(x, min=-1.0, max=1.0), Tensor[[N, M]])
        assert_type(torch.clip(x, min=-1.0, max=1.0), Tensor[[N, M]])
        assert_type(x.clamp(min=-1.0, max=1.0), Tensor[[N, M]])
