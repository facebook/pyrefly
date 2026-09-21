# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type, TYPE_CHECKING

import torch
import torch.nn.functional as F
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
        x.asin(),
        x.acos(),
        x.atan(),
        x.sinh(),
        x.cosh(),
        x.exp2(),
        x.expm1(),
        x.log2(),
        x.log10(),
        x.log1p(),
        x.rsqrt(),
        x.square(),
        x.reciprocal(),
        x.sign(),
        x.sigmoid(),
        x.trunc(),
        x.frac(),
        x.erfinv(),
        x.lgamma(),
        x.digamma(),
        x.polygamma(2),
        x.asinh(),
        x.acosh(),
        x.atanh(),
        x.deg2rad(),
        x.rad2deg(),
    ):
        assert_shape(result.shape, (2, 3))

    assert_shape(x.sin().shape, (2, 3))


def test_logical_activation_and_clamp_shapes() -> None:
    x = torch.ones((2, 3, 4))

    assert_shape(torch.logical_not(x).shape, (2, 3, 4))
    assert_shape(torch.relu(x).shape, (2, 3, 4))
    assert_shape(x.relu().shape, (2, 3, 4))
    for result in (
        F.relu(x),
        F.gelu(x),
        F.silu(x),
        F.selu(x),
        F.elu(x),
        F.leaky_relu(x),
        F.relu6(x),
        F.softplus(x),
        F.softsign(x),
        F.hardtanh(x),
        F.hardsigmoid(x),
        F.hardswish(x),
        F.sigmoid(x),
        F.tanh(x),
        F.mish(x),
        F.prelu(x, torch.ones(3)),
        F.rrelu(x),
        F.celu(x),
    ):
        assert_shape(result.shape, (2, 3, 4))
    assert_shape(torch.clamp(x, min=-1.0, max=1.0).shape, (2, 3, 4))
    assert_shape(torch.clip(x, min=-1.0, max=1.0).shape, (2, 3, 4))
    assert_shape(x.clamp(min=-1.0, max=1.0).shape, (2, 3, 4))
    assert_shape(x.isnan().shape, (2, 3, 4))
    assert_shape(x.isinf().shape, (2, 3, 4))
    assert_shape(x.isfinite().shape, (2, 3, 4))
    assert_shape(x.isreal().shape, (2, 3, 4))
    assert_shape(x.isposinf().shape, (2, 3, 4))
    assert_shape(x.isneginf().shape, (2, 3, 4))


if TYPE_CHECKING:

    def check_symbolic_unary_shapes[N: IntVar, M: IntVar](
        x: Tensor[[N, M]], weight: Tensor
    ) -> None:
        assert_type(torch.abs(x), Tensor[[N, M]])
        assert_type(torch.neg(x), Tensor[[N, M]])
        assert_type(torch.sin(x), Tensor[[N, M]])
        assert_type(x.sin(), Tensor[[N, M]])
        assert_type(torch.logical_not(x), Tensor[[N, M]])
        assert_type(torch.relu(x), Tensor[[N, M]])
        assert_type(x.relu(), Tensor[[N, M]])
        assert_type(F.relu(x), Tensor[[N, M]])
        assert_type(F.gelu(x), Tensor[[N, M]])
        assert_type(F.silu(x), Tensor[[N, M]])
        assert_type(F.selu(x), Tensor[[N, M]])
        assert_type(F.elu(x), Tensor[[N, M]])
        assert_type(F.leaky_relu(x), Tensor[[N, M]])
        assert_type(F.relu6(x), Tensor[[N, M]])
        assert_type(F.softplus(x), Tensor[[N, M]])
        assert_type(F.softsign(x), Tensor[[N, M]])
        assert_type(F.hardtanh(x), Tensor[[N, M]])
        assert_type(F.hardsigmoid(x), Tensor[[N, M]])
        assert_type(F.hardswish(x), Tensor[[N, M]])
        assert_type(F.sigmoid(x), Tensor[[N, M]])
        assert_type(F.tanh(x), Tensor[[N, M]])
        assert_type(F.mish(x), Tensor[[N, M]])
        assert_type(F.prelu(x, weight), Tensor[[N, M]])
        assert_type(F.rrelu(x), Tensor[[N, M]])
        assert_type(F.celu(x), Tensor[[N, M]])
        assert_type(torch.clamp(x, min=-1.0, max=1.0), Tensor[[N, M]])
        assert_type(torch.clip(x, min=-1.0, max=1.0), Tensor[[N, M]])
        assert_type(x.clamp(min=-1.0, max=1.0), Tensor[[N, M]])
