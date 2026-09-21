# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type, TYPE_CHECKING

import torch
import torch.nn as nn
from shape_extensions import assert_shape, Int, IntVar
from torch import Tensor


class AddVectors[N: IntVar]:
    def __call__(self, left: Tensor[[N]], right: Tensor[[N]]) -> Tensor[[N]]:
        return left + right


class OuterProduct[N: IntVar]:
    def __call__[M: IntVar](
        self, left: Tensor[[M]], right: Tensor[[N]]
    ) -> Tensor[[M, N]]:
        return torch.einsum("i,j->ij", left, right)


class RMSNorm[D: IntVar](nn.Module):
    def __init__(self, dim: Int[D]) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))


class Normalized[D: IntVar](nn.Module):
    normalization: RMSNorm[D]

    def __init__(self, dim: Int[D]) -> None:
        super().__init__()
        self.normalization = RMSNorm(dim)
        assert_type(self.normalization, RMSNorm[D])


def test_class_and_method_generics() -> None:
    assert_shape(AddVectors()(torch.randn(5), torch.randn(5)).shape, (5,))
    assert_shape(OuterProduct()(torch.randn(3), torch.randn(5)).shape, (3, 5))


def test_generic_submodule_attribute() -> None:
    module = Normalized(6)
    assert_type(module.normalization, RMSNorm[6])
    assert_shape(module.normalization.weight.shape, (6,))


if TYPE_CHECKING:

    def check_linear_specialization[N: IntVar](n: Int[N]) -> None:
        assert_type(nn.Linear[N, 3 * N], type[nn.Linear[N, 3 * N]])
        assert_type(nn.Linear(n, 3 * n), nn.Linear[N, 3 * N])
