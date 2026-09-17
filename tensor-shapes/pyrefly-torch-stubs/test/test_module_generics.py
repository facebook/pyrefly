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


class Projection[In: IntVar, Out: IntVar](nn.Module):
    weight: Tensor[[Out, In]]

    def __init__(self, in_features: Int[In], out_features: Int[Out]) -> None:
        super().__init__()
        self.weight = torch.randn(out_features, in_features)

    def forward[Batch: IntVar](
        self, value: Tensor[[Batch, In]]
    ) -> Tensor[[Batch, Out]]:
        return torch.matmul(value, self.weight.transpose(0, 1))


class TwoLayer[In: IntVar, Hidden: IntVar, Out: IntVar](nn.Module):
    first: Projection[In, Hidden]
    second: Projection[Hidden, Out]

    def __init__(
        self,
        in_features: Int[In],
        hidden_features: Int[Hidden],
        out_features: Int[Out],
    ) -> None:
        super().__init__()
        self.first = Projection(in_features, hidden_features)
        self.second = Projection(hidden_features, out_features)

    def forward[Batch: IntVar](
        self, value: Tensor[[Batch, In]]
    ) -> Tensor[[Batch, Out]]:
        return self.second(torch.relu(self.first(value)))


class ProjectionConfig[In: IntVar, Out: IntVar]:
    __slots__ = ("in_features", "out_features")

    def __init__(self, in_features: Int[In], out_features: Int[Out]) -> None:
        self.in_features = in_features
        self.out_features = out_features


class ConfiguredProjection[In: IntVar, Out: IntVar](nn.Module):
    projection: Projection[In, Out]

    def __init__(self, config: ProjectionConfig[In, Out]) -> None:
        super().__init__()
        self.projection = Projection(config.in_features, config.out_features)

    def forward[Batch: IntVar](
        self, value: Tensor[[Batch, In]]
    ) -> Tensor[[Batch, Out]]:
        return self.projection(value)


def test_class_and_method_generics() -> None:
    assert_shape(AddVectors()(torch.randn(5), torch.randn(5)).shape, (5,))
    assert_shape(OuterProduct()(torch.randn(3), torch.randn(5)).shape, (3, 5))


def test_generic_submodule_attribute() -> None:
    module = Normalized(6)
    assert_type(module.normalization, RMSNorm[6])
    assert_shape(module.normalization.weight.shape, (6,))


def test_nested_generic_modules() -> None:
    module = TwoLayer(5, 7, 3)
    assert_type(module, TwoLayer[5, 7, 3])
    assert_shape(module(torch.randn(4, 5)).shape, (4, 3))

    configured = ConfiguredProjection(ProjectionConfig(5, 3))
    assert_type(configured, ConfiguredProjection[5, 3])
    assert_shape(configured(torch.randn(4, 5)).shape, (4, 3))


if TYPE_CHECKING:

    def check_linear_specialization[N: IntVar](n: Int[N]) -> None:
        assert_type(nn.Linear[N, 3 * N], type[nn.Linear[N, 3 * N]])
        assert_type(nn.Linear(n, 3 * n), nn.Linear[N, 3 * N])
