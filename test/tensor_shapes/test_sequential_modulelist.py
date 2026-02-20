# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Test nn.Sequential and nn.ModuleList with symbolic dimensions

Note: These containers may have limitations because they're type-erased
(they hold generic Module objects, not specifically typed ones)
"""

from typing import assert_type, TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from torch import Tensor
    from torch_shapes import Dim

# ============================================================================
# Helper Modules for Testing
# ============================================================================


class LinearLayer[N, M](nn.Module):
    """Reusable linear layer for testing"""

    weight: Tensor[M, N]

    def __init__(self, in_features: Dim[N], out_features: Dim[M]):
        super().__init__()
        # Now M and N are bound to runtime values via Literal types
        self.weight = torch.randn(out_features, in_features)

    def forward[B](self, x: Tensor[B, N]) -> Tensor[B, M]:
        weight_t: Tensor[N, M] = self.weight.transpose(0, 1)
        return torch.matmul(x, weight_t)


class ReLULayer(nn.Module):
    """Simple ReLU wrapper (truly shape-preserving - works with any dimension)"""

    def forward[B, N](self, x: Tensor[B, N]) -> Tensor[B, N]:
        return torch.relu(x)


# ============================================================================
# Test 1: Basic Sequential Construction
# ============================================================================


def test_sequential_construction():
    """Test that we can construct Sequential with modules"""
    layer1 = LinearLayer(5, 10)
    layer2 = LinearLayer(10, 10)

    # Can we create Sequential?
    seq = nn.Sequential(layer1, layer2)

    # What happens when we call it? (Sequential is callable via __call__)
    x: Tensor[16, 5] = torch.randn(16, 5)
    y = seq(x)

    # Check what type we get back
    assert_type(y, Tensor)


# ============================================================================
# Test 2: Manual Sequential Forwarding
# ============================================================================


class ManualSequential[N, M, K](nn.Module):
    """Manually implement sequential forwarding to show expected behavior"""

    layer1: LinearLayer[N, M]
    layer2: LinearLayer[M, K]

    def __init__(
        self,
        in_features: Dim[N],
        hidden_features: Dim[M],
        out_features: Dim[K],
    ):
        super().__init__()
        # Now N, M, K are bound to runtime values via Literal types
        self.layer1 = LinearLayer(in_features, hidden_features)
        self.layer2 = LinearLayer(hidden_features, out_features)

    def forward[B](self, x: Tensor[B, N]):
        # This is what Sequential *should* do type-wise
        # Note: layer outputs have concrete dimensions (Tensor[B, 10])
        h = self.layer1(x)
        y = self.layer2(h)
        return y


def test_manual_sequential():
    """Test manual sequential as baseline"""
    model = ManualSequential(5, 10, 10)

    assert_type(model, ManualSequential[5, 10, 10])
    assert_type(model.layer1, LinearLayer[5, 10])
    assert_type(model.layer2, LinearLayer[10, 10])

    x: Tensor[16, 5] = torch.randn(16, 5)
    y = model(x)
    assert_type(y, Tensor[16, 10])
    assert_type(y, Tensor[16, 10])


class TypedSequential[N, M, K](nn.Module):
    """Sequential that takes layer types instead of dimension literals"""

    layer1: LinearLayer[N, M]
    layer2: LinearLayer[M, K]

    def __init__(self, layer1: LinearLayer[N, M], layer2: LinearLayer[M, K]):
        super().__init__()
        # Type parameters N, M, K should be inferred from the layer types
        self.layer1 = layer1
        self.layer2 = layer2

    def forward[B](self, x: Tensor[B, N]):
        h = self.layer1(x)
        y = self.layer2(h)
        return y


def test_typed_sequential():
    """Test sequential with layer types as parameters"""
    layer1 = LinearLayer(5, 10)
    layer2 = LinearLayer(10, 10)
    model = TypedSequential(layer1, layer2)

    # Type parameters should be inferred: N=5, M=10, K=10
    assert_type(model, TypedSequential[5, 10, 10])

    assert_type(model.layer1, LinearLayer[5, 10])
    assert_type(model.layer2, LinearLayer[10, 10])

    x: Tensor[16, 5] = torch.randn(16, 5)
    y = model(x)
    assert_type(y, Tensor[16, 10])


# ============================================================================
# Test 3: Basic ModuleList Construction
# ============================================================================


def test_modulelist_construction():
    """Test that we can construct ModuleList"""
    # ModuleList without initial modules
    module_list: nn.ModuleList[LinearLayer] = nn.ModuleList()

    # Add modules one by one
    module_list.append(LinearLayer(5, 10))
    module_list.append(LinearLayer(10, 10))
    assert_type(module_list[0], LinearLayer)
    assert_type(module_list[1], LinearLayer)
