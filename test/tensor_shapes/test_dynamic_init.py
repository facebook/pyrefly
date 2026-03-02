# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Test dynamic initialization patterns with symbolic dimensions

Key question: How do runtime dimension values (passed to __init__)
connect to generic type parameters?
"""

from typing import assert_type, TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from torch import Tensor
    from torch_shapes import Dim

# ============================================================================
# Test 1: Dynamic Initialization with Literal Parameters
# ============================================================================


class DynamicLinear[N, M](nn.Module):
    """Linear layer with runtime dimension parameters"""

    weight: Tensor[M, N]

    def __init__(self, in_features: Dim[N], out_features: Dim[M]):
        super().__init__()
        # Now N and M are bound via Literal params, so we can create tensors
        self.weight = torch.randn(out_features, in_features)

    def forward[B](self, x: Tensor[B, N]) -> Tensor[B, M]:
        weight_t: Tensor[N, M] = self.weight.transpose(0, 1)
        return torch.matmul(x, weight_t)


def test_dynamic_init():
    """Test dynamic initialization with Literal parameters"""
    layer = DynamicLinear(64, 128)

    x: Tensor[32, 64] = torch.randn(32, 64)
    y = layer(x)
    assert_type(y, Tensor[32, 128])


# ============================================================================
# Test 2: Multiple Instances with Different Dimensions
# ============================================================================


def test_multiple_instances():
    """
    Test that the same generic class can be used with different dimensions

    This is the key power of generics: one definition, many uses
    """
    # Instance 1: 64 -> 128
    layer1 = DynamicLinear(64, 128)

    x1: Tensor[8, 64] = torch.randn(8, 64)
    y1 = layer1(x1)
    assert_type(y1, Tensor[8, 128])

    # Instance 2: 256 -> 512 (different dimensions!)
    layer2 = DynamicLinear(256, 512)

    x2: Tensor[16, 256] = torch.randn(16, 256)
    y2 = layer2(x2)
    assert_type(y2, Tensor[16, 512])

    # Instance 3: 32 -> 64 (yet another set of dimensions)
    layer3 = DynamicLinear(32, 64)

    x3: Tensor[4, 32] = torch.randn(4, 32)
    y3 = layer3(x3)
    assert_type(y3, Tensor[4, 64])


# ============================================================================
# Test 4: Configuration Object Pattern (Now Works!)
# ============================================================================


class ModelConfig[N, M, K]:
    """Generic configuration with dimension type parameters"""

    input_dim: Dim[N]
    hidden_dim: Dim[M]
    output_dim: Dim[K]

    def __init__(self, input_dim: Dim[N], hidden_dim: Dim[M], output_dim: Dim[K]):
        # Store Literal parameters - they retain their types when assigned to typed fields
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim


class ConfiguredModel[N, M, K](nn.Module):
    """Model configured via generic config object"""

    w1: Tensor[M, N]
    w2: Tensor[K, M]

    def __init__(self, config: ModelConfig[N, M, K]):
        super().__init__()
        # Now config.hidden_dim has type Dim[M], config.input_dim has type Dim[N]
        # So torch.randn() gets Literal arguments and can infer shapes!
        self.w1 = torch.randn(config.hidden_dim, config.input_dim)
        self.w2 = torch.randn(config.output_dim, config.hidden_dim)

    def forward[B](self, x: Tensor[B, N]) -> Tensor[B, K]:
        w1_t: Tensor[N, M] = self.w1.transpose(0, 1)
        h: Tensor[B, M] = torch.matmul(x, w1_t)
        h_relu: Tensor[B, M] = torch.relu(h)

        w2_t: Tensor[M, K] = self.w2.transpose(0, 1)
        y: Tensor[B, K] = torch.matmul(h_relu, w2_t)
        return y


def test_config_pattern():
    """Test configuration object pattern"""
    # Create config with specific dimensions
    config = ModelConfig(64, 128, 32)

    # Create model from config
    model = ConfiguredModel(config)

    x: Tensor[16, 64] = torch.randn(16, 64)
    y = model(x)
    assert_type(y, Tensor[16, 32])


def test_config_multiple_instances():
    """Test multiple instances with different configs"""
    # Instance 1: 64 -> 128 -> 32
    config1 = ModelConfig(64, 128, 32)
    model1 = ConfiguredModel(config1)

    x1: Tensor[8, 64] = torch.randn(8, 64)
    y1 = model1(x1)
    assert_type(y1, Tensor[8, 32])

    # Instance 2: 256 -> 512 -> 128 (different dimensions!)
    config2 = ModelConfig(256, 512, 128)
    model2 = ConfiguredModel(config2)

    x2: Tensor[16, 256] = torch.randn(16, 256)
    y2 = model2(x2)
    assert_type(y2, Tensor[16, 128])

    # Instance 3: 32 -> 64 -> 16 (yet another set)
    config3 = ModelConfig(32, 64, 16)
    model3 = ConfiguredModel(config3)

    x3: Tensor[4, 32] = torch.randn(4, 32)
    y3 = model3(x3)
    assert_type(y3, Tensor[4, 16])


# ============================================================================
# Test 3: Summary - Solution with Literal Parameters
# ============================================================================

# SOLUTION: Use Dim[N] parameters to bind type variables!
#
# Previously these were LIMITATIONS, but now they work:
#
# 1. torch.randn() with Literal parameters works! ✅
#    def __init__(self, in_features: Dim[N], out_features: Dim[M]):
#        self.weight = torch.randn(out_features, in_features)
#
# 2. Runtime __init__ parameters CAN connect to generic type params via Literal ✅
#    The Dim[N] annotation binds the type variable N to the runtime value
#
# 3. Multiple instances with different dimensions work! ✅
#    layer1 = DynamicLinear(64, 128)
#    layer2 = DynamicLinear(256, 512)
#
# Key insight: Dim[N] creates a bridge between compile-time type variables
# and runtime dimension values. When you pass a literal value to __init__,
# the type variable becomes bound to that concrete dimension.
