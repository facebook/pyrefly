# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Test nested nn.Module patterns
Critical for real PyTorch code organization
"""

from typing import assert_type, TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from torch import Tensor
    from torch_shapes import Dim

# ============================================================================
# Test 1: Basic Nested Modules
# ============================================================================


class LinearLayer[N, M](nn.Module):
    """Basic linear layer (reusable component)"""

    weight: Tensor[M, N]

    def __init__(self, in_features: Dim[N], out_features: Dim[M]):
        super().__init__()
        self.weight = torch.randn(out_features, in_features)

    def forward[B](self, x: Tensor[B, N]) -> Tensor[B, M]:
        weight_t: Tensor[N, M] = self.weight.transpose(0, 1)
        return torch.matmul(x, weight_t)


class TwoLayerMLP[N, M, K](nn.Module):
    """MLP composed of nested LinearLayer modules"""

    # Can we declare typed attributes?
    layer1: LinearLayer[N, M]
    layer2: LinearLayer[M, K]

    def __init__(
        self,
        in_features: Dim[N],
        hidden_features: Dim[M],
        out_features: Dim[K],
    ):
        super().__init__()
        # Can we initialize typed modules?
        self.layer1 = LinearLayer(in_features, hidden_features)
        self.layer2 = LinearLayer(hidden_features, out_features)

    def forward[B](self, x: Tensor[B, N]) -> Tensor[B, K]:
        # Does calling nested module preserve types?
        h: Tensor[B, M] = self.layer1(x)
        h_relu: Tensor[B, M] = torch.relu(h)
        y: Tensor[B, K] = self.layer2(h_relu)
        return y


def test_basic_nested_modules():
    """Test basic module composition"""
    mlp = TwoLayerMLP(5, 10, 10)

    x: Tensor[16, 5] = torch.randn(16, 5)
    y = mlp(x)
    assert_type(y, Tensor[16, 10])


# ============================================================================
# Test 2: Nested Modules Without Type Annotations
# ============================================================================


class SimpleMLP[N, M, K](nn.Module):
    """MLP without explicit attribute type annotations"""

    def __init__(
        self,
        in_features: Dim[N],
        hidden_features: Dim[M],
        out_features: Dim[K],
    ):
        super().__init__()
        # Just assign, no type annotation
        self.layer1 = LinearLayer(in_features, hidden_features)
        self.layer2 = LinearLayer(hidden_features, out_features)

    def forward[B](self, x: Tensor[B, N]) -> Tensor[B, K]:
        h: Tensor[B, M] = self.layer1(x)
        y: Tensor[B, K] = self.layer2(h)
        return y


def test_nested_without_annotations():
    """Test if nested modules work without explicit type annotations"""
    mlp = SimpleMLP(5, 10, 10)

    x: Tensor[16, 5] = torch.randn(16, 5)
    y = mlp(x)
    assert_type(y, Tensor[16, 10])


# ============================================================================
# Test 3: Three-Level Nesting
# ============================================================================


class Block[N, M](nn.Module):
    """Block with two layers"""

    linear: LinearLayer[N, M]

    def __init__(self, in_features: Dim[N], out_features: Dim[M]):
        super().__init__()
        self.linear = LinearLayer(in_features, out_features)

    def forward[B](self, x: Tensor[B, N]) -> Tensor[B, M]:
        out: Tensor[B, M] = self.linear(x)
        out_relu: Tensor[B, M] = torch.relu(out)
        return out_relu


class DeepMLP[N, M, K, L](nn.Module):
    """Deep MLP with three-level nesting"""

    block1: Block[N, M]
    block2: Block[M, K]
    final_layer: LinearLayer[K, L]

    def __init__(
        self,
        in_features: Dim[N],
        hidden1: Dim[M],
        hidden2: Dim[K],
        out_features: Dim[L],
    ):
        super().__init__()
        self.block1 = Block(in_features, hidden1)  # N -> M
        self.block2 = Block(hidden1, hidden2)  # M -> K
        self.final_layer = LinearLayer(hidden2, out_features)  # K -> L

    def forward[B](self, x: Tensor[B, N]) -> Tensor[B, L]:
        h1: Tensor[B, M] = self.block1(x)
        h2: Tensor[B, K] = self.block2(h1)
        y: Tensor[B, L] = self.final_layer(h2)
        return y


def test_three_level_nesting():
    """Test three levels of module nesting"""
    model = DeepMLP(5, 10, 10, 15)

    x: Tensor[8, 5] = torch.randn(8, 5)
    y = model(x)
    assert_type(y, Tensor[8, 15])


# ============================================================================
# Test 4: Mixing Nested Modules with Direct Operations
# ============================================================================


class HybridModel[N, M, K](nn.Module):
    """Model mixing nested modules and direct operations"""

    encoder: LinearLayer[N, M]
    decoder: LinearLayer[M, K]

    def __init__(
        self,
        in_features: Dim[N],
        hidden_features: Dim[M],
        out_features: Dim[K],
    ):
        super().__init__()
        self.encoder = LinearLayer(in_features, hidden_features)  # N -> M
        self.decoder = LinearLayer(hidden_features, out_features)  # M -> K

    def forward[B](self, x: Tensor[B, N]) -> Tensor[B, K]:
        # Nested module
        encoded: Tensor[B, M] = self.encoder(x)

        # Direct operation
        encoded_norm: Tensor[B, M] = encoded / torch.std(encoded, dim=1, keepdim=True)

        # Another nested module
        decoded: Tensor[B, K] = self.decoder(encoded_norm)

        return decoded


def test_hybrid_model():
    """Test mixing nested modules with direct operations"""
    model = HybridModel(5, 10, 7)

    x: Tensor[4, 5] = torch.randn(4, 5)
    y = model(x)
    assert_type(y, Tensor[4, 7])


# ============================================================================
# Test 5: Attention with Nested Projection Layers
# ============================================================================


class Projection[D_in, D_out](nn.Module):
    """Projection layer"""

    weight: Tensor[D_out, D_in]

    def __init__(self, in_dim: Dim[D_in], out_dim: Dim[D_out]):
        super().__init__()
        self.weight = torch.randn(out_dim, in_dim)

    def forward[B, T](self, x: Tensor[B, T, D_in]) -> Tensor[B, T, D_out]:
        # Simple projection using einsum
        return torch.einsum("btd,od->bto", x, self.weight)


class AttentionWithProjections[D](nn.Module):
    """Attention using nested projection modules"""

    q_proj: Projection[D, D]
    k_proj: Projection[D, D]
    v_proj: Projection[D, D]

    def __init__(self, d_model: Dim[D]):
        super().__init__()
        self.q_proj = Projection(d_model, d_model)  # D -> D
        self.k_proj = Projection(d_model, d_model)  # D -> D
        self.v_proj = Projection(d_model, d_model)  # D -> D

    def forward[B, T](self, x: Tensor[B, T, D]) -> Tensor[B, T, D]:
        # Project to Q, K, V
        q: Tensor[B, T, D] = self.q_proj(x)
        k: Tensor[B, T, D] = self.k_proj(x)
        v: Tensor[B, T, D] = self.v_proj(x)

        # Attention
        scores: Tensor[B, T, T] = torch.einsum("btd,bsd->bts", q, k)
        output: Tensor[B, T, D] = torch.einsum("bts,bsd->btd", scores, v)
        return output


def test_attention_with_projections():
    """Test attention with nested projection layers"""
    attn = AttentionWithProjections(512)

    x: Tensor[2, 128, 512] = torch.randn(2, 128, 512)
    y = attn(x)
    assert_type(y, Tensor[2, 128, 512])


# ============================================================================
# Test 6: ResNet-Style Skip Connections with Nested Modules
# ============================================================================


class ConvBlock[C_in, C_out](nn.Module):
    """Convolutional block"""

    weight: Tensor[C_out, C_in, 3, 3]

    def __init__(self, in_channels: Dim[C_in], out_channels: Dim[C_out]):
        super().__init__()
        self.weight = torch.randn(out_channels, in_channels, 3, 3)

    def forward[B, H, W](self, x: Tensor[B, C_in, H, W]) -> Tensor[B, C_out, H, W]:
        import torch.nn.functional as F

        return F.conv2d(x, self.weight, padding=1)


class ResBlock[C](nn.Module):
    """Residual block with nested conv blocks"""

    conv1: ConvBlock[C, C]
    conv2: ConvBlock[C, C]

    def __init__(self, channels: Dim[C]):
        super().__init__()
        self.conv1 = ConvBlock(channels, channels)  # C -> C
        self.conv2 = ConvBlock(channels, channels)  # C -> C

    def forward[B, H, W](self, x: Tensor[B, C, H, W]) -> Tensor[B, C, H, W]:
        identity: Tensor[B, C, H, W] = x

        out: Tensor[B, C, H, W] = self.conv1(x)
        out_relu: Tensor[B, C, H, W] = torch.relu(out)
        out2: Tensor[B, C, H, W] = self.conv2(out_relu)

        # Skip connection
        final: Tensor[B, C, H, W] = out2 + identity
        return torch.relu(final)


def test_resnet_style_block():
    """Test ResNet-style block with nested modules"""
    block = ResBlock(64)

    x: Tensor[4, 64, 28, 28] = torch.randn(4, 64, 28, 28)
    y = block(x)
    assert_type(y, Tensor[4, 64, 28, 28])


# ============================================================================
# Test 7: Multiple Nested Modules at Same Level
# ============================================================================


class ParallelBranches[N, M1, M2, K](nn.Module):
    """Model with parallel branches"""

    branch1_layer1: LinearLayer[N, M1]
    branch1_layer2: LinearLayer[M1, K]
    branch2_layer1: LinearLayer[N, M2]
    branch2_layer2: LinearLayer[M2, K]

    def __init__(
        self,
        in_features: Dim[N],
        hidden1: Dim[M1],
        hidden2: Dim[M2],
        out_features: Dim[K],
    ):
        super().__init__()
        self.branch1_layer1 = LinearLayer(in_features, hidden1)  # N -> M1
        self.branch1_layer2 = LinearLayer(hidden1, out_features)  # M1 -> K
        self.branch2_layer1 = LinearLayer(in_features, hidden2)  # N -> M2
        self.branch2_layer2 = LinearLayer(hidden2, out_features)  # M2 -> K

    def forward[B](self, x: Tensor[B, N]) -> Tensor[B, K]:
        # Branch 1
        h1: Tensor[B, M1] = self.branch1_layer1(x)
        out1: Tensor[B, K] = self.branch1_layer2(h1)

        # Branch 2
        h2: Tensor[B, M2] = self.branch2_layer1(x)
        out2: Tensor[B, K] = self.branch2_layer2(h2)

        # Combine
        final: Tensor[B, K] = out1 + out2
        return final


def test_parallel_branches():
    """Test model with parallel branches"""
    model = ParallelBranches(5, 10, 8, 15)

    x: Tensor[8, 5] = torch.randn(8, 5)
    y = model(x)
    assert_type(y, Tensor[8, 15])
