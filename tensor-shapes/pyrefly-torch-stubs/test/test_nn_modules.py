# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Test class-based patterns with symbolic dimensions
Explores what works for organizing PyTorch code into classes

Key Findings:
- ✅ PEP 695 generic class syntax is recognized
- ✅ Class-level generic params work with nn.Module
- ✅ Generic methods work (declare generics on each method)
- ✅ nn.Module instances are callable
- ✅ Generic classes callable via nn.Module

Recommended Pattern: nn.Module subclasses with generic methods
"""

from typing import assert_type, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from torch import Tensor

# ============================================================================
# Working Pattern 1: Class with Generic Methods
# ============================================================================


class LinearLayer(nn.Module):
    """
    Linear layer using generic methods (not class-level generics)
    Each method declares its own generic parameters
    """

    def __init__(self):
        super().__init__()

    def forward[B, N, M](self, x: Tensor[B, N], weight: Tensor[M, N]) -> Tensor[B, M]:
        """Generic method - all dims visible in method scope"""
        # Use einsum for reliable symbolic dimension handling
        result: Tensor[B, M] = torch.einsum("bn,mn->bm", x, weight)
        return result


def test_linear_generic_method():
    """Test class with generic method"""
    layer = LinearLayer()

    x: Tensor[32, 5] = torch.randn(32, 5)
    weight: Tensor[10, 5] = torch.randn(10, 5)

    # Call module directly (nn.Module instances are callable)
    y = layer(x, weight)
    assert_type(y, Tensor[32, 10])


# ============================================================================
# Working Pattern 2: Multi-Layer Class with Generic Methods
# ============================================================================


class TwoLayerMLP(nn.Module):
    """
    MLP with two transformations
    Each forward declares its own generics
    """

    def __init__(self):
        super().__init__()

    def forward[B, N, M, K](
        self, x: Tensor[B, N], w1: Tensor[M, N], w2: Tensor[K, M]
    ) -> Tensor[B, K]:
        """Two-layer forward pass"""
        # Use einsum for reliable shape inference
        h: Tensor[B, M] = torch.einsum("bn,mn->bm", x, w1)
        h_relu: Tensor[B, M] = torch.relu(h)
        y: Tensor[B, K] = torch.einsum("bm,km->bk", h_relu, w2)
        return y


def test_mlp_generic_method():
    """Test MLP with generic method"""
    mlp = TwoLayerMLP()

    x: Tensor[16, 64] = torch.randn(16, 64)
    w1: Tensor[128, 64] = torch.randn(128, 64)
    w2: Tensor[32, 128] = torch.randn(32, 128)

    y = mlp(x, w1, w2)
    assert_type(y, Tensor[16, 32])


# ============================================================================
# Working Pattern 3: CNN Layer with Generic Method
# ============================================================================


class ConvLayer(nn.Module):
    """Convolutional layer with generic method"""

    def __init__(self):
        super().__init__()

    def forward[B, C_in, C_out, H, W](
        self, x: Tensor[B, C_in, H, W], weight: Tensor[C_out, C_in, 3, 3]
    ) -> Tensor[B, C_out, H, W]:
        """Conv with padding=1 preserves spatial dims"""
        return F.conv2d(x, weight, padding=1)


def test_conv_generic_method():
    """Test CNN layer with generic method"""
    conv = ConvLayer()

    x: Tensor[8, 32, 56, 56] = torch.randn(8, 32, 56, 56)
    weight: Tensor[64, 32, 3, 3] = torch.randn(64, 32, 3, 3)

    y = conv(x, weight)
    assert_type(y, Tensor[8, 64, 56, 56])


# ============================================================================
# Working Pattern 4: Attention with Generic Method
# ============================================================================


class SelfAttention(nn.Module):
    """Self-attention using einsum"""

    def __init__(self):
        super().__init__()

    def forward[B, T, D](self, x: Tensor[B, T, D]) -> Tensor[B, T, D]:
        """Self-attention with Q=K=V=x (simplified)"""
        q: Tensor[B, T, D] = x
        k: Tensor[B, T, D] = x
        v: Tensor[B, T, D] = x

        scores: Tensor[B, T, T] = torch.einsum("btd,bsd->bts", q, k)
        output: Tensor[B, T, D] = torch.einsum("bts,bsd->btd", scores, v)
        return output


def test_self_attention_generic_method():
    """Test attention with generic method"""
    attn = SelfAttention()

    x: Tensor[2, 128, 512] = torch.randn(2, 128, 512)
    y = attn(x)
    assert_type(y, Tensor[2, 128, 512])


# ============================================================================
# Working Pattern 5: Multi-Head Attention
# ============================================================================


class MultiHeadAttention(nn.Module):
    """Multi-head attention"""

    def __init__(self):
        super().__init__()

    def forward[B, H, T, D](self, x: Tensor[B, H, T, D]) -> Tensor[B, H, T, D]:
        """Attention across heads"""
        q: Tensor[B, H, T, D] = x
        k: Tensor[B, H, T, D] = x
        v: Tensor[B, H, T, D] = x

        scores: Tensor[B, H, T, T] = torch.einsum("bhid,bhjd->bhij", q, k)
        output: Tensor[B, H, T, D] = torch.einsum("bhij,bhjd->bhid", scores, v)
        return output


def test_multi_head_attention():
    """Test multi-head attention"""
    mha = MultiHeadAttention()

    x: Tensor[2, 8, 128, 64] = torch.randn(2, 8, 128, 64)
    y = mha(x)
    assert_type(y, Tensor[2, 8, 128, 64])


# ============================================================================
# Working Pattern 6: Cross-Attention
# ============================================================================


class CrossAttention(nn.Module):
    """Cross-attention between different sequences"""

    def __init__(self):
        super().__init__()

    def forward[B, Tq, Tkv, D](
        self,
        queries: Tensor[B, Tq, D],
        keys: Tensor[B, Tkv, D],
        values: Tensor[B, Tkv, D],
    ) -> Tensor[B, Tq, D]:
        """Cross-attention with different sequence lengths"""
        scores: Tensor[B, Tq, Tkv] = torch.einsum("bqd,bkd->bqk", queries, keys)
        output: Tensor[B, Tq, D] = torch.einsum("bqk,bkd->bqd", scores, values)
        return output


def test_cross_attention():
    """Test cross-attention"""
    cross_attn = CrossAttention()

    q: Tensor[2, 50, 512] = torch.randn(2, 50, 512)
    k: Tensor[2, 100, 512] = torch.randn(2, 100, 512)
    v: Tensor[2, 100, 512] = torch.randn(2, 100, 512)

    y = cross_attn(q, k, v)
    assert_type(y, Tensor[2, 50, 512])


# ============================================================================
# Working Pattern 7: Residual Block
# ============================================================================


class ResidualBlock(nn.Module):
    """Residual connection with skip"""

    def __init__(self):
        super().__init__()

    def forward[B, C, H, W](
        self, x: Tensor[B, C, H, W], weight: Tensor[C, C, 3, 3]
    ) -> Tensor[B, C, H, W]:
        """Residual: out + skip"""
        out: Tensor[B, C, H, W] = F.conv2d(x, weight, padding=1)
        out_relu: Tensor[B, C, H, W] = torch.relu(out)
        final: Tensor[B, C, H, W] = out_relu + x
        return final


def test_residual_block():
    """Test residual connection"""
    block = ResidualBlock()

    x: Tensor[4, 64, 28, 28] = torch.randn(4, 64, 28, 28)
    weight: Tensor[64, 64, 3, 3] = torch.randn(64, 64, 3, 3)

    y = block(x, weight)
    assert_type(y, Tensor[4, 64, 28, 28])


# ============================================================================
# Working Pattern 8: Pooling Operation
# ============================================================================


class GlobalAvgPool(nn.Module):
    """Global average pooling"""

    def __init__(self):
        super().__init__()

    def forward[B, C, H, W](self, x: Tensor[B, C, H, W]) -> Tensor[B, C]:
        """Pool over spatial dimensions"""
        # Mean over H dimension, then W dimension
        pooled_h: Tensor[B, C, W] = torch.mean(x, dim=2)
        pooled_hw: Tensor[B, C] = torch.mean(pooled_h, dim=2)
        return pooled_hw


def test_global_avg_pool():
    """Test pooling operation"""
    pool = GlobalAvgPool()

    x: Tensor[16, 512, 7, 7] = torch.randn(16, 512, 7, 7)
    y = pool(x)
    assert_type(y, Tensor[16, 512])


# ============================================================================
# Working Pattern 9: Normalization
# ============================================================================


class LayerNorm(nn.Module):
    """Layer normalization (simplified)"""

    def __init__(self):
        super().__init__()

    def forward[B, T, D](self, x: Tensor[B, T, D]) -> Tensor[B, T, D]:
        """Normalize over last dimension"""
        mean: Tensor[B, T] = torch.mean(x, dim=2)
        std: Tensor[B, T] = torch.std(x, dim=2)

        normalized: Tensor[B, T, D] = (x - mean.unsqueeze(2)) / std.unsqueeze(2)
        return normalized


def test_layer_norm():
    """Test normalization"""
    norm = LayerNorm()

    x: Tensor[4, 128, 512] = torch.randn(4, 128, 512)
    y = norm(x)
    assert_type(y, Tensor[4, 128, 512])


# ============================================================================
# Working Pattern 10: Bilinear Interaction
# ============================================================================


class BilinearPooling(nn.Module):
    """Bilinear pooling for multi-modal fusion"""

    def __init__(self):
        super().__init__()

    def forward[B, C](
        self,
        feat_a: Tensor[B, C, 49],  # Flattened spatial (7*7)
        feat_b: Tensor[B, C, 49],
    ) -> Tensor[B, C, C]:
        """Compute channel-wise outer products"""
        pooled: Tensor[B, C, C] = torch.einsum("bci,bdi->bcd", feat_a, feat_b)
        return pooled


def test_bilinear_pooling():
    """Test bilinear pooling"""
    bilinear = BilinearPooling()

    a: Tensor[2, 512, 49] = torch.randn(2, 512, 49)
    b: Tensor[2, 512, 49] = torch.randn(2, 512, 49)

    y = bilinear(a, b)
    assert_type(y, Tensor[2, 512, 512])


# ============================================================================
# Comparison: Functional Style (Recommended)
# ============================================================================


def batched_linear[B, N, M](x: Tensor[B, N], weight: Tensor[M, N]) -> Tensor[B, M]:
    """
    Functional style - simpler and equally expressive
    This is the recommended pattern for PyRefly
    """
    # Use einsum for reliable shape inference
    return torch.einsum("bn,mn->bm", x, weight)


def test_functional_style():
    """Functional style is simpler and works perfectly"""
    x: Tensor[32, 5] = torch.randn(32, 5)
    weight: Tensor[10, 5] = torch.randn(10, 5)

    y = batched_linear(x, weight)
    assert_type(y, Tensor[32, 10])
