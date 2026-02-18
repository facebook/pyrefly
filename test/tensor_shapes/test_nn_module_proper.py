# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Test nn.Module with proper class-level generics
This tests the USER'S correct pattern, not my buggy one!
"""

from typing import assert_type, TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from torch import Tensor
    from torch_shapes import Dim

# ============================================================================
# Pattern 1: nn.Module with Class-Level Generics
# ============================================================================


class LinearLayer[N, M](nn.Module):
    """
    Linear layer with class-level dimension parameters
    N and M should be visible in all methods
    """

    # Declare weights as class attributes with generic types
    weight: Tensor[M, N]
    bias: Tensor[M]

    def __init__(self, in_features: Dim[N], out_features: Dim[M]):
        super().__init__()
        # Now N and M are bound via Literal params, so we can create tensors
        self.weight = torch.randn(out_features, in_features)
        self.bias = torch.randn(out_features)

    def forward[B](self, x: Tensor[B, N]) -> Tensor[B, M]:
        """
        B is method-level generic (batch dimension)
        N, M come from class (input/output dimensions)
        """
        # Use the typed weights
        weight_t: Tensor[N, M] = self.weight.transpose(0, 1)
        y: Tensor[B, M] = torch.matmul(x, weight_t)
        y_bias: Tensor[B, M] = y + self.bias
        return y_bias


def test_linear_with_matmul():
    """Test @ operator with symbolic dimensions"""
    layer = LinearLayer(6, 9)
    assert_type(layer, LinearLayer[6, 9])

    x: Tensor[16, 6] = torch.randn(16, 6)
    y = layer(x)
    assert_type(y, Tensor[16, 9])


# ============================================================================
# Pattern 2: Multi-Layer MLP with Class Generics
# ============================================================================


class TwoLayerMLP[N, M, K](nn.Module):
    """
    Two-layer MLP with class-level dimension parameters
    """

    # Declare weights as class attributes with generic types
    w1: Tensor[M, N]
    w2: Tensor[K, M]

    def __init__(
        self,
        in_features: Dim[N],
        hidden_features: Dim[M],
        out_features: Dim[K],
    ):
        super().__init__()
        # Now N, M, K are bound via Literal params, so we can create tensors
        self.w1 = torch.randn(hidden_features, in_features)
        self.w2 = torch.randn(out_features, hidden_features)

    def forward[B](self, x: Tensor[B, N]) -> Tensor[B, K]:
        """
        B is batch (method-level)
        N, M, K are dimensions (class-level)
        """
        # Use torch.matmul (@ operator doesn't have meta-shape support yet)
        w1_t: Tensor[N, M] = self.w1.transpose(0, 1)
        w2_t: Tensor[M, K] = self.w2.transpose(0, 1)

        h: Tensor[B, M] = torch.matmul(x, w1_t)
        h_relu: Tensor[B, M] = torch.relu(h)
        y: Tensor[B, K] = torch.matmul(h_relu, w2_t)
        return y


def test_mlp_with_matmul():
    """Test MLP using @ operator"""
    mlp = TwoLayerMLP(64, 128, 32)

    x: Tensor[16, 64] = torch.randn(16, 64)
    y = mlp(x)
    assert_type(y, Tensor[16, 32])


# ============================================================================
# Pattern 3: Attention Module
# ============================================================================


class SelfAttention[D](nn.Module):
    """Self-attention with class-level dimension"""

    def forward[B, T](self, x: Tensor[B, T, D]) -> Tensor[B, T, D]:
        """
        B, T are method-level (batch, sequence length)
        D is class-level (model dimension)
        """
        q: Tensor[B, T, D] = x
        k: Tensor[B, T, D] = x
        v: Tensor[B, T, D] = x

        # Einsum for attention
        scores: Tensor[B, T, T] = torch.einsum("btd,bsd->bts", q, k)
        output: Tensor[B, T, D] = torch.einsum("bts,bsd->btd", scores, v)
        return output


def test_self_attention():
    """Test attention module"""
    attn = SelfAttention()

    x: Tensor[2, 128, 512] = torch.randn(2, 128, 512)
    y = attn(x)
    assert_type(y, Tensor[2, 128, 512])


# ============================================================================
# Pattern 4: CNN Module
# ============================================================================


class ConvBlock[C_in, C_out](nn.Module):
    """Convolutional block with class-level channel dims"""

    # Declare weight as class attribute with generic type
    weight: Tensor[C_out, C_in, 3, 3]

    def __init__(self, in_channels: Dim[C_in], out_channels: Dim[C_out]):
        super().__init__()
        # Now C_in and C_out are bound via Literal params
        self.weight = torch.randn(out_channels, in_channels, 3, 3)

    def forward[B, H, W](self, x: Tensor[B, C_in, H, W]) -> Tensor[B, C_out, H, W]:
        """
        B, H, W are method-level (batch, spatial)
        C_in, C_out are class-level (channels)
        """
        import torch.nn.functional as F

        out: Tensor[B, C_out, H, W] = F.conv2d(x, self.weight, padding=1)
        return out


def test_conv_block():
    """Test CNN module"""
    conv = ConvBlock(32, 64)

    x: Tensor[8, 32, 56, 56] = torch.randn(8, 32, 56, 56)
    y = conv(x)
    assert_type(y, Tensor[8, 64, 56, 56])


# ============================================================================
# Pattern 5: Residual Block
# ============================================================================


class ResidualBlock[C](nn.Module):
    """Residual block with class-level channel dimension"""

    # Declare weight as class attribute with generic type
    weight: Tensor[C, C, 3, 3]

    def __init__(self, channels: Dim[C]):
        super().__init__()
        # Now C is bound via Literal param
        self.weight = torch.randn(channels, channels, 3, 3)

    def forward[B, H, W](self, x: Tensor[B, C, H, W]) -> Tensor[B, C, H, W]:
        """
        B, H, W are method-level
        C is class-level (channels preserved in residual)
        """
        import torch.nn.functional as F

        out: Tensor[B, C, H, W] = F.conv2d(x, self.weight, padding=1)
        out_relu: Tensor[B, C, H, W] = torch.relu(out)

        # Skip connection
        final: Tensor[B, C, H, W] = out_relu + x
        return final


def test_residual_block():
    """Test residual connection"""
    block = ResidualBlock(64)

    x: Tensor[4, 64, 28, 28] = torch.randn(4, 64, 28, 28)
    y = block(x)
    assert_type(y, Tensor[4, 64, 28, 28])
