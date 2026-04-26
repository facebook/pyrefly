# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Test nn.Module parameters and buffers with typed shapes
Critical for models with learnable weights
"""

from typing import assert_type, TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from torch import Tensor
    from torch_shapes import Dim

# ============================================================================
# Test 1: Basic Parameter with Type Annotation
# ============================================================================


class LinearWithParams[N, M](nn.Module):
    """Linear layer with typed weight parameter"""

    # Can we declare typed parameters as class attributes?
    weight: Tensor[M, N]
    bias: Tensor[M]

    def __init__(self, in_features: Dim[N], out_features: Dim[M]):
        super().__init__()
        # Now N and M are bound to runtime values via Literal types
        self.weight = torch.randn(out_features, in_features)
        self.bias = torch.randn(out_features)

    def forward[B](self, x: Tensor[B, N]) -> Tensor[B, M]:
        # Does using typed parameters preserve shapes?
        weight_t: Tensor[N, M] = self.weight.transpose(0, 1)
        y: Tensor[B, M] = torch.matmul(x, weight_t)
        y_bias: Tensor[B, M] = y + self.bias
        return y_bias


def test_basic_parameters():
    """Test basic parameter usage"""
    layer = LinearWithParams(5, 10)

    x: Tensor[32, 5] = torch.randn(32, 5)
    y = layer(x)
    assert_type(y, Tensor[32, 10])


# ============================================================================
# Test 2: Parameters Without Type Annotations
# ============================================================================


class LinearNoAnnotations[N, M](nn.Module):
    """Linear layer without explicit parameter type annotations"""

    def __init__(self, in_features: Dim[N], out_features: Dim[M]):
        super().__init__()
        # Just assign, no type annotation (but using Literal params to bind N, M)
        self.weight = torch.randn(out_features, in_features)
        self.bias = torch.randn(out_features)

    def forward[B](self, x: Tensor[B, N]) -> Tensor[B, M]:
        # Can PyRefly infer parameter types from initialization?
        weight_t: Tensor[N, M] = self.weight.transpose(0, 1)
        y: Tensor[B, M] = torch.matmul(x, weight_t)
        y_bias: Tensor[B, M] = y + self.bias
        return y_bias


def test_parameters_no_annotations():
    """Test parameters without explicit annotations"""
    layer = LinearNoAnnotations(5, 10)

    x: Tensor[32, 5] = torch.randn(32, 5)
    y = layer(x)
    assert_type(y, Tensor[32, 10])


# ============================================================================
# Test 3: Multiple Parameters with Different Shapes
# ============================================================================


class MultiLayerParams[N, M, K](nn.Module):
    """Module with multiple parameters of different shapes"""

    w1: Tensor[M, N]
    w2: Tensor[K, M]
    b1: Tensor[M]
    b2: Tensor[K]

    def __init__(
        self,
        in_features: Dim[N],
        hidden_features: Dim[M],
        out_features: Dim[K],
    ):
        super().__init__()
        # Now N, M, K are bound to runtime values via Literal types
        self.w1 = torch.randn(hidden_features, in_features)
        self.w2 = torch.randn(out_features, hidden_features)
        self.b1 = torch.randn(hidden_features)
        self.b2 = torch.randn(out_features)

    def forward[B](self, x: Tensor[B, N]) -> Tensor[B, K]:
        w1_t: Tensor[N, M] = self.w1.transpose(0, 1)
        h: Tensor[B, M] = torch.matmul(x, w1_t) + self.b1
        h_relu: Tensor[B, M] = torch.relu(h)

        w2_t: Tensor[M, K] = self.w2.transpose(0, 1)
        y: Tensor[B, K] = torch.matmul(h_relu, w2_t) + self.b2
        return y


def test_multiple_parameters():
    """Test multiple parameters with different shapes"""
    model = MultiLayerParams(64, 128, 32)

    x: Tensor[16, 64] = torch.randn(16, 64)
    y = model(x)
    assert_type(y, Tensor[16, 32])


# ============================================================================
# Test 4: Buffers (Non-Learnable State)
# ============================================================================

# SKIPPED: .view() doesn't preserve symbolic shapes for broadcasting
# This would require meta-shape support for .view() / .reshape()
#
# class BatchNormLike[C](nn.Module):
#     """Module with buffers (running statistics)"""
#     running_mean: Tensor[C]
#     running_var: Tensor[C]
#     gamma: Tensor[C]
#     beta: Tensor[C]
#
#     def forward[B, H, W](
#         self, x: Tensor[B, C, H, W]
#     ) -> Tensor[B, C, H, W]:
#         # .view() returns Unknown, breaking broadcasting
#         mean_expanded = self.running_mean.view(1, -1, 1, 1)
#         # ...


# Simple test for buffers without .view()
class SimpleBufferModule[N](nn.Module):
    """Module with simple buffer (no reshaping)"""

    running_sum: Tensor[N]

    def __init__(self, n: Dim[N]):
        super().__init__()
        # Now N is bound to runtime value via Literal type
        self.running_sum = torch.zeros(n)

    def forward[B](self, x: Tensor[B, N]) -> Tensor[B, N]:
        # Simple buffer usage without reshaping
        return x + self.running_sum


def test_buffers():
    """Test buffers without .view() operations"""
    model = SimpleBufferModule(10)

    x: Tensor[8, 10] = torch.randn(8, 10)
    y = model(x)
    assert_type(y, Tensor[8, 10])


# ============================================================================
# Test 5: Nested Modules with Parameters
# ============================================================================


class LinearLayerWithParams[N, M](nn.Module):
    """Reusable linear layer with parameters"""

    weight: Tensor[M, N]
    bias: Tensor[M]

    def __init__(self, in_features: Dim[N], out_features: Dim[M]):
        super().__init__()
        # Now N and M are bound to runtime values via Literal types
        self.weight = torch.randn(out_features, in_features)
        self.bias = torch.randn(out_features)

    def forward[B](self, x: Tensor[B, N]) -> Tensor[B, M]:
        weight_t: Tensor[N, M] = self.weight.transpose(0, 1)
        y: Tensor[B, M] = torch.matmul(x, weight_t) + self.bias
        return y


class MLPWithNestedParams[N, M, K](nn.Module):
    """MLP with nested modules that have parameters"""

    layer1: LinearLayerWithParams[N, M]
    layer2: LinearLayerWithParams[M, K]

    def __init__(
        self,
        in_features: Dim[N],
        hidden_features: Dim[M],
        out_features: Dim[K],
    ):
        super().__init__()
        # Now N, M, K are bound, so we can construct the nested modules
        self.layer1 = LinearLayerWithParams(in_features, hidden_features)
        self.layer2 = LinearLayerWithParams(hidden_features, out_features)

    def forward[B](self, x: Tensor[B, N]) -> Tensor[B, K]:
        # Do nested parameters work correctly?
        h: Tensor[B, M] = self.layer1(x)
        h_relu: Tensor[B, M] = torch.relu(h)
        y: Tensor[B, K] = self.layer2(h_relu)
        return y


def test_nested_modules_with_parameters():
    """Test nested modules with their own parameters"""
    mlp = MLPWithNestedParams(5, 10, 10)

    x: Tensor[16, 5] = torch.randn(16, 5)
    y = mlp(x)
    assert_type(y, Tensor[16, 10])


# ============================================================================
# Test 6: Shared Parameters
# ============================================================================


class WeightTying[N, M](nn.Module):
    """Module with shared parameters (weight tying pattern)"""

    shared_weight: Tensor[M, N]

    def __init__(self, in_features: Dim[N], out_features: Dim[M]):
        super().__init__()
        # Now N and M are bound to runtime values via Literal types
        self.shared_weight = torch.randn(out_features, in_features)

    def encode[B](self, x: Tensor[B, N]) -> Tensor[B, M]:
        """Encoder using shared weight"""
        weight_t: Tensor[N, M] = self.shared_weight.transpose(0, 1)
        return torch.matmul(x, weight_t)

    def decode[B](self, h: Tensor[B, M]) -> Tensor[B, N]:
        """Decoder using transposed shared weight"""
        weight_tt: Tensor[M, N] = self.shared_weight
        return torch.matmul(h, weight_tt)


def test_shared_parameters():
    """Test weight tying (shared parameters)"""
    model = WeightTying(256, 512)

    x: Tensor[8, 256] = torch.randn(8, 256)
    h = model.encode(x)
    assert_type(h, Tensor[8, 512])

    x_recon = model.decode(h)
    assert_type(x_recon, Tensor[8, 256])


# ============================================================================
# Test 7: Complex Parameter Shapes (Conv Weights)
# ============================================================================


class ConvWithParams[C_in, C_out](nn.Module):
    """Convolutional layer with 4D weight tensor"""

    weight: Tensor[C_out, C_in, 3, 3]
    bias: Tensor[C_out]

    def __init__(self, in_channels: Dim[C_in], out_channels: Dim[C_out]):
        super().__init__()
        # Now C_in and C_out are bound to runtime values via Literal types
        self.weight = torch.randn(out_channels, in_channels, 3, 3)
        self.bias = torch.randn(out_channels)

    def forward[B, H, W](self, x: Tensor[B, C_in, H, W]) -> Tensor[B, C_out, H, W]:
        import torch.nn.functional as F

        # Use typed conv weight
        out: Tensor[B, C_out, H, W] = F.conv2d(x, self.weight, self.bias, padding=1)
        return out


def test_conv_parameters():
    """Test 4D convolutional weight tensors"""
    conv = ConvWithParams(32, 64)

    x: Tensor[4, 32, 28, 28] = torch.randn(4, 32, 28, 28)
    y = conv(x)
    assert_type(y, Tensor[4, 64, 28, 28])


# ============================================================================
# Test 8: Attention with Projection Parameters
# ============================================================================


class AttentionParams[D](nn.Module):
    """Attention with explicit Q/K/V projection weights"""

    w_q: Tensor[D, D]
    w_k: Tensor[D, D]
    w_v: Tensor[D, D]
    w_o: Tensor[D, D]

    def __init__(self, d_model: Dim[D]):
        super().__init__()
        # Now D is bound to runtime value via Literal type
        self.w_q = torch.randn(d_model, d_model)
        self.w_k = torch.randn(d_model, d_model)
        self.w_v = torch.randn(d_model, d_model)
        self.w_o = torch.randn(d_model, d_model)

    def forward[B, T](self, x: Tensor[B, T, D]) -> Tensor[B, T, D]:
        # Project to Q, K, V
        q: Tensor[B, T, D] = torch.einsum("btd,de->bte", x, self.w_q)
        k: Tensor[B, T, D] = torch.einsum("btd,de->bte", x, self.w_k)
        v: Tensor[B, T, D] = torch.einsum("btd,de->bte", x, self.w_v)

        # Attention
        scores: Tensor[B, T, T] = torch.einsum("btd,bsd->bts", q, k)
        attn_output: Tensor[B, T, D] = torch.einsum("bts,bsd->btd", scores, v)

        # Output projection
        output: Tensor[B, T, D] = torch.einsum("btd,de->bte", attn_output, self.w_o)
        return output


def test_attention_parameters():
    """Test attention with projection parameters"""
    attn = AttentionParams(512)

    x: Tensor[2, 128, 512] = torch.randn(2, 128, 512)
    y = attn(x)
    assert_type(y, Tensor[2, 128, 512])
