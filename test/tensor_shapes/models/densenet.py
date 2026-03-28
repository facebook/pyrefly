# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
DenseNet from TorchBenchmark with shape annotations.

Original: pytorch/benchmark/torchbenchmark/models/phlippe_densenet/__init__.py

Port notes:
- DenseLayer breaks the original nn.Sequential into explicit calls for shape
    tracking through BatchNorm2d → ReLU → Conv2d(1x1) → BatchNorm2d → ReLU →
    Conv2d(3x3), then concatenates input with output (dense connection)
- WORKAROUND: Uses F.relu instead of configurable act_fn class parameter
    (the original passes nn.ReLU/nn.Tanh/etc. as a class constructor)
- TransitionLayer uses nn.AvgPool2d for spatial downsampling
    (nn.AvgPool2d's forward returns unrefined Tensor, no DSL redirect)
- DenseBlock chains DenseLayers explicitly in forward (not via nn.Sequential)
    because each layer has different input channel counts
- DenseNet uses concrete default config (growth_rate=16, bn_size=2,
    num_layers=[6,6,6,6], num_classes=10) since channel arithmetic is dynamic
"""

from typing import Any, assert_type, overload, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from torch import Tensor
    from torch_shapes import Dim


# ============================================================================
# Building Blocks
# ============================================================================


class DenseLayer[InC, BnC, GR](nn.Module):
    """Single dense layer with bottleneck.

    Architecture: BN → ReLU → 1x1 Conv(InC → BnC) → BN → ReLU → 3x3 Conv(BnC → GR)
    Then concatenates output (GR channels) with input (InC channels).

    Input:  Tensor[B, InC, H, W]
    Output: Tensor[B, InC + GR, H, W]

    BnC = bn_size * growth_rate (bottleneck channels for the 1x1 conv).
    """

    def __init__(
        self, c_in: Dim[InC], bn_channels: Dim[BnC], growth_rate: Dim[GR]
    ) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(c_in)
        self.conv1 = nn.Conv2d(c_in, bn_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(bn_channels)
        self.conv2 = nn.Conv2d(
            bn_channels, growth_rate, kernel_size=3, padding=1, bias=False
        )

    def forward[B, H, W](self, x: Tensor[B, InC, H, W]) -> Tensor[B, InC + GR, H, W]:
        # WORKAROUND: F.relu instead of configurable act_fn()
        out0 = F.relu(self.bn1(x))
        assert_type(out0, Tensor[B, InC, H, W])
        out1 = self.conv1(out0)
        assert_type(out1, Tensor[B, BnC, H, W])
        out2 = F.relu(self.bn2(out1))
        assert_type(out2, Tensor[B, BnC, H, W])
        out3 = self.conv2(out2)
        assert_type(out3, Tensor[B, GR, H, W])
        # Note: torch.cat returns unrefined Tensor in generic body;
        # shape is verified at concrete call sites via test functions
        result = torch.cat([out3, x], dim=1)
        return result


class TransitionLayer[InC, OutC](nn.Module):
    """Transition between dense blocks: BN → ReLU → 1x1 Conv → AvgPool(2).

    Reduces channels from InC to OutC and halves spatial dimensions.

    Input:  Tensor[B, InC, H, W]
    Output: Tensor[B, OutC, (H-2)//2+1, (W-2)//2+1]
    """

    def __init__(self, c_in: Dim[InC], c_out: Dim[OutC]) -> None:
        super().__init__()
        self.bn = nn.BatchNorm2d(c_in)
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(2)

    def forward[B, H, W](
        self, x: Tensor[B, InC, H, W]
    ) -> Tensor[B, OutC, (H - 2) // 2 + 1, (W - 2) // 2 + 1]:
        # WORKAROUND: F.relu instead of configurable act_fn()
        out0 = F.relu(self.bn(x))
        assert_type(out0, Tensor[B, InC, H, W])
        out1 = self.conv(out0)
        assert_type(out1, Tensor[B, OutC, H, W])
        out2 = self.pool(out1)
        assert_type(out2, Tensor[B, OutC, (H - 2) // 2 + 1, (W - 2) // 2 + 1])
        return out2


# ============================================================================
# DenseBlock: 6 chained DenseLayers (default config)
# ============================================================================


class DenseBlock[C, GR, BnC](nn.Module):
    """Dense block with 6 layers, using recursive forward.

    Each DenseLayer adds GR channels via concatenation.
    Input channels grow: C → C+GR → C+2*GR → ... → C+6*GR

    Uses _apply_layer + _chain for recursive shape verification instead
    of manually unrolled forward. The inductive step relies on symbolic
    product distribution: (Ch + GR) + GR*(I-1) = Ch + GR*I.
    """

    def __init__(
        self, c_in: Dim[C], growth_rate: Dim[GR], bn_channels: Dim[BnC]
    ) -> None:
        super().__init__()
        layers: list[DenseLayer[Any, Any, Any]] = [
            DenseLayer(c_in, bn_channels, growth_rate),
            DenseLayer(c_in + growth_rate, bn_channels, growth_rate),
            DenseLayer(c_in + 2 * growth_rate, bn_channels, growth_rate),
            DenseLayer(c_in + 3 * growth_rate, bn_channels, growth_rate),
            DenseLayer(c_in + 4 * growth_rate, bn_channels, growth_rate),
            DenseLayer(c_in + 5 * growth_rate, bn_channels, growth_rate),
        ]
        self.layers = nn.ModuleList(layers)

    def _apply_layer[B, Ch, H, W](
        self, x: Tensor[B, Ch, H, W], depth: int
    ) -> Tensor[B, Ch + GR, H, W]:
        idx = len(self.layers) - depth
        layer: DenseLayer[Ch, BnC, GR] = self.layers[idx]
        return layer(x)

    def forward[B, H, W](self, x: Tensor[B, C, H, W]) -> Tensor[B, C + 6 * GR, H, W]:
        return _dense_chain(self, x, 6)


@overload
def _dense_chain[GR, B, Ch, H, W](
    block: DenseBlock[Any, GR, Any], x: Tensor[B, Ch, H, W], depth: Dim[1]
) -> Tensor[B, Ch + GR, H, W]: ...


@overload
def _dense_chain[I, GR, B, Ch, H, W](
    block: DenseBlock[Any, GR, Any], x: Tensor[B, Ch, H, W], depth: Dim[I]
) -> Tensor[B, Ch + I * GR, H, W]: ...


def _dense_chain[I, GR, B, Ch, H, W](
    block: DenseBlock[Any, GR, Any], x: Tensor[B, Ch, H, W], depth: Dim[I]
) -> Tensor[B, Ch + GR, H, W] | Tensor[B, Ch + I * GR, H, W]:
    y = block._apply_layer(x, depth)
    if depth == 1:
        return y
    return _dense_chain(block, y, depth - 1)


# ============================================================================
# DenseNet (default config: growth_rate=16, bn_size=2, num_layers=[6,6,6,6])
# ============================================================================


class DenseNet(nn.Module):
    """DenseNet with default configuration for CIFAR-10.

    Config: growth_rate=16, bn_size=2, num_layers=[6,6,6,6], num_classes=10
    Input: 3×32×32 (CIFAR-10)

    Channel progression:
    - Input conv: 3 → 32 (= growth_rate * bn_size)
    - Block 1: 32 → 128 (= 32 + 6*16), spatial 32×32
    - Transition 1: 128 → 64, spatial 32→16
    - Block 2: 64 → 160 (= 64 + 6*16), spatial 16×16
    - Transition 2: 160 → 80, spatial 16→8
    - Block 3: 80 → 176 (= 80 + 6*16), spatial 8×8
    - Transition 3: 176 → 88, spatial 8→4
    - Block 4: 88 → 184 (= 88 + 6*16), spatial 4×4
    - Output: BN → ReLU → AdaptiveAvgPool(1,1) → Flatten → Linear(184, 10)
    """

    def __init__(self) -> None:
        super().__init__()
        # bn_channels = bn_size * growth_rate = 2 * 16 = 32
        # Input convolution
        self.input_conv = nn.Conv2d(3, 32, kernel_size=3, padding=1)

        # Dense blocks and transitions
        self.block1 = DenseBlock(32, 16, 32)
        self.trans1 = TransitionLayer(128, 64)
        self.block2 = DenseBlock(64, 16, 32)
        self.trans2 = TransitionLayer(160, 80)
        self.block3 = DenseBlock(80, 16, 32)
        self.trans3 = TransitionLayer(176, 88)
        self.block4 = DenseBlock(88, 16, 32)

        # Output layers
        self.out_bn = nn.BatchNorm2d(184)
        self.out_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_flatten = nn.Flatten()
        self.out_linear = nn.Linear(184, 10)

    def forward[B](self, x: Tensor[B, 3, 32, 32]) -> Tensor[B, 10]:
        # Input convolution
        h0 = self.input_conv(x)
        assert_type(h0, Tensor[B, 32, 32, 32])

        # Block 1 + Transition 1
        h1 = self.block1(h0)
        assert_type(h1, Tensor[B, 128, 32, 32])
        h1t = self.trans1(h1)
        assert_type(h1t, Tensor[B, 64, 16, 16])

        # Block 2 + Transition 2
        h2 = self.block2(h1t)
        assert_type(h2, Tensor[B, 160, 16, 16])
        h2t = self.trans2(h2)
        assert_type(h2t, Tensor[B, 80, 8, 8])

        # Block 3 + Transition 3
        h3 = self.block3(h2t)
        assert_type(h3, Tensor[B, 176, 8, 8])
        h3t = self.trans3(h3)
        assert_type(h3t, Tensor[B, 88, 4, 4])

        # Block 4 (no transition after last block)
        h4 = self.block4(h3t)
        assert_type(h4, Tensor[B, 184, 4, 4])

        # Output: BN → ReLU → AdaptiveAvgPool → Flatten → Linear
        # WORKAROUND: F.relu instead of configurable act_fn()
        out_bn = F.relu(self.out_bn(h4))
        assert_type(out_bn, Tensor[B, 184, 4, 4])
        out_pool = self.out_pool(out_bn)
        assert_type(out_pool, Tensor[B, 184, 1, 1])
        out_flat = self.out_flatten(out_pool)
        assert_type(out_flat, Tensor[B, 184])
        logits = self.out_linear(out_flat)
        assert_type(logits, Tensor[B, 10])
        return logits


# ============================================================================
# Smoke tests
# ============================================================================


def test_dense_layer():
    """Test single dense layer: cat adds growth_rate channels."""
    layer = DenseLayer(32, 32, 16)
    x: Tensor[4, 32, 8, 8] = torch.randn(4, 32, 8, 8)
    out = layer(x)
    assert_type(out, Tensor[4, 48, 8, 8])


def test_dense_layer_accumulated():
    """Test dense layer with accumulated channels (3rd layer in a block)."""
    layer = DenseLayer(64, 32, 16)
    x: Tensor[4, 64, 8, 8] = torch.randn(4, 64, 8, 8)
    out = layer(x)
    assert_type(out, Tensor[4, 80, 8, 8])


def test_transition_layer():
    """Test transition: halves channels and spatial dims."""
    trans = TransitionLayer(128, 64)
    x: Tensor[4, 128, 32, 32] = torch.randn(4, 128, 32, 32)
    out = trans(x)
    assert_type(out, Tensor[4, 64, 16, 16])


def test_dense_block():
    """Test dense block with 6 layers: adds 6*growth_rate channels."""
    block = DenseBlock(32, 16, 32)
    x: Tensor[4, 32, 32, 32] = torch.randn(4, 32, 32, 32)
    out = block(x)
    assert_type(out, Tensor[4, 128, 32, 32])


def test_dense_block_2():
    """Test second dense block with different input channels."""
    block = DenseBlock(64, 16, 32)
    x: Tensor[4, 64, 16, 16] = torch.randn(4, 64, 16, 16)
    out = block(x)
    assert_type(out, Tensor[4, 160, 16, 16])


def test_densenet():
    """End-to-end: DenseNet for CIFAR-10 classification."""
    model = DenseNet()
    x: Tensor[2, 3, 32, 32] = torch.randn(2, 3, 32, 32)
    out = model(x)
    assert_type(out, Tensor[2, 10])
