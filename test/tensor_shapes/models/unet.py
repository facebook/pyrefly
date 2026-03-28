# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
UNet from TorchBenchmark with shape annotations.

Original: pytorch/benchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/

Port notes:
- Removes dynamic padding in Up.forward (assumes power-of-2 spatial dims,
    which is the standard UNet usage; the original pads to handle odd sizes)
- Splits Up into Up (non-bilinear) and UpBilinear to give each variant a
    clear type signature; the original uses a runtime bilinear flag
"""

from typing import Any, assert_type, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from torch import Tensor
    from torch_shapes import Dim


# ============================================================================
# Building Blocks
# ============================================================================


class DoubleConv[InC, OutC](nn.Module):
    """(convolution => [BN] => ReLU) * 2

    Shape: (B, InC, H, W) -> (B, OutC, H, W)  [spatial-preserving]

    Conv2d with kernel_size=3 and padding=1 preserves spatial dimensions:
        (H + 2*1 - 1*(3-1) - 1) // 1 + 1 = H
    """

    def __init__(
        self, c_in: Dim[InC], c_out: Dim[OutC], c_mid: int | None = None
    ) -> None:
        super().__init__()
        mid = c_mid if c_mid is not None else c_out
        self.double_conv = nn.Sequential(
            nn.Conv2d(c_in, mid, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, c_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )

    def forward[B, H, W](self, x: Tensor[B, InC, H, W]) -> Tensor[B, OutC, H, W]:
        out = self.double_conv(x)
        assert_type(out, Tensor[B, OutC, H, W])
        return out


class Down[InC, OutC](nn.Module):
    """Downscaling with maxpool then double conv.

    Shape: (B, InC, H, W) -> (B, OutC, H//2, W//2)

    MaxPool2d(kernel_size=2) with stride=2 halves spatial dimensions.
    """

    def __init__(self, c_in: Dim[InC], c_out: Dim[OutC]) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(c_in, c_out)

    def forward[B, H, W](
        self, x: Tensor[B, InC, H, W]
    ) -> Tensor[B, OutC, (H - 2) // 2 + 1, (W - 2) // 2 + 1]:
        x_pooled = self.pool(x)
        assert_type(x_pooled, Tensor[B, InC, (H - 2) // 2 + 1, (W - 2) // 2 + 1])
        out = self.conv(x_pooled)
        assert_type(out, Tensor[B, OutC, (H - 2) // 2 + 1, (W - 2) // 2 + 1])
        return out


class Up[C_in, C_out](nn.Module):
    """Upscaling with transposed convolution, then skip-connection cat, then double conv.

    x1: (B, C_in, H, W)        — deep feature map from previous layer
    x2: (B, C_in // 2, H2, W2) — skip connection from encoder

    ConvTranspose2d(C_in, C_in // 2, kernel_size=2, stride=2) doubles spatial
    dims and halves channels. Then torch.cat along dim=1 concatenates the skip
    connection (also C_in // 2 channels), yielding C_in channels total.
    DoubleConv then reduces to C_out.
    """

    def __init__(self, c_in: Dim[C_in], c_out: Dim[C_out]) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(c_in, c_in // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(c_in, c_out)

    def forward[B, H1, W1, H2, W2](
        self, x1: Tensor[B, C_in, H1, W1], x2: Tensor[B, C_in // 2, H2, W2]
    ) -> Tensor[B, C_out, H2, W2]:
        x1_up = self.up(x1)
        x = torch.cat([x2, x1_up], dim=1)
        return self.conv(x)


class UpBilinear[C_cat, C_out](nn.Module):
    """Upscaling with bilinear interpolation, then skip-connection cat, then double conv.

    x1: (B, C1, H, W)   — deep feature map (channels = C_cat // 2 in standard UNet)
    x2: (B, C2, H2, W2) — skip connection from encoder

    nn.Upsample(scale_factor=2) doubles spatial dims without changing channels.
    Then torch.cat along dim=1 concatenates with the skip connection, and
    DoubleConv (with mid_channels = C_cat // 2) reduces channels to C_out.

    C_cat is the channel count after concatenation (= C1 + C2 = in_channels
    in the original code).
    """

    def __init__(self, c_cat: Dim[C_cat], c_out: Dim[C_out]) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(c_cat, c_out, c_mid=c_cat // 2)

    def forward[B, C1, C2, H1, W1, H2, W2](
        self, x1: Tensor[B, C1, H1, W1], x2: Tensor[B, C2, H2, W2]
    ) -> Tensor[B, C_out, H2, W2]:
        x1_up = self.up(x1)
        assert_type(x1_up, Tensor[B, C1, H1 * 2, W1 * 2])
        x = torch.cat([x2, x1_up], dim=1)
        return self.conv(x)


class OutConv[InC, OutC](nn.Module):
    """1x1 convolution for final output.

    Shape: (B, InC, H, W) -> (B, OutC, H, W)

    Conv2d with kernel_size=1, padding=0 preserves spatial dimensions:
        (H + 0 - 1*(1-1) - 1) // 1 + 1 = H
    """

    def __init__(self, c_in: Dim[InC], c_out: Dim[OutC]) -> None:
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=1)

    def forward[B, H, W](self, x: Tensor[B, InC, H, W]) -> Tensor[B, OutC, H, W]:
        out = self.conv(x)
        assert_type(out, Tensor[B, OutC, H, W])
        return out


# ============================================================================
# Model (non-bilinear variant)
# ============================================================================


class UNet[NChannels, NClasses](nn.Module):
    """U-Net: encoder-decoder with skip connections.

    Non-bilinear variant using ConvTranspose2d for upsampling.
    Channel progression: NChannels -> 64 -> 128 -> 256 -> 512 -> 1024
    then back: 1024 -> 512 -> 256 -> 128 -> 64 -> NClasses

    Each Down block halves spatial dimensions; each Up block doubles them.
    Skip connections concatenate encoder features with decoder features.

    Uses _encode/_decode with list[Stage[Any]] + narrowing annotation for
    ModuleList dispatch, and shape-preserving recursive forward. The recursive
    signature Tensor[B,C,H,W] -> Tensor[B,C,H,W] is verified by the type
    checker: each level encodes (C -> 2C), recurses (preserves shape by
    inductive hypothesis), then decodes (restores shape via skip connection).
    """

    def __init__(self, n_channels: Dim[NChannels], n_classes: Dim[NClasses]) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 64)
        downs: list[Down[Any, Any]] = [
            Down(64, 128),
            Down(128, 256),
            Down(256, 512),
            Down(512, 1024),
        ]
        self.downs = nn.ModuleList(downs)
        ups: list[Up[Any, Any]] = [
            Up(128, 64),
            Up(256, 128),
            Up(512, 256),
            Up(1024, 512),
        ]
        self.ups = nn.ModuleList(ups)
        self.outc = OutConv(64, n_classes)

    def _encode[B, C, H, W](
        self, x: Tensor[B, C, H, W], depth: int
    ) -> Tensor[B, 2 * C, (H - 2) // 2 + 1, (W - 2) // 2 + 1]:
        """Encode one level: doubles channels, halves spatial via Down[C, 2*C]."""
        idx = len(self.downs) - depth
        down: Down[C, 2 * C] = self.downs[idx]
        return down(x)

    def _decode[B, C, H, W](
        self,
        skip: Tensor[B, C, H, W],
        deep: Tensor[B, 2 * C, (H - 2) // 2 + 1, (W - 2) // 2 + 1],
        depth: int,
    ) -> Tensor[B, C, H, W]:
        """Decode one level: restores shape via Up[2*C, C] with skip connection."""
        idx = len(self.ups) - depth
        up: Up[2 * C, C] = self.ups[idx]
        return up(deep, skip)

    def recurse[I, B, C, H, W](
        self, x: Tensor[B, C, H, W], depth: Dim[I]
    ) -> Tensor[B, C, H, W]:
        """Shape-preserving recursive encoder-decoder.

        Base case (depth=0): identity (bottleneck).
        Inductive step: encode (C -> 2C), recurse (preserves 2C), decode (2C -> C).
        """
        if depth == 0:
            return x
        skip = x
        encoded = self._encode(x, depth)
        middle = self.recurse(encoded, depth - 1)
        decoded = self._decode(skip, middle, depth)
        return decoded

    def forward[B](
        self, x: Tensor[B, NChannels, 256, 256]
    ) -> Tensor[B, NClasses, 256, 256]:
        features = self.inc(x)
        assert_type(features, Tensor[B, 64, 256, 256])
        features = self.recurse(features, 4)
        assert_type(features, Tensor[B, 64, 256, 256])
        logits = self.outc(features)
        return logits


# ============================================================================
# Model (bilinear variant)
# ============================================================================


class UNetBilinear[NChannels, NClasses](nn.Module):
    """U-Net with bilinear upsampling.

    Uses nn.Upsample(scale_factor=2, mode='bilinear') instead of
    ConvTranspose2d for upsampling.

    Channel progression differs from non-bilinear: the bottleneck outputs
    512 channels instead of 1024 (factor=2 halves the bottleneck).
    """

    def __init__(self, n_channels: Dim[NChannels], n_classes: Dim[NClasses]) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)  # 1024 // factor, factor=2
        self.up1 = UpBilinear(1024, 256)  # cat(512+512)=1024 -> 256
        self.up2 = UpBilinear(512, 128)  # cat(256+256)=512 -> 128
        self.up3 = UpBilinear(256, 64)  # cat(128+128)=256 -> 64
        self.up4 = UpBilinear(128, 64)  # cat(64+64)=128 -> 64
        self.outc = OutConv(64, n_classes)

    def forward[B](
        self, x: Tensor[B, NChannels, 256, 256]
    ) -> Tensor[B, NClasses, 256, 256]:
        # Encoder
        x1 = self.inc(x)
        assert_type(x1, Tensor[B, 64, 256, 256])
        x2 = self.down1(x1)
        assert_type(x2, Tensor[B, 128, 128, 128])
        x3 = self.down2(x2)
        assert_type(x3, Tensor[B, 256, 64, 64])
        x4 = self.down3(x3)
        assert_type(x4, Tensor[B, 512, 32, 32])
        x5 = self.down4(x4)
        assert_type(x5, Tensor[B, 512, 16, 16])

        # Decoder with skip connections
        d4 = self.up1(x5, x4)
        assert_type(d4, Tensor[B, 256, 32, 32])
        d3 = self.up2(d4, x3)
        assert_type(d3, Tensor[B, 128, 64, 64])
        d2 = self.up3(d3, x2)
        assert_type(d2, Tensor[B, 64, 128, 128])
        d1 = self.up4(d2, x1)
        assert_type(d1, Tensor[B, 64, 256, 256])

        logits = self.outc(d1)
        return logits


# ============================================================================
# Smoke tests
# ============================================================================


def test_double_conv():
    """Test spatial-preserving double convolution."""
    conv = DoubleConv(3, 64)
    x: Tensor[4, 3, 256, 256] = torch.randn(4, 3, 256, 256)
    out = conv(x)
    assert_type(out, Tensor[4, 64, 256, 256])


def test_double_conv_mid_channels():
    """Test double conv with explicit mid_channels (used in bilinear Up)."""
    conv = DoubleConv(1024, 256, c_mid=512)
    x: Tensor[4, 1024, 32, 32] = torch.randn(4, 1024, 32, 32)
    out = conv(x)
    assert_type(out, Tensor[4, 256, 32, 32])


def test_down():
    """Test downsampling block: halves spatial dims, transforms channels."""
    down = Down(64, 128)
    x: Tensor[4, 64, 256, 256] = torch.randn(4, 64, 256, 256)
    out = down(x)
    assert_type(out, Tensor[4, 128, 128, 128])


def test_up():
    """Test upsampling block with transposed convolution and skip connection."""
    up = Up(1024, 512)
    x1: Tensor[4, 1024, 16, 16] = torch.randn(4, 1024, 16, 16)
    x2: Tensor[4, 512, 32, 32] = torch.randn(4, 512, 32, 32)
    out = up(x1, x2)
    assert_type(out, Tensor[4, 512, 32, 32])


def test_up_bilinear():
    """Test upsampling block with bilinear interpolation and skip connection."""
    up = UpBilinear(1024, 256)
    x1: Tensor[4, 512, 16, 16] = torch.randn(4, 512, 16, 16)
    x2: Tensor[4, 512, 32, 32] = torch.randn(4, 512, 32, 32)
    out = up(x1, x2)
    assert_type(out, Tensor[4, 256, 32, 32])


def test_out_conv():
    """Test 1x1 output convolution."""
    outc = OutConv(64, 2)
    x: Tensor[4, 64, 256, 256] = torch.randn(4, 64, 256, 256)
    out = outc(x)
    assert_type(out, Tensor[4, 2, 256, 256])


def test_unet():
    """End-to-end: non-bilinear UNet for 2-class segmentation on 256x256 input."""
    model = UNet(3, 2)
    x: Tensor[1, 3, 256, 256] = torch.randn(1, 3, 256, 256)
    out = model(x)
    assert_type(out, Tensor[1, 2, 256, 256])


def test_unet_bilinear():
    """End-to-end: bilinear UNet for 2-class segmentation on 256x256 input."""
    model = UNetBilinear(3, 2)
    x: Tensor[1, 3, 256, 256] = torch.randn(1, 3, 256, 256)
    out = model(x)
    assert_type(out, Tensor[1, 2, 256, 256])
