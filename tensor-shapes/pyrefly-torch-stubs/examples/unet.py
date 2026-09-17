# Portions (c) Meta Platforms, Inc. and affiliates.
#
# This source code is adapted from pytorch/benchmark (TorchBenchmark),
# which is licensed under the BSD 3-Clause License:
# https://github.com/pytorch/benchmark/blob/main/LICENSE
#
# This adaptation adds tensor shape type annotations for pyrefly.

"""
UNet from TorchBenchmark with shape annotations.

Original: pytorch/benchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/

Port notes:
- Removes dynamic padding in Up.forward (assumes power-of-2 spatial dims,
    which is the standard UNet usage; the original pads to handle odd sizes)
- Splits Up into Up (non-bilinear) and UpBilinear to give each variant a
    clear type signature; the original uses a runtime bilinear flag
"""

from typing import assert_type, TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from shape_extensions import Int, IntVar
    from torch import Tensor


# ============================================================================
# Building Blocks
# ============================================================================


class DoubleConv[InC: IntVar, OutC: IntVar](nn.Module):
    """(convolution => [BN] => ReLU) * 2

    Shape: (B, InC, H, W) -> (B, OutC, H, W)  [spatial-preserving]

    Conv2d with kernel_size=3 and padding=1 preserves spatial dimensions:
        (H + 2*1 - 1*(3-1) - 1) // 1 + 1 = H
    """

    def __init__(
        self, c_in: Int[InC], c_out: Int[OutC], c_mid: int | None = None
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

    def forward[B: IntVar, H: IntVar, W: IntVar](
        self, x: Tensor[[B, InC, H, W]]
    ) -> Tensor[[B, OutC, H, W]]:
        out = self.double_conv(x)
        assert_type(out, Tensor[[B, OutC, H, W]])
        return out


class Down[InC: IntVar, OutC: IntVar](nn.Module):
    """Downscaling with maxpool then double conv.

    Shape: (B, InC, H, W) -> (B, OutC, H//2, W//2)

    MaxPool2d(kernel_size=2) with stride=2 halves spatial dimensions.
    """

    def __init__(self, c_in: Int[InC], c_out: Int[OutC]) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(c_in, c_out)

    def forward[B: IntVar, H: IntVar, W: IntVar](
        self, x: Tensor[[B, InC, H, W]]
    ) -> Tensor[[B, OutC, (H - 2) // 2 + 1, (W - 2) // 2 + 1]]:
        x_pooled = self.pool(x)
        assert_type(x_pooled, Tensor[[B, InC, (H - 2) // 2 + 1, (W - 2) // 2 + 1]])
        out = self.conv(x_pooled)
        assert_type(out, Tensor[[B, OutC, (H - 2) // 2 + 1, (W - 2) // 2 + 1]])
        return out


class Up[C_half: IntVar, C_out: IntVar](nn.Module):
    """Upscaling with transposed convolution, then skip-connection cat, then double conv.

    x1: (B, 2 * C_half, H, W)     — deep feature map from previous layer
    x2: (B, C_half, 2 * H, 2 * W) — skip connection from encoder

    ConvTranspose2d(2 * C_half, C_half, kernel_size=2, stride=2) doubles spatial
    dims and halves channels, landing exactly on the skip connection's shape. The
    cat along dim=1 therefore joins two (B, C_half, 2 * H, 2 * W) maps and
    provably has 2 * C_half channels, which DoubleConv reduces to C_out.

    Both relations in the signature exist to make that cat decidable: naming the
    half rather than the concatenated total (`C // 2 + C // 2` is not `C` for odd
    `C`), and naming the skip extent as twice the deep extent rather than as a
    free variable the checker could not relate to the upsampled one.
    """

    def __init__(self, c_half: Int[C_half], c_out: Int[C_out]) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(c_half * 2, c_half, kernel_size=2, stride=2)
        self.conv = DoubleConv(c_half * 2, c_out)

    def forward[B: IntVar, H: IntVar, W: IntVar](
        self, x1: Tensor[[B, C_half * 2, H, W]], x2: Tensor[[B, C_half, H * 2, W * 2]]
    ) -> Tensor[[B, C_out, H * 2, W * 2]]:
        x1_up = self.up(x1)
        assert_type(x1_up, Tensor[[B, C_half, H * 2, W * 2]])
        x = torch.cat([x2, x1_up], dim=1)
        assert_type(x, Tensor[[B, C_half * 2, H * 2, W * 2]])
        return self.conv(x)


class UpBilinear[C_half: IntVar, C_out: IntVar](nn.Module):
    """Upscaling with bilinear interpolation, then skip-connection cat, then double conv.

    x1: (B, C_half, H, W)         — deep feature map from previous layer
    x2: (B, C_half, 2 * H, 2 * W) — skip connection from encoder

    nn.Upsample(scale_factor=2) doubles spatial dims without changing channels, so
    the upsampled map lands on the skip connection's shape and the cat along dim=1
    provably has 2 * C_half channels. DoubleConv (with mid_channels = C_half)
    reduces that to C_out.

    As in Up, the signature names the per-branch channel count and ties the skip
    extent to the deep one, which is what makes the concatenation decidable.
    """

    def __init__(self, c_half: Int[C_half], c_out: Int[C_out]) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(c_half * 2, c_out, c_mid=c_half)

    def forward[B: IntVar, H: IntVar, W: IntVar](
        self, x1: Tensor[[B, C_half, H, W]], x2: Tensor[[B, C_half, H * 2, W * 2]]
    ) -> Tensor[[B, C_out, H * 2, W * 2]]:
        x1_up = self.up(x1)
        assert_type(x1_up, Tensor[[B, C_half, H * 2, W * 2]])
        x = torch.cat([x2, x1_up], dim=1)
        assert_type(x, Tensor[[B, C_half * 2, H * 2, W * 2]])
        return self.conv(x)


class OutConv[InC: IntVar, OutC: IntVar](nn.Module):
    """1x1 convolution for final output.

    Shape: (B, InC, H, W) -> (B, OutC, H, W)

    Conv2d with kernel_size=1, padding=0 preserves spatial dimensions:
        (H + 0 - 1*(1-1) - 1) // 1 + 1 = H
    """

    def __init__(self, c_in: Int[InC], c_out: Int[OutC]) -> None:
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=1)

    def forward[B: IntVar, H: IntVar, W: IntVar](
        self, x: Tensor[[B, InC, H, W]]
    ) -> Tensor[[B, OutC, H, W]]:
        out = self.conv(x)
        assert_type(out, Tensor[[B, OutC, H, W]])
        return out


# ============================================================================
# Model (non-bilinear variant)
# ============================================================================


class UNet[NChannels: IntVar, NClasses: IntVar](nn.Module):
    """U-Net: encoder-decoder with skip connections.

    Non-bilinear variant using ConvTranspose2d for upsampling.
    Channel progression: NChannels -> 64 -> 128 -> 256 -> 512 -> 1024
    then back: 1024 -> 512 -> 256 -> 128 -> 64 -> NClasses

    Each Down block halves spatial dimensions; each Up block doubles them.
    Skip connections concatenate encoder features with decoder features.

    Every level is its own attribute with concrete channel counts. The levels
    are written out rather than recursed over: Up proves its own cat only when
    the skip extent is exactly twice the deep extent, and a depth-generic
    recursion cannot state that its input is divisible by two once per
    remaining level.
    """

    def __init__(self, n_channels: Int[NChannels], n_classes: Int[NClasses]) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(512, 512)  # cat(512+512)=1024 -> 512
        self.up2 = Up(256, 256)  # cat(256+256)=512 -> 256
        self.up3 = Up(128, 128)  # cat(128+128)=256 -> 128
        self.up4 = Up(64, 64)  # cat(64+64)=128 -> 64
        self.outc = OutConv(64, n_classes)

    def forward[B: IntVar](
        self, x: Tensor[[B, NChannels, 256, 256]]
    ) -> Tensor[[B, NClasses, 256, 256]]:
        # Encoder
        x1 = self.inc(x)
        assert_type(x1, Tensor[[B, 64, 256, 256]])
        x2 = self.down1(x1)
        assert_type(x2, Tensor[[B, 128, 128, 128]])
        x3 = self.down2(x2)
        assert_type(x3, Tensor[[B, 256, 64, 64]])
        x4 = self.down3(x3)
        assert_type(x4, Tensor[[B, 512, 32, 32]])
        x5 = self.down4(x4)
        assert_type(x5, Tensor[[B, 1024, 16, 16]])

        # Decoder with skip connections
        d4 = self.up1(x5, x4)
        assert_type(d4, Tensor[[B, 512, 32, 32]])
        d3 = self.up2(d4, x3)
        assert_type(d3, Tensor[[B, 256, 64, 64]])
        d2 = self.up3(d3, x2)
        assert_type(d2, Tensor[[B, 128, 128, 128]])
        d1 = self.up4(d2, x1)
        assert_type(d1, Tensor[[B, 64, 256, 256]])

        logits = self.outc(d1)
        return logits


# ============================================================================
# Model (bilinear variant)
# ============================================================================


class UNetBilinear[NChannels: IntVar, NClasses: IntVar](nn.Module):
    """U-Net with bilinear upsampling.

    Uses nn.Upsample(scale_factor=2, mode='bilinear') instead of
    ConvTranspose2d for upsampling.

    Channel progression differs from non-bilinear: the bottleneck outputs
    512 channels instead of 1024 (factor=2 halves the bottleneck).
    """

    def __init__(self, n_channels: Int[NChannels], n_classes: Int[NClasses]) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)  # 1024 // factor, factor=2
        self.up1 = UpBilinear(512, 256)  # cat(512+512)=1024 -> 256
        self.up2 = UpBilinear(256, 128)  # cat(256+256)=512 -> 128
        self.up3 = UpBilinear(128, 64)  # cat(128+128)=256 -> 64
        self.up4 = UpBilinear(64, 64)  # cat(64+64)=128 -> 64
        self.outc = OutConv(64, n_classes)

    def forward[B: IntVar](
        self, x: Tensor[[B, NChannels, 256, 256]]
    ) -> Tensor[[B, NClasses, 256, 256]]:
        # Encoder
        x1 = self.inc(x)
        assert_type(x1, Tensor[[B, 64, 256, 256]])
        x2 = self.down1(x1)
        assert_type(x2, Tensor[[B, 128, 128, 128]])
        x3 = self.down2(x2)
        assert_type(x3, Tensor[[B, 256, 64, 64]])
        x4 = self.down3(x3)
        assert_type(x4, Tensor[[B, 512, 32, 32]])
        x5 = self.down4(x4)
        assert_type(x5, Tensor[[B, 512, 16, 16]])

        # Decoder with skip connections
        d4 = self.up1(x5, x4)
        assert_type(d4, Tensor[[B, 256, 32, 32]])
        d3 = self.up2(d4, x3)
        assert_type(d3, Tensor[[B, 128, 64, 64]])
        d2 = self.up3(d3, x2)
        assert_type(d2, Tensor[[B, 64, 128, 128]])
        d1 = self.up4(d2, x1)
        assert_type(d1, Tensor[[B, 64, 256, 256]])

        logits = self.outc(d1)
        return logits


# ============================================================================
# Smoke tests
# ============================================================================


def test_double_conv():
    """Test spatial-preserving double convolution."""
    conv = DoubleConv(3, 64)
    x: Tensor[[4, 3, 256, 256]] = torch.randn(4, 3, 256, 256)
    out = conv(x)
    assert_type(out, Tensor[[4, 64, 256, 256]])


def test_double_conv_mid_channels():
    """Test double conv with explicit mid_channels (used in bilinear Up)."""
    conv = DoubleConv(1024, 256, c_mid=512)
    x: Tensor[[4, 1024, 32, 32]] = torch.randn(4, 1024, 32, 32)
    out = conv(x)
    assert_type(out, Tensor[[4, 256, 32, 32]])


def test_down():
    """Test downsampling block: halves spatial dims, transforms channels."""
    down = Down(64, 128)
    x: Tensor[[4, 64, 256, 256]] = torch.randn(4, 64, 256, 256)
    out = down(x)
    assert_type(out, Tensor[[4, 128, 128, 128]])


def test_up():
    """Test upsampling block with transposed convolution and skip connection."""
    up = Up(512, 512)
    x1: Tensor[[4, 1024, 16, 16]] = torch.randn(4, 1024, 16, 16)
    x2: Tensor[[4, 512, 32, 32]] = torch.randn(4, 512, 32, 32)
    out = up(x1, x2)
    assert_type(out, Tensor[[4, 512, 32, 32]])


def test_up_bilinear():
    """Test upsampling block with bilinear interpolation and skip connection."""
    up = UpBilinear(512, 256)
    x1: Tensor[[4, 512, 16, 16]] = torch.randn(4, 512, 16, 16)
    x2: Tensor[[4, 512, 32, 32]] = torch.randn(4, 512, 32, 32)
    out = up(x1, x2)
    assert_type(out, Tensor[[4, 256, 32, 32]])


def test_out_conv():
    """Test 1x1 output convolution."""
    outc = OutConv(64, 2)
    x: Tensor[[4, 64, 256, 256]] = torch.randn(4, 64, 256, 256)
    out = outc(x)
    assert_type(out, Tensor[[4, 2, 256, 256]])


def test_unet():
    """End-to-end: non-bilinear UNet for 2-class segmentation on 256x256 input."""
    model = UNet(3, 2)
    x: Tensor[[1, 3, 256, 256]] = torch.randn(1, 3, 256, 256)
    out = model(x)
    assert_type(out, Tensor[[1, 2, 256, 256]])


def test_unet_bilinear():
    """End-to-end: bilinear UNet for 2-class segmentation on 256x256 input."""
    model = UNetBilinear(3, 2)
    x: Tensor[[1, 3, 256, 256]] = torch.randn(1, 3, 256, 256)
    out = model(x)
    assert_type(out, Tensor[[1, 2, 256, 256]])
