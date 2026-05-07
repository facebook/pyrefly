# Portions (c) Meta Platforms, Inc. and affiliates.
#
# This source code is adapted from pytorch/benchmark (TorchBenchmark),
# which is licensed under the BSD 3-Clause License:
# https://github.com/pytorch/benchmark/blob/main/LICENSE
#
# This adaptation adds tensor shape type annotations for pyrefly.

"""
StarGAN conditional image-to-image translation from TorchBenchmark with shape annotations.

Original: pytorch/benchmark/torchbenchmark/models/pytorch_stargan/model.py

Port notes:
- Generator: Encoder-decoder with condition injection and residual bottleneck.
  Condition (B, CDim) is broadcast to (B, CDim, H, W) and concatenated with the
  image along the channel dimension before encoding. No skip connections (simpler
  than UNet).
- Discriminator: PatchGAN with dual heads — source (real/fake patch prediction)
  and classification (domain prediction).
- Original builds Generator as one big nn.Sequential and Discriminator backbone
  as nn.Sequential. Port preserves this structure using direct nn.Sequential
  construction (the original uses list-based construction which doesn't type-track).
- Fixed conv_dim=64 (default) and 128x128 image size (StarGAN standard).
  repeat_num=6 for both Generator bottleneck and Discriminator depth.
- Condition broadcast uses .view().expand() — both operations are fully
  shape-tracked when args are Dim values or literals. Verified by assert_type.

Key patterns exercised:
- nn.Sequential for large pipelines (faithful to original)
- Condition injection: broadcast (B, CDim) → (B, CDim, 128, 128) + concat with image
- Shape-preserving residual block with InstanceNorm2d
- Encoder-decoder without skip connections (contrast with UNet)
- Dual-head discriminator: patch source + domain classifier
"""

from typing import assert_type, TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from torch import Tensor
    from torch_shapes import Dim


# ============================================================================
# Residual Block
# ============================================================================


class ResidualBlock[C](nn.Module):
    """Shape-preserving residual block with InstanceNorm2d.

    Conv2d(C, C, 3, 1, 1) → InstanceNorm → ReLU → Conv2d(C, C, 3, 1, 1) → InstanceNorm
    + skip connection.

    (B, C, H, W) → (B, C, H, W)
    """

    def __init__(self, dim: Dim[C]) -> None:
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(dim, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(dim, affine=True, track_running_stats=True),
        )

    def forward[B, H, W](self, x: Tensor[B, C, H, W]) -> Tensor[B, C, H, W]:
        return x + self.conv_block(x)


# ============================================================================
# Generator (128x128, conv_dim=64, repeat_num=6)
# ============================================================================


class Generator[CDim](nn.Module):
    """StarGAN generator: image + condition → translated image.

    Architecture (128x128, conv_dim=64):
        Condition injection: (B, CDim) → broadcast → (B, CDim, 128, 128)
          → cat with image → (B, 3+CDim, 128, 128)
        Main pipeline (nn.Sequential):
          Encoder:
            Conv2d(3+CDim, 64, 7, 1, 3) → IN → ReLU → (B, 64, 128, 128)
            Conv2d(64, 128, 4, 2, 1) → IN → ReLU → (B, 128, 64, 64)
            Conv2d(128, 256, 4, 2, 1) → IN → ReLU → (B, 256, 32, 32)
          Bottleneck: 6 × ResidualBlock[256] → (B, 256, 32, 32)
          Decoder:
            ConvTranspose2d(256, 128, 4, 2, 1) → IN → ReLU → (B, 128, 64, 64)
            ConvTranspose2d(128, 64, 4, 2, 1) → IN → ReLU → (B, 64, 128, 128)
          Output:
            Conv2d(64, 3, 7, 1, 3) → Tanh → (B, 3, 128, 128)

    (B, 3, 128, 128), (B, CDim) → (B, 3, 128, 128)
    """

    def __init__(self, c_dim: Dim[CDim]) -> None:
        super().__init__()
        self.main = nn.Sequential(
            # Encoder
            nn.Conv2d(3 + c_dim, 64, kernel_size=7, stride=1, padding=3),
            nn.InstanceNorm2d(64, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256, affine=True, track_running_stats=True),
            nn.ReLU(),
            # Bottleneck: 6 ResidualBlocks (shape-preserving)
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            # Decoder
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64, affine=True, track_running_stats=True),
            nn.ReLU(),
            # Output
            nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=3),
            nn.Tanh(),
        )

    def forward[B, S](
        self, x: Tensor[B, 3, S, S], c: Tensor[B, CDim]
    ) -> Tensor[B, 3, S, S]:
        # Condition injection: broadcast (B, CDim) → (B, CDim, S, S)
        h, w = x.shape[2], x.shape[3]
        c_4d = c.view(c.size(0), c.size(1), 1, 1)
        assert_type(c_4d, Tensor[B, CDim, 1, 1])
        c_spatial = c_4d.expand(-1, -1, h, w)
        assert_type(c_spatial, Tensor[B, CDim, S, S])
        x_c = torch.cat((x, c_spatial), dim=1)
        assert_type(x_c, Tensor[B, 3 + CDim, S, S])
        out = self.main(x_c)
        # A1: conv chain produces 4*(S//4), can't prove = S
        return out  # type: ignore[bad-return]


# ============================================================================
# Discriminator (128x128, conv_dim=64, repeat_num=6)
# ============================================================================


class Discriminator[CDim, S](nn.Module):
    """StarGAN PatchGAN discriminator with dual heads.

    Architecture (conv_dim=64, repeat_num=6):
        Backbone: 6 × Conv2d(k=4, s=2, p=1) + LReLU → S//64 spatial
        Dual heads:
          src: Conv2d(2048, 1, 3, 1, 1) → (B, 1, S//64, S//64)
          cls: Conv2d(2048, CDim, k=S//64) → (B, CDim, 1, 1) → view → (B, CDim)

    (B, 3, S, S) → (Tensor[B, 1, S//64, S//64], Tensor[B, CDim])
    """

    def __init__(self, c_dim: Dim[CDim], image_size: Dim[S]) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(1024, 2048, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.01),
        )
        # Dual heads: cls kernel adapts to image size (original pattern)
        self.conv_src = nn.Conv2d(2048, 1, kernel_size=3, stride=1, padding=1)
        self.conv_cls = nn.Conv2d(2048, c_dim, kernel_size=image_size // 64)

    def forward[B](
        self, x: Tensor[B, 3, S, S]
    ) -> tuple[Tensor[B, 1, S // 64, S // 64], Tensor[B, CDim]]:
        h = self.main(x)
        out_src = self.conv_src(h)
        out_cls = self.conv_cls(h)
        # cls Conv2d(k=S//64) on spatial S//64 → (S//64 - S//64)//1 + 1 = 1
        out_cls_flat = out_cls.view(out_cls.size(0), -1)
        return out_src, out_cls_flat


# ============================================================================
# Smoke tests
# ============================================================================


def test_residual_block():
    """Test shape-preserving residual block."""
    block = ResidualBlock(256)
    x: Tensor[4, 256, 32, 32] = torch.randn(4, 256, 32, 32)
    out = block(x)
    assert_type(out, Tensor[4, 256, 32, 32])


def test_generator():
    """Test generator: image + condition → translated image."""
    gen = Generator(5)
    img: Tensor[4, 3, 128, 128] = torch.randn(4, 3, 128, 128)
    cond: Tensor[4, 5] = torch.randn(4, 5)
    out = gen(img, cond)
    assert_type(out, Tensor[4, 3, 128, 128])


def test_generator_different_cdim():
    """Test generator with different condition dimension."""
    gen = Generator(10)
    img: Tensor[2, 3, 128, 128] = torch.randn(2, 3, 128, 128)
    cond: Tensor[2, 10] = torch.randn(2, 10)
    out = gen(img, cond)
    assert_type(out, Tensor[2, 3, 128, 128])


def test_discriminator():
    """Test discriminator: image → (patch_src, domain_cls)."""
    disc = Discriminator(5, 128)
    img: Tensor[4, 3, 128, 128] = torch.randn(4, 3, 128, 128)
    src, cls = disc(img)
    assert_type(src, Tensor[4, 1, 2, 2])
    assert_type(cls, Tensor[4, 5])


def test_discriminator_different_cdim():
    """Test discriminator with different condition dimension."""
    disc = Discriminator(10, 128)
    img: Tensor[2, 3, 128, 128] = torch.randn(2, 3, 128, 128)
    src, cls = disc(img)
    assert_type(src, Tensor[2, 1, 2, 2])
    assert_type(cls, Tensor[2, 10])
