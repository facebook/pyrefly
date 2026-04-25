# Portions (c) Meta Platforms, Inc. and affiliates.
#
# This source code is adapted from pytorch/benchmark (TorchBenchmark),
# which is licensed under the BSD 3-Clause License:
# https://github.com/pytorch/benchmark/blob/main/LICENSE
#
# This adaptation adds tensor shape type annotations for pyrefly.

"""
DCGAN from TorchBenchmark with shape annotations.

Original: pytorch/benchmark/torchbenchmark/models/dcgan/__init__.py
"""

from typing import Any, assert_type, Final, overload, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from torch import Tensor
    from torch_shapes import Dim


class DCGAN:
    # Number of channels in the training images. For color images this is 3
    nc: Final = 3

    # Size of z latent vector (i.e. size of generator input)
    nz: Final = 100

    # Size of feature maps in generator
    ngf: Final = 64

    # Size of feature maps in discriminator
    ndf: Final = 64


# custom weights initialization called on netG and netD
def weights_init(m: nn.Module) -> None:
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0)


# ============================================================================
# Generic stage blocks
# ============================================================================


class GenUpStage[InC](nn.Module):
    """Generator upsample stage: halves channels, doubles spatial.

    ConvTranspose2d(C, C//2, 4, 2, 1) + BN + ReLU.
    """

    def __init__(self, in_ch: Dim[InC]) -> None:
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_ch, in_ch // 2, 4, 2, 1, bias=False)
        self.bn = nn.BatchNorm2d(in_ch // 2)

    def forward[B, H, W](
        self, x: Tensor[B, InC, H, W]
    ) -> Tensor[B, InC // 2, (H - 1) * 2 + 2, (W - 1) * 2 + 2]:
        return F.relu(self.bn(self.deconv(x)))


class DiscDownStage[InC](nn.Module):
    """Discriminator downsample stage: doubles channels, halves spatial.

    Conv2d(C, 2*C, 4, 2, 1) + BN + LeakyReLU.
    """

    def __init__(self, in_ch: Dim[InC]) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, 2 * in_ch, 4, 2, 1, bias=False)
        self.bn = nn.BatchNorm2d(2 * in_ch)

    def forward[B, H, W](
        self, x: Tensor[B, InC, H, W]
    ) -> Tensor[B, 2 * InC, (H - 2) // 2 + 1, (W - 2) // 2 + 1]:
        return F.leaky_relu(self.bn(self.conv(x)), 0.2)


# ============================================================================
# Generator: recursive upsample chain (Pattern C — exponential)
# ============================================================================


class Generator(nn.Module):
    """DCGAN Generator with recursive upsample chain.

    Architecture:
    - Project: ConvTranspose2d(nz=100, ngf*8=512, 4, 1, 0) → 4×4
    - Upsample chain (3 stages): 512→256→128→64, spatial 4→8→16→32
    - Output: ConvTranspose2d(ngf=64, nc=3, 4, 2, 1) + Tanh → 64×64

    The 3 middle stages use _chain with return type
    Tensor[B, C // 2**I, H * 2**I, W * 2**I].
    """

    def __init__(self) -> None:
        super().__init__()
        # Project: z → ngf*8 × 4 × 4
        self.project = nn.ConvTranspose2d(DCGAN.nz, DCGAN.ngf * 8, 4, 1, 0, bias=False)
        self.project_bn = nn.BatchNorm2d(DCGAN.ngf * 8)
        # Upsample stages (each halves channels, doubles spatial)
        stages: list[GenUpStage[Any]] = [
            GenUpStage(DCGAN.ngf * 8),  # 512 → 256
            GenUpStage(DCGAN.ngf * 4),  # 256 → 128
            GenUpStage(DCGAN.ngf * 2),  # 128 → 64
        ]
        self.up_stages = nn.ModuleList(stages)
        # Output: ngf → nc
        self.output = nn.ConvTranspose2d(DCGAN.ngf, DCGAN.nc, 4, 2, 1, bias=False)

    def _apply_stage[B, C, H, W](
        self, x: Tensor[B, C, H, W], depth: int
    ) -> Tensor[B, C // 2, (H - 1) * 2 + 2, (W - 1) * 2 + 2]:
        idx = len(self.up_stages) - depth
        stage: GenUpStage[C] = self.up_stages[idx]
        return stage(x)

    @overload
    def _chain[B, C, H, W](
        self, x: Tensor[B, C, H, W], depth: Dim[1]
    ) -> Tensor[B, C // 2, H * 2, W * 2]: ...

    @overload
    def _chain[I, B, C, H, W](
        self, x: Tensor[B, C, H, W], depth: Dim[I]
    ) -> Tensor[B, C // 2**I, H * 2**I, W * 2**I]: ...

    def _chain[I, B, C, H, W](
        self, x: Tensor[B, C, H, W], depth: Dim[I]
    ) -> Tensor[B, C // 2, H * 2, W * 2] | Tensor[B, C // 2**I, H * 2**I, W * 2**I]:
        y = self._apply_stage(x, depth)
        if depth == 1:
            return y
        return self._chain(y, depth - 1)

    def forward[B](self, input: Tensor[B, 100, 1, 1]) -> Tensor[B, 3, 64, 64]:
        h0 = F.relu(self.project_bn(self.project(input)))
        assert_type(h0, Tensor[B, 512, 4, 4])
        h1 = self._chain(h0, 3)  # 512→64, 4→32
        assert_type(h1, Tensor[B, 64, 32, 32])
        return torch.tanh(self.output(h1))


# ============================================================================
# Discriminator: recursive downsample chain (Pattern C — exponential)
# ============================================================================


class Discriminator(nn.Module):
    """DCGAN Discriminator with recursive downsample chain.

    Architecture:
    - Input: Conv2d(nc=3, ndf=64, 4, 2, 1) + LeakyReLU → 32×32
    - Downsample chain (3 stages): 64→128→256→512, spatial 32→16→8→4
    - Output: Conv2d(ndf*8=512, 1, 4, 1, 0) + Sigmoid → 1×1

    The 3 middle stages use _chain with return type
    Tensor[B, C * 2**I, H // 2**I, W // 2**I].
    """

    def __init__(self) -> None:
        super().__init__()
        # Input: nc → ndf (no BN — standard DCGAN convention)
        self.input_conv = nn.Conv2d(DCGAN.nc, DCGAN.ndf, 4, 2, 1, bias=False)
        # Downsample stages (each doubles channels, halves spatial)
        stages: list[DiscDownStage[Any]] = [
            DiscDownStage(DCGAN.ndf),  # 64 → 128
            DiscDownStage(DCGAN.ndf * 2),  # 128 → 256
            DiscDownStage(DCGAN.ndf * 4),  # 256 → 512
        ]
        self.down_stages = nn.ModuleList(stages)
        # Output: ndf*8 → 1
        self.output_conv = nn.Conv2d(DCGAN.ndf * 8, 1, 4, 1, 0, bias=False)

    def _apply_stage[B, C, H, W](
        self, x: Tensor[B, C, H, W], depth: int
    ) -> Tensor[B, 2 * C, (H - 2) // 2 + 1, (W - 2) // 2 + 1]:
        idx = len(self.down_stages) - depth
        stage: DiscDownStage[C] = self.down_stages[idx]
        return stage(x)

    @overload
    def _chain[B, C, H, W](
        self, x: Tensor[B, C, H, W], depth: Dim[1]
    ) -> Tensor[B, 2 * C, H // 2, W // 2]: ...

    @overload
    def _chain[I, B, C, H, W](
        self, x: Tensor[B, C, H, W], depth: Dim[I]
    ) -> Tensor[B, C * 2**I, H // 2**I, W // 2**I]: ...

    def _chain[I, B, C, H, W](
        self, x: Tensor[B, C, H, W], depth: Dim[I]
    ) -> Tensor[B, 2 * C, H // 2, W // 2] | Tensor[B, C * 2**I, H // 2**I, W // 2**I]:
        y = self._apply_stage(x, depth)
        if depth == 1:
            return y
        return self._chain(y, depth - 1)

    def forward[B](self, input: Tensor[B, 3, 64, 64]) -> Tensor[B, 1, 1, 1]:
        h0 = F.leaky_relu(self.input_conv(input), 0.2)
        assert_type(h0, Tensor[B, 64, 32, 32])
        h1 = self._chain(h0, 3)  # 64→512, 32→4
        assert_type(h1, Tensor[B, 512, 4, 4])
        return torch.sigmoid(self.output_conv(h1))


# ============================================================================
# Smoke test
# ============================================================================


def test_disc_down_stage():
    """Test single discriminator stage: doubles channels, halves spatial."""
    stage = DiscDownStage(64)
    x: Tensor[4, 64, 32, 32] = torch.randn(4, 64, 32, 32)
    out = stage(x)
    assert_type(out, Tensor[4, 128, 16, 16])


def test_gen_up_stage():
    """Test single generator stage: halves channels, doubles spatial."""
    stage = GenUpStage(512)
    x: Tensor[4, 512, 4, 4] = torch.randn(4, 512, 4, 4)
    out = stage(x)
    assert_type(out, Tensor[4, 256, 8, 8])


def test_generator():
    netG = Generator()
    noise: Tensor[64, 100, 1, 1] = torch.randn(64, 100, 1, 1)
    fake = netG(noise)
    assert_type(fake, Tensor[64, 3, 64, 64])


def test_discriminator():
    netD = Discriminator()
    img: Tensor[32, 3, 64, 64] = torch.randn(32, 3, 64, 64)
    out = netD(img)
    assert_type(out, Tensor[32, 1, 1, 1])


def test_gan_pipeline():
    """End-to-end: generate fake images, then discriminate them."""
    netG = Generator()
    netD = Discriminator()

    noise: Tensor[16, 100, 1, 1] = torch.randn(16, 100, 1, 1)
    fake = netG(noise)
    assert_type(fake, Tensor[16, 3, 64, 64])

    verdict = netD(fake)
    assert_type(verdict, Tensor[16, 1, 1, 1])
