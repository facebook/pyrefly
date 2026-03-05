# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
DCGAN from TorchBenchmark with shape annotations.

Original: pytorch/benchmark/torchbenchmark/models/dcgan/__init__.py
See model_port_changes.md for full change analysis.
"""

from typing import assert_type, Final, TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from torch import Tensor


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


class Generator(nn.Module):
    def __init__(self, dcgan: DCGAN) -> None:
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(dcgan.nz, dcgan.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(dcgan.ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(dcgan.ngf * 8, dcgan.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dcgan.ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(dcgan.ngf * 4, dcgan.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dcgan.ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(dcgan.ngf * 2, dcgan.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dcgan.ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(dcgan.ngf, dcgan.nc, 4, 2, 1, bias=False),
            nn.Tanh(),
            # state size. (nc) x 64 x 64
        )

    def forward[B](self, input: Tensor[B, 100, 1, 1]) -> Tensor[B, 3, 64, 64]:
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, ncgan: DCGAN) -> None:
        super(Discriminator, self).__init__()
        nc = ncgan.nc
        ndf = ncgan.ndf

        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
            # state size. 1 x 1 x 1
        )

    def forward[B](self, input: Tensor[B, 3, 64, 64]) -> Tensor[B, 1, 1, 1]:
        return self.main(input)


# ============================================================================
# Smoke test
# ============================================================================


def test_generator():
    dcgan = DCGAN()
    netG = Generator(dcgan)
    noise: Tensor[64, 100, 1, 1] = torch.randn(64, 100, 1, 1)
    fake = netG(noise)
    assert_type(fake, Tensor[64, 3, 64, 64])


def test_discriminator():
    dcgan = DCGAN()
    netD = Discriminator(dcgan)
    img: Tensor[32, 3, 64, 64] = torch.randn(32, 3, 64, 64)
    out = netD(img)
    assert_type(out, Tensor[32, 1, 1, 1])


def test_gan_pipeline():
    """End-to-end: generate fake images, then discriminate them."""
    dcgan = DCGAN()
    netG = Generator(dcgan)
    netD = Discriminator(dcgan)

    noise: Tensor[16, 100, 1, 1] = torch.randn(16, 100, 1, 1)
    fake = netG(noise)
    assert_type(fake, Tensor[16, 3, 64, 64])

    verdict = netD(fake)
    assert_type(verdict, Tensor[16, 1, 1, 1])
