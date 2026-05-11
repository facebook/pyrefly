# Portions (c) Meta Platforms, Inc. and affiliates.
#
# This source code is adapted from pytorch/benchmark (TorchBenchmark),
# which is licensed under the BSD 3-Clause License:
# https://github.com/pytorch/benchmark/blob/main/LICENSE
#
# This adaptation adds tensor shape type annotations for pyrefly.

"""
Background Matting conditional image generation from TorchBenchmark with shape annotations.

Original: pytorch/benchmark/torchbenchmark/models/Background_Matting/networks.py

Port notes:
- Generator (ResnetConditionHR): multi-branch encoder-decoder with condition
  fusion and dual-head output (alpha matte + foreground).
  - 4 encoder branches: image, background, segmentation, multi-frame.
    Each branch uses nn.Sequential: ReflPad(3)+Conv2d(7)+BN+ReLU → 2× strided Conv2d
    → (B, 256, H//4, W//4).
  - Cross-branch fusion: concat img_feat with each branch → 1×1 conv → 64 channels.
    Then concat all 3 fused features → (B, 192, H//4, W//4).
  - Bottleneck: concat(img_feat, fused) → 1×1 conv → 7 ResBlocks → (B, 256, H//4, W//4).
  - Alpha branch: 3 ResBlocks → 2× upsample+conv → ReflPad+Conv → tanh → (B, 1, H, W).
  - Foreground branch: 3 ResBlocks → upsample+conv → skip concat with enc1
    → upsample+conv → ReflPad+Conv → (B, 3, H, W).
- Discriminator (NLayerDiscriminator): PatchGAN with 3 strided conv layers.
  Built as nn.Sequential (faithful to original). Fixed ndf=64, n_layers=3.
- ResnetBlock: conv_block as nn.Sequential + skip connection (faithful to original).
- nn.Upsample(scale_factor=2) doubles spatial dims.
- Fixed ngf=64, nf_part=64 throughout.

Key patterns exercised:
- nn.Sequential for encoder branches, ResBlock conv path, discriminator backbone
- Multi-branch encoder with cross-branch fusion via concatenation
- Dual-head decoder (alpha + foreground) with skip connection
- ReflectionPad2d + Conv2d for shape-preserving convolution
- nn.Upsample for spatial upsampling
- Feature concatenation across multiple branches
- Weight initialization (conv_init)
- Helper module factories (conv3x3, conv1x1, etc.) — present in original but
  only used by other models in the same file, not by ResnetConditionHR or
  NLayerDiscriminator. Included as factories for completeness.
"""

from typing import assert_type, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from torch import Tensor
    from torch_shapes import Dim


# ============================================================================
# Weight Initialization
# ============================================================================


def conv_init(m: nn.Module) -> None:
    """Initialize Conv2d and BatchNorm2d weights.

    Original: networks.py conv_init function.
    Applied via model.apply(conv_init).
    """
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)


# ============================================================================
# Helper Module Factories
# ============================================================================
# These are present in the original networks.py but only used by other models
# (not ResnetConditionHR or NLayerDiscriminator). Included for completeness.


def conv3x3[InC, OutC](in_channels: Dim[InC], out_channels: Dim[OutC]) -> nn.Sequential:
    """3×3 conv + BN + ReLU, shape-preserving (padding=1)."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )


def conv1x1[InC, OutC](in_channels: Dim[InC], out_channels: Dim[OutC]) -> nn.Sequential:
    """1×1 conv + BN + ReLU, changes channels only."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )


def upconv3x3[InC, OutC](
    in_channels: Dim[InC], out_channels: Dim[OutC]
) -> nn.Sequential:
    """Upsample(2×) + 3×3 conv + BN + ReLU."""
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear"),
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )


# ============================================================================
# ResnetBlock (shape-preserving)
# ============================================================================


class ResnetBlock[C](nn.Module):
    """Residual block with ReflectionPad2d.

    conv_block = nn.Sequential(
        ReflPad(1) + Conv2d(C, C, 3, pad=0) + BN + ReLU
        + ReflPad(1) + Conv2d(C, C, 3, pad=0) + BN
    ) + skip.

    ReflPad(1) adds 1 px per side → (H+2, W+2).
    Conv2d(3, pad=0) on (H+2, W+2) → ((H+2)-3+1) = H. Shape-preserving.

    (B, C, H, W) → (B, C, H, W)
    """

    def __init__(self, dim: Dim[C]) -> None:
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0),
            nn.BatchNorm2d(dim),
        )

    def forward[B, H, W](self, x: Tensor[B, C, H, W]) -> Tensor[B, C, H, W]:
        return x + self.conv_block(x)


# ============================================================================
# Encoder Branch
# ============================================================================


class EncoderBranch[InC](nn.Module):
    """Single encoder branch: input → 256 channels at 1/4 spatial resolution.

    Architecture (ngf=64), built as nn.Sequential:
        ReflPad(3) + Conv2d(InC, 64, 7, pad=0) → BN → ReLU → (B, 64, H, W)
        Conv2d(64, 128, 3, stride=2, pad=1) → BN → ReLU → (B, 128, H//2, W//2)
        Conv2d(128, 256, 3, stride=2, pad=1) → BN → ReLU → (B, 256, H//4, W//4)

    (B, InC, H, W) → (B, 256, (H-1)//4+1, (W-1)//4+1)
    Note: For H divisible by 4, (H-1)//4+1 is not exactly H//4 but close.
    We use H, W that are multiples of 4 in practice.
    """

    def __init__(self, in_channels: Dim[InC]) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, 64, kernel_size=7, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

    def forward[B, H, W](
        self, x: Tensor[B, InC, H, W]
    ) -> Tensor[B, 256, (H - 1) // 4 + 1, (W - 1) // 4 + 1]:
        return self.model(x)


# ============================================================================
# Image Encoder (first branch, returns intermediate for skip connection)
# ============================================================================


class ImageEncoder(nn.Module):
    """Image encoder with two stages, exposing intermediate for skip connection.

    Stage 1 (nn.Sequential):
        ReflPad(3) + Conv2d(3, 64, 7) → BN → ReLU → (B, 64, H, W)
        Conv2d(64, 128, 3, stride=2, pad=1) → BN → ReLU → (B, 128, H', W')
    Stage 2 (nn.Sequential):
        Conv2d(128, 256, 3, stride=2, pad=1) → BN → ReLU → (B, 256, H'', W'')

    Returns (stage1_output, stage2_output) for skip connection.
    """

    def __init__(self) -> None:
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, 64, kernel_size=7, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

    def forward[B, H, W](
        self, x: Tensor[B, 3, H, W]
    ) -> tuple[
        Tensor[B, 128, (H - 1) // 2 + 1, (W - 1) // 2 + 1],
        Tensor[B, 256, (H - 1) // 4 + 1, (W - 1) // 4 + 1],
    ]:
        feat1 = self.stage1(x)
        assert_type(feat1, Tensor[B, 128, (H - 1) // 2 + 1, (W - 1) // 2 + 1])
        feat2 = self.stage2(feat1)
        return feat1, feat2


# ============================================================================
# Generator (ResnetConditionHR, ngf=64)
# ============================================================================


class Generator(nn.Module):
    """Background matting generator with multi-branch encoder and dual-head decoder.

    Inputs (all at same spatial resolution H×W):
        image: (B, 3, H, W)
        back:  (B, 3, H, W)     — background image
        seg:   (B, 1, H, W)     — segmentation mask
        multi: (B, 4, H, W)     — multi-frame features

    Outputs:
        alpha: (B, 1, H, W)     — alpha matte (tanh)
        fg:    (B, 3, H, W)     — foreground

    Architecture (ngf=64, nf_part=64, n_blocks1=7, n_blocks2=3):
        4 encoder branches → cross-branch fusion → bottleneck
        → alpha branch (3 ResBlocks + upsample) + fg branch (3 ResBlocks + skip + upsample)

    Uses concrete spatial dim 256×256 for the full forward shape annotations.
    """

    def __init__(self) -> None:
        super().__init__()
        # Image encoder (separate for skip connection)
        self.img_encoder = ImageEncoder()
        # Other branch encoders
        self.back_encoder = EncoderBranch(3)
        self.seg_encoder = EncoderBranch(1)
        self.multi_encoder = EncoderBranch(4)
        # Cross-branch fusion: cat(img_feat, branch_feat) → 1×1 conv
        # Each takes (B, 512, H', W') → (B, 64, H', W')
        self.comb_back = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.comb_seg = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.comb_multi = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        # Bottleneck: cat(img_feat, oth_feat) = (256+192) → 1×1 conv → 256
        self.dec_reduce = nn.Sequential(
            nn.Conv2d(448, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        # Shared ResBlocks (7 blocks)
        self.res_shared = nn.Sequential(
            ResnetBlock(256),
            ResnetBlock(256),
            ResnetBlock(256),
            ResnetBlock(256),
            ResnetBlock(256),
            ResnetBlock(256),
            ResnetBlock(256),
        )
        # Alpha branch: 3 ResBlocks + 2× upsample
        self.al_res = nn.Sequential(
            ResnetBlock(256),
            ResnetBlock(256),
            ResnetBlock(256),
        )
        self.al_up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.al_up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.al_out = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 1, kernel_size=7, padding=0),
            nn.Tanh(),
        )
        # Foreground branch: 3 ResBlocks + upsample + skip + upsample
        self.fg_res = nn.Sequential(
            ResnetBlock(256),
            ResnetBlock(256),
            ResnetBlock(256),
        )
        self.fg_up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        # After skip concat: 128 (upsampled) + 128 (enc1) = 256
        self.fg_up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.fg_out = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, kernel_size=7, padding=0),
        )

    def forward[B](
        self,
        image: Tensor[B, 3, 256, 256],
        back: Tensor[B, 3, 256, 256],
        seg: Tensor[B, 1, 256, 256],
        multi: Tensor[B, 4, 256, 256],
    ) -> tuple[Tensor[B, 1, 256, 256], Tensor[B, 3, 256, 256]]:
        # Encode image (with skip connection)
        img_feat1, img_feat = self.img_encoder(image)
        assert_type(img_feat1, Tensor[B, 128, 128, 128])
        assert_type(img_feat, Tensor[B, 256, 64, 64])
        # Encode other branches
        back_feat = self.back_encoder(back)
        assert_type(back_feat, Tensor[B, 256, 64, 64])
        seg_feat = self.seg_encoder(seg)
        assert_type(seg_feat, Tensor[B, 256, 64, 64])
        multi_feat = self.multi_encoder(multi)
        assert_type(multi_feat, Tensor[B, 256, 64, 64])
        # Cross-branch fusion
        comb_b = self.comb_back(torch.cat((img_feat, back_feat), dim=1))
        assert_type(comb_b, Tensor[B, 64, 64, 64])
        comb_s = self.comb_seg(torch.cat((img_feat, seg_feat), dim=1))
        assert_type(comb_s, Tensor[B, 64, 64, 64])
        comb_m = self.comb_multi(torch.cat((img_feat, multi_feat), dim=1))
        assert_type(comb_m, Tensor[B, 64, 64, 64])
        # Concat fused features
        oth_feat = torch.cat((comb_b, comb_s, comb_m), dim=1)
        assert_type(oth_feat, Tensor[B, 192, 64, 64])
        # Bottleneck: reduce channels + ResBlocks
        dec = self.dec_reduce(torch.cat((img_feat, oth_feat), dim=1))
        assert_type(dec, Tensor[B, 256, 64, 64])
        dec = self.res_shared(dec)
        assert_type(dec, Tensor[B, 256, 64, 64])
        # Alpha branch
        al = self.al_res(dec)
        assert_type(al, Tensor[B, 256, 64, 64])
        al = self.al_up1(al)
        assert_type(al, Tensor[B, 128, 128, 128])
        al = self.al_up2(al)
        assert_type(al, Tensor[B, 64, 256, 256])
        alpha = self.al_out(al)
        assert_type(alpha, Tensor[B, 1, 256, 256])
        # Foreground branch
        fg = self.fg_res(dec)
        assert_type(fg, Tensor[B, 256, 64, 64])
        fg_up = self.fg_up1(fg)
        assert_type(fg_up, Tensor[B, 128, 128, 128])
        # Skip connection: concat with img_feat1
        fg_skip = torch.cat((fg_up, img_feat1), dim=1)
        assert_type(fg_skip, Tensor[B, 256, 128, 128])
        fg_out = self.fg_up2(fg_skip)
        assert_type(fg_out, Tensor[B, 64, 256, 256])
        foreground = self.fg_out(fg_out)
        assert_type(foreground, Tensor[B, 3, 256, 256])
        return alpha, foreground


# ============================================================================
# Discriminator (NLayerDiscriminator, ndf=64, n_layers=3)
# ============================================================================


class Discriminator(nn.Module):
    """PatchGAN discriminator with fixed ndf=64, n_layers=3.

    Built as nn.Sequential (faithful to original).
    Channel sequence: 3→64→128→256→512→1
    All with kernel_size=4, padding=2.
    First 3 layers have stride=2, last 2 have stride=1.

    (B, 3, H, W) → (B, 1, H', W') where H' ≈ H//8
    """

    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            # Layer 0: Conv2d(3, 64, 4, stride=2, pad=2) + LReLU
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=2),
            nn.LeakyReLU(0.2),
            # Layer 1: Conv2d(64, 128, 4, stride=2, pad=2) + BN + LReLU
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            # Layer 2: Conv2d(128, 256, 4, stride=2, pad=2) + BN + LReLU
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            # Layer 3: Conv2d(256, 512, 4, stride=1, pad=2) + BN + LReLU
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            # Output: Conv2d(512, 1, 4, stride=1, pad=2)
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=2),
        )

    def forward[B, H, W](
        self, x: Tensor[B, 3, H, W]
    ) -> Tensor[
        B, 1, 3 + (1 + (1 + H // 2) // 2) // 2, 3 + (1 + (1 + W // 2) // 2) // 2
    ]:
        return self.model(x)


# ============================================================================
# MultiscaleDiscriminator
# ============================================================================


class MultiscaleDiscriminator(nn.Module):
    """Multi-scale PatchGAN discriminator.

    Original: Background_Matting/networks.py MultiscaleDiscriminator class.

    Runs num_D discriminators at progressively downsampled scales.
    Each scale applies an NLayerDiscriminator (Discriminator class above).
    Dynamic module registration via setattr/getattr — modules are stored as
    "layer0", "layer1", etc. This pattern is not type-trackable.

    Returns a list of discriminator outputs, one per scale.
    """

    def __init__(
        self,
        num_D: int = 3,
        n_layers: int = 3,
        ndf: int = 64,
    ) -> None:
        super().__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)

        for i in range(num_D):
            netD = Discriminator()
            # Dynamic module registration — not type-trackable (setattr/getattr)
            setattr(self, "layer" + str(i), netD.model)

    def forward(self, x: Tensor) -> list[Tensor]:
        """Run each discriminator at progressively downsampled scales.

        Returns list of discriminator outputs (one per scale).
        All outputs are bare Tensor because:
        1. Dynamic attribute access via getattr returns unrefined
        2. Input spatial dims change at each scale (data-dependent downsampling)
        """
        results: list[Tensor] = []
        inp: Tensor = x
        for i in range(self.num_D):
            model: nn.Sequential = getattr(self, "layer" + str(i))  # type: ignore[assignment]
            results.append(model(inp))
            if i != self.num_D - 1:
                inp = self.downsample(inp)
        return results


# ============================================================================
# Smoke tests
# ============================================================================


def test_resnet_block():
    """Test shape-preserving residual block."""
    block = ResnetBlock(256)
    x: Tensor[2, 256, 64, 64] = torch.randn(2, 256, 64, 64)
    out = block(x)
    assert_type(out, Tensor[2, 256, 64, 64])


def test_encoder_branch():
    """Test encoder branch: (B, 3, 256, 256) → (B, 256, 64, 64)."""
    enc = EncoderBranch(3)
    x: Tensor[2, 3, 256, 256] = torch.randn(2, 3, 256, 256)
    out = enc(x)
    assert_type(out, Tensor[2, 256, 64, 64])


def test_encoder_branch_seg():
    """Test encoder branch with 1-channel input (segmentation)."""
    enc = EncoderBranch(1)
    x: Tensor[2, 1, 256, 256] = torch.randn(2, 1, 256, 256)
    out = enc(x)
    assert_type(out, Tensor[2, 256, 64, 64])


def test_image_encoder():
    """Test image encoder with skip connection output."""
    enc = ImageEncoder()
    x: Tensor[2, 3, 256, 256] = torch.randn(2, 3, 256, 256)
    feat1, feat2 = enc(x)
    assert_type(feat1, Tensor[2, 128, 128, 128])
    assert_type(feat2, Tensor[2, 256, 64, 64])


def test_generator():
    """Test full generator: 4 inputs → (alpha, foreground)."""
    gen = Generator()
    image: Tensor[2, 3, 256, 256] = torch.randn(2, 3, 256, 256)
    back: Tensor[2, 3, 256, 256] = torch.randn(2, 3, 256, 256)
    seg: Tensor[2, 1, 256, 256] = torch.randn(2, 1, 256, 256)
    multi: Tensor[2, 4, 256, 256] = torch.randn(2, 4, 256, 256)
    alpha, fg = gen(image, back, seg, multi)
    assert_type(alpha, Tensor[2, 1, 256, 256])
    assert_type(fg, Tensor[2, 3, 256, 256])


def test_discriminator():
    """Test PatchGAN discriminator: (B, 3, 256, 256) → (B, 1, 35, 35)."""
    disc = Discriminator()
    x: Tensor[2, 3, 256, 256] = torch.randn(2, 3, 256, 256)
    out = disc(x)
    assert_type(out, Tensor[2, 1, 35, 35])


def test_multiscale_discriminator():
    """Test multi-scale discriminator: returns list of outputs at different scales."""
    msd = MultiscaleDiscriminator(num_D=3)
    x: Tensor[2, 3, 256, 256] = torch.randn(2, 3, 256, 256)
    results = msd(x)
    # Returns list of bare Tensor — setattr/getattr and variable-scale inputs
    # prevent shape tracking
    assert_type(results, list[Tensor])
