# Portions (c) Meta Platforms, Inc. and affiliates.
#
# This source code is adapted from pytorch/benchmark (TorchBenchmark),
# which is licensed under the BSD 3-Clause License:
# https://github.com/pytorch/benchmark/blob/main/LICENSE
#
# This adaptation adds tensor shape type annotations for pyrefly.

"""
Super SloMo from TorchBenchmark with shape annotations.

Original: pytorch/benchmark/torchbenchmark/models/Super_SloMo/slomo_model.py

Port notes:
- Uses nn.AvgPool2d for spatial shape tracking (DSL redirect computes shapes)
- Uses scale_factor=2 (int) instead of 2.0 (float) for F.interpolate
    (the DSL's interpolate_ir expects int|symint for scale_factor)
- Uses (k - 1) // 2 instead of int((filterSize - 1) / 2) for padding
    (equivalent for odd kernel sizes, keeps Dim type tracking)
- Variable reassignment with shape change requires unique variable names
- backWarp ported as BackWarp[W, H]: added torch.meshgrid, expand_as,
    F.grid_sample stubs; register_buffer → nn.Buffer; variable renames
- getFlowCoeff/getWarpCoeff ported: torch.linspace, None-indexing, unary
    negation, permute all work; tensor-as-index (t[ind]) returns shapeless Tensor
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


class Down[InC, OutC](nn.Module):
    """Average Pooling --> Conv + LeakyReLU --> Conv + LeakyReLU

    Halves spatial dims via avg_pool2d(2), transforms channels.
    Conv uses padding=(filterSize-1)//2 to preserve spatial dims after pooling.

    WORKAROUND: filterSize is a plain int (not Dim[K]) because the padding
    expression 2*((K-1)//2) doesn't simplify to K-1 symbolically, causing
    Conv2d spatial formulas to not reduce properly. Using plain int means
    concrete literal values propagate through Conv2d type params at each
    instantiation site.
    """

    def __init__(self, c_in: Dim[InC], c_out: Dim[OutC], filter_size: int) -> None:
        super().__init__()
        padding = (filter_size - 1) // 2
        self.pool = nn.AvgPool2d(2)
        self.conv1 = nn.Conv2d(c_in, c_out, filter_size, stride=1, padding=padding)
        self.conv2 = nn.Conv2d(c_out, c_out, filter_size, stride=1, padding=padding)

    def forward[B, H, W](
        self, x: Tensor[B, InC, H, W]
    ) -> Tensor[B, OutC, (H - 2) // 2 + 1, (W - 2) // 2 + 1]:
        x_pooled = self.pool(x)
        assert_type(x_pooled, Tensor[B, InC, (H - 2) // 2 + 1, (W - 2) // 2 + 1])
        # Note: conv outputs have Any spatial dims in generic body because
        # filter_size is int (WORKAROUND), so padding is int, and Conv2d
        # can't compute the spatial formula. Shape is verified at concrete
        # call sites via the test functions below.
        x1 = F.leaky_relu(self.conv1(x_pooled), negative_slope=0.1)
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.1)
        return x2


class Up[InC, OutC](nn.Module):
    """Bilinear interpolation --> Conv + LeakyReLU --> Cat + Conv + LeakyReLU

    Doubles spatial dims via F.interpolate(scale_factor=2), then concatenates
    skip connection and convolves.

    conv1: InC -> OutC (channel reduction after upsampling)
    conv2: 2*OutC -> OutC (after cat with skip connection of OutC channels)
    """

    def __init__(self, c_in: Dim[InC], c_out: Dim[OutC]) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(2 * c_out, c_out, 3, stride=1, padding=1)

    def forward[B, H, W](
        self, x: Tensor[B, InC, H, W], skp: Tensor[B, OutC, H * 2, W * 2]
    ) -> Tensor[B, OutC, H * 2, W * 2]:
        # WORKAROUND: F.interpolate scale_factor=2 (int) not 2.0 (float)
        # DSL's interpolate_ir expects int|symint for scale_factor
        x_up = F.interpolate(x, scale_factor=2, mode="bilinear")
        assert_type(x_up, Tensor[B, InC, H * 2, W * 2])
        x1 = F.leaky_relu(self.conv1(x_up), negative_slope=0.1)
        assert_type(x1, Tensor[B, OutC, H * 2, W * 2])
        cat_out = torch.cat((x1, skp), 1)
        assert_type(cat_out, Tensor[B, 2 * OutC, H * 2, W * 2])
        x2 = F.leaky_relu(self.conv2(cat_out), negative_slope=0.1)
        assert_type(x2, Tensor[B, OutC, H * 2, W * 2])
        return x2


# ============================================================================
# UNet
# ============================================================================


class UNet[InC, OutC](nn.Module):
    """UNet architecture for Super SloMo.

    Used twice in the full model:
    - flowComp: UNet(6, 4) — computes optical flow from two RGB frames
    - ArbTimeFlowIntrp: UNet(20, 5) — computes flow residuals and visibility maps

    Architecture:
    - Input convolutions with 7x7 kernels (padding=3, spatial-preserving)
    - 4 regular downsampling blocks (C -> 2C) + 1 bottleneck (C -> C)
    - 4 regular upsampling blocks with skip connections + 1 bottleneck
    - Output convolution with 3x3 kernel (padding=1, spatial-preserving)

    Uses list[Stage[Any]] + narrowing annotation for ModuleList dispatch.
    The bottleneck (down5 + up1, both 512->512) is stored separately since
    it preserves channels rather than doubling/halving.
    """

    def __init__(self, c_in: Dim[InC], c_out: Dim[OutC]) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, 32, 7, stride=1, padding=3)
        self.conv2 = nn.Conv2d(32, 32, 7, stride=1, padding=3)
        # Regular encode levels: each doubles channels
        downs: list[Down[Any, Any]] = [
            Down(32, 64, 5),
            Down(64, 128, 3),
            Down(128, 256, 3),
            Down(256, 512, 3),
        ]
        self.downs = nn.ModuleList(downs)
        # Bottleneck: channels stay the same (512 -> 512)
        bn_downs: list[Down[Any, Any]] = [Down(512, 512, 3)]
        self.bn_downs = nn.ModuleList(bn_downs)
        bn_ups: list[Up[Any, Any]] = [Up(512, 512)]
        self.bn_ups = nn.ModuleList(bn_ups)
        # Regular decode levels: each halves channels
        ups: list[Up[Any, Any]] = [
            Up(64, 32),
            Up(128, 64),
            Up(256, 128),
            Up(512, 256),
        ]
        self.ups = nn.ModuleList(ups)
        self.conv3 = nn.Conv2d(32, c_out, 3, stride=1, padding=1)

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
        """Decode one level: restores shape via Up[2*C, C] with skip connection.

        Up expects skp spatial = deep_H * 2, but skip has H.
        ((H-2)//2+1)*2 = H for even H (standard UNet usage), but the type
        checker cannot prove this algebraic identity.
        """
        idx = len(self.ups) - depth
        up: Up[2 * C, C] = self.ups[idx]
        return up(deep, skip)  # type: ignore[bad-argument-type]

    def _bottleneck[B, C, H, W](self, x: Tensor[B, C, H, W]) -> Tensor[B, C, H, W]:
        """Shape-preserving bottleneck: down5 (512->512) + up1 (512->512).

        The last encoder level doesn't double channels (512->512), and the
        first decoder level doesn't halve them (512->512). Together they form
        a shape-preserving bottleneck at the deepest level.

        Same algebraic gap as _decode: ((H-2)//2+1)*2 = H for even H.
        """
        down: Down[C, C] = self.bn_downs[0]
        up: Up[C, C] = self.bn_ups[0]
        deep = down(x)
        return up(deep, x)  # type: ignore[bad-argument-type]

    def recurse[I, B, C, H, W](
        self, x: Tensor[B, C, H, W], depth: Dim[I]
    ) -> Tensor[B, C, H, W]:
        """Shape-preserving recursive encoder-decoder.

        Base case (depth=0): bottleneck (down5 + up1, shape-preserving).
        Inductive step: encode (C -> 2C), recurse (preserves 2C), decode (2C -> C).
        """
        if depth == 0:
            return self._bottleneck(x)
        skip = x
        encoded = self._encode(x, depth)
        middle = self.recurse(encoded, depth - 1)
        decoded = self._decode(skip, middle, depth)
        return decoded

    def forward[B](self, x: Tensor[B, InC, 352, 352]) -> Tensor[B, OutC, 352, 352]:
        x0 = F.leaky_relu(self.conv1(x), negative_slope=0.1)
        assert_type(x0, Tensor[B, 32, 352, 352])
        s1 = F.leaky_relu(self.conv2(x0), negative_slope=0.1)
        assert_type(s1, Tensor[B, 32, 352, 352])
        features = self.recurse(s1, 4)
        assert_type(features, Tensor[B, 32, 352, 352])
        out = F.leaky_relu(self.conv3(features), negative_slope=0.1)
        return out


# ============================================================================
# BackWarp (image warping using optical flow)
# ============================================================================
# Original: slomo_model.py class backWarp
# Warps an image using optical flow via grid_sample.
#
# Changes from original:
# - Class name backWarp → BackWarp (PEP 8)
# - register_buffer("gridX/Y", ...) → self.gridX/Y = nn.Buffer(...)
# - Added type annotations with generic W, H for spatial dims
# - flow[:, 0, :, :] and flow[:, 1, :, :] use integer indexing (reduces rank by 1)
# - expand_as, torch.meshgrid, F.grid_sample stubs added
# - Variable reassignment x → x_coord, y → y_coord (shape changes)


class BackWarp[W, H](nn.Module):
    """Backwarping module: warps an image using optical flow.

    Given optical flow F_0_1 and frame I1, generates I0 via grid_sample.
    Stores precomputed coordinate grids as buffers.
    """

    def __init__(self, width: Dim[W], height: Dim[H], device: torch.device) -> None:
        super().__init__()
        self.W = width
        self.H = height
        gridX, gridY = torch.meshgrid(
            torch.arange(width, device=device),
            torch.arange(height, device=device),
            indexing="xy",
        )
        self.gridX = nn.Buffer(gridX)
        self.gridY = nn.Buffer(gridY)

    def forward[B, C](
        self, img: Tensor[B, C, H, W], flow: Tensor[B, 2, H, W]
    ) -> Tensor[B, C, H, W]:
        # Extract horizontal and vertical flows
        u = flow[:, 0, :, :]
        v = flow[:, 1, :, :]
        x_coord = self.gridX.unsqueeze(0).expand_as(u).to(dtype=u.dtype) + u
        y_coord = self.gridY.unsqueeze(0).expand_as(v).to(dtype=u.dtype) + v
        # Normalize to range [-1, 1]
        x_norm = 2 * (x_coord / self.W - 0.5)
        y_norm = 2 * (y_coord / self.H - 0.5)
        # Stack X and Y into grid [B, H, W, 2]
        grid = torch.stack((x_norm, y_norm), dim=3)
        # Sample pixels using bilinear interpolation
        img_out = F.grid_sample(img, grid)
        return img_out


# ============================================================================
# Flow Coefficient Functions
# ============================================================================
# Original: slomo_model.py getFlowCoeff / getWarpCoeff
# These compute interpolation coefficients for intermediate frame synthesis.
#
# Pattern chain:
#   torch.linspace(...) → Tensor[7]           (1D tensor with 7 time steps)
#   t[ind]              → Tensor              (tensor-as-index → shapeless)
#   C00[None, None, None, :] → Tensor         (None-indexing, but on shapeless input)
#   .permute(3, 0, 1, 2)     → Tensor         (permute of shapeless stays shapeless)
#   .to(device)              → Tensor         (shape-preserving)
#
# Changes from original:
# - Added type annotations on parameters and return
# - linspace gets device/dtype kwargs (stub updated)
# - Tensor-as-index (t[ind]) returns shapeless Tensor — all downstream ops stay shapeless
# - unary negation (-) works on shaped tensors (Type::Tensor added to unop_infer)


def getFlowCoeff(
    indices: Tensor, device: torch.device, dtype: torch.dtype
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Get flow coefficients for intermediate optical flow computation.

    F_t_0 = C00 * F_0_1 + C01 * F_1_0
    F_t_1 = C10 * F_0_1 + C11 * F_1_0

    C00 = -(1 - t) * t,  C01 = t * t
    C10 = (1 - t)^2,     C11 = -t * (1 - t)
    """
    t: Tensor = torch.linspace(0.125, 0.875, 7, device=device, dtype=dtype)
    ind = indices
    # t[ind] is tensor-as-index: returns shapeless Tensor
    C11 = C00 = -(1 - (t[ind])) * (t[ind])
    C01 = (t[ind]) * (t[ind])
    C10 = (1 - (t[ind])) * (1 - (t[ind]))
    return (
        C00[None, None, None, :].permute(3, 0, 1, 2).to(device),
        C01[None, None, None, :].permute(3, 0, 1, 2).to(device),
        C10[None, None, None, :].permute(3, 0, 1, 2).to(device),
        C11[None, None, None, :].permute(3, 0, 1, 2).to(device),
    )


def getWarpCoeff(
    indices: Tensor, device: torch.device, dtype: torch.dtype
) -> tuple[Tensor, Tensor]:
    """Get warp coefficients for intermediate frame synthesis.

    C0 = 1 - t,  C1 = t
    """
    t: Tensor = torch.linspace(0.125, 0.875, 7, device=device, dtype=dtype)
    ind = indices
    C0 = 1 - t[ind]
    C1 = t[ind]
    return (
        C0[None, None, None, :].permute(3, 0, 1, 2).to(device),
        C1[None, None, None, :].permute(3, 0, 1, 2).to(device),
    )


# ============================================================================
# Smoke tests
# ============================================================================


def test_down():
    """Test downsampling block with kernel_size=5."""
    down = Down(32, 64, 5)
    x: Tensor[4, 32, 352, 352] = torch.randn(4, 32, 352, 352)
    out = down(x)
    assert_type(out, Tensor[4, 64, 176, 176])


def test_down_k3():
    """Test downsampling with kernel_size=3."""
    down = Down(64, 128, 3)
    x: Tensor[4, 64, 176, 176] = torch.randn(4, 64, 176, 176)
    out = down(x)
    assert_type(out, Tensor[4, 128, 88, 88])


def test_up():
    """Test upsampling block with skip connection."""
    up_block = Up(512, 512)
    x: Tensor[4, 512, 11, 11] = torch.randn(4, 512, 11, 11)
    skp: Tensor[4, 512, 22, 22] = torch.randn(4, 512, 22, 22)
    out = up_block(x, skp)
    assert_type(out, Tensor[4, 512, 22, 22])


def test_up_channel_change():
    """Test upsampling block that changes channels."""
    up_block = Up(512, 256)
    x: Tensor[4, 512, 22, 22] = torch.randn(4, 512, 22, 22)
    skp: Tensor[4, 256, 44, 44] = torch.randn(4, 256, 44, 44)
    out = up_block(x, skp)
    assert_type(out, Tensor[4, 256, 44, 44])


def test_flow_unet():
    """Test flow computation UNet: 6 input channels (2 RGB frames), 4 output (2 flow fields)."""
    flow_comp = UNet(6, 4)
    frames: Tensor[2, 6, 352, 352] = torch.randn(2, 6, 352, 352)
    flow = flow_comp(frames)
    assert_type(flow, Tensor[2, 4, 352, 352])


def test_interp_unet():
    """Test interpolation UNet: 20 input channels, 5 output (flow residuals + visibility)."""
    interp = UNet(20, 5)
    combined: Tensor[2, 20, 352, 352] = torch.randn(2, 20, 352, 352)
    out = interp(combined)
    assert_type(out, Tensor[2, 5, 352, 352])


def test_back_warp():
    """Test BackWarp: warp image using optical flow.

    Original: backWarp(W=352, H=352, device)
    Uses torch.meshgrid, nn.Buffer, expand_as, F.grid_sample.
    """
    device = torch.device("cpu")
    warp = BackWarp(352, 352, device)
    img: Tensor[2, 3, 352, 352] = torch.randn(2, 3, 352, 352)
    flow: Tensor[2, 2, 352, 352] = torch.randn(2, 2, 352, 352)
    out = warp(img, flow)
    assert_type(out, Tensor[2, 3, 352, 352])
