# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Minimal test for recursive encoder-decoder with skip connections.

Tests whether Pyrefly can type-check a recursive function generic in depth,
where each level: encodes (adds K channels), recurses, then decodes (removes
K channels using skip connection). The signature is shape-preserving for all
depths: Tensor[B, C] -> Tensor[B, C].

Uses addition instead of multiplication for channel scaling, so level i has
C + i*K channels (avoids needing exponentials in the expression language).
Keeps things 1D (just [B, C]) to focus on the recursion, not spatial dims.
"""

from typing import assert_type, overload, TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from torch import Tensor
    from torch_shapes import Dim


# ============================================================================
# Minimal encoder/decoder modules — fully generic
# ============================================================================
#
# Each encoder adds K channels, each decoder removes K channels.
# They are generic in C so the same type signature works at every level.


class Encode[C, K](nn.Module):
    """Adds K channels: Tensor[B, C] -> Tensor[B, C + K]."""

    def __init__(self, c: Dim[C], k: Dim[K]) -> None:
        super().__init__()
        self.linear = nn.Linear(c, c + k)

    def forward[B](self, x: Tensor[B, C]) -> Tensor[B, C + K]:
        return self.linear(x)


class Decode[C, K](nn.Module):
    """Removes K channels using skip connection.

    Takes deep features (C + K channels) and skip features (C channels),
    concatenates them, and projects back to C.
    """

    def __init__(self, c: Dim[C], k: Dim[K]) -> None:
        super().__init__()
        self.linear = nn.Linear(2 * c + k, c)

    def forward[B](self, deep: Tensor[B, C + K], skip: Tensor[B, C]) -> Tensor[B, C]:
        combined = torch.cat([deep, skip], dim=1)
        return self.linear(combined)


class Bottleneck[C](nn.Module):
    """Identity-shaped bottleneck: Tensor[B, C] -> Tensor[B, C]."""

    def __init__(self, c: Dim[C]) -> None:
        super().__init__()
        self.linear = nn.Linear(c, c)

    def forward[B](self, x: Tensor[B, C]) -> Tensor[B, C]:
        return self.linear(x)


# ============================================================================
# Test 1: Manually unrolled 3-level encoder-decoder (baseline)
# ============================================================================


def test_manual_unroll():
    """Manually unrolled 3-level skip-connection pattern. This should work today."""
    enc0 = Encode(8, 4)  # 8 -> 12
    enc1 = Encode(12, 4)  # 12 -> 16
    enc2 = Encode(16, 4)  # 16 -> 20
    bottleneck = Bottleneck(20)
    dec2 = Decode(16, 4)  # (20, 16) -> 16
    dec1 = Decode(12, 4)  # (16, 12) -> 12
    dec0 = Decode(8, 4)  # (12, 8) -> 8

    x: Tensor[2, 8] = torch.randn(2, 8)

    # Encode
    e1 = enc0(x)
    assert_type(e1, Tensor[2, 12])
    e2 = enc1(e1)
    assert_type(e2, Tensor[2, 16])
    e3 = enc2(e2)
    assert_type(e3, Tensor[2, 20])

    # Bottleneck
    b = bottleneck(e3)
    assert_type(b, Tensor[2, 20])

    # Decode with skip connections
    d2 = dec2(b, e2)
    assert_type(d2, Tensor[2, 16])
    d1 = dec1(d2, e1)
    assert_type(d1, Tensor[2, 12])
    d0 = dec0(d1, x)
    assert_type(d0, Tensor[2, 8])


# ============================================================================
# Test 2: Shape-preserving recursive function (symbolic, no module indexing)
# ============================================================================
#
# Key hypothesis: a function with signature Tensor[B, C] -> Tensor[B, C]
# whose body does encode -> recurse -> decode should type-check, because
# the recursive call also returns its input shape.
#
# We avoid module indexing entirely — just test the shape flow with
# standalone generic functions.


def encode_step[B, C](x: Tensor[B, C], enc: Encode[C, 4]) -> Tensor[B, C + 4]:
    """One encode step: adds 4 channels."""
    return enc(x)


def decode_step[B, C](
    deep: Tensor[B, C + 4], skip: Tensor[B, C], dec: Decode[C, 4]
) -> Tensor[B, C]:
    """One decode step: removes 4 channels using skip."""
    return dec(deep, skip)


# A shape-preserving function that does one level of encode-decode
def one_level[B, C](
    x: Tensor[B, C],
    enc: Encode[C, 4],
    dec: Decode[C, 4],
    inner: Bottleneck[C + 4],
) -> Tensor[B, C]:
    """One level: encode, bottleneck, decode. Tests the shape flow."""
    skip = x
    encoded = enc(x)  # Tensor[B, C + 4]
    middle = inner(encoded)  # Tensor[B, C + 4]
    decoded = dec(middle, skip)  # Tensor[B, C]
    return decoded


def test_one_level():
    """Test that one level of encode-bottleneck-decode preserves shape."""
    enc = Encode(8, 4)
    dec = Decode(8, 4)
    bn = Bottleneck(12)

    x: Tensor[2, 8] = torch.randn(2, 8)
    y = one_level(x, enc, dec, bn)
    assert_type(y, Tensor[2, 8])


# ============================================================================
# Test 3: Two-level nesting (the inductive step, manually)
# ============================================================================
#
# If one_level works, can we nest it? The inner "bottleneck" is itself
# a one_level call. This tests whether the shape-preservation composes.


def two_levels[B, C](
    x: Tensor[B, C],
    enc0: Encode[C, 4],
    dec0: Decode[C, 4],
    enc1: Encode[C + 4, 4],
    dec1: Decode[C + 4, 4],
    bn: Bottleneck[C + 4 + 4],
) -> Tensor[B, C]:
    """Two levels: encode, encode, bottleneck, decode, decode."""
    skip0 = x
    e0 = enc0(x)  # Tensor[B, C + 4]
    skip1 = e0
    e1 = enc1(e0)  # Tensor[B, C + 4 + 4]
    mid = bn(e1)  # Tensor[B, C + 4 + 4]
    d1 = dec1(mid, skip1)  # Tensor[B, C + 4]
    d0 = dec0(d1, skip0)  # Tensor[B, C]
    return d0


def test_two_levels():
    """Test two-level nesting with generic channels."""
    enc0 = Encode(8, 4)
    dec0 = Decode(8, 4)
    enc1 = Encode(12, 4)
    dec1 = Decode(12, 4)
    bn = Bottleneck(16)

    x: Tensor[2, 8] = torch.randn(2, 8)
    y = two_levels(x, enc0, dec0, enc1, dec1, bn)
    assert_type(y, Tensor[2, 8])


# ============================================================================
# Test 4: Generic encoder/decoder that work at any channel count
# ============================================================================
#
# For the recursive pattern to work, we need encoders/decoders that are
# generic in their input channel count. These don't use module type params
# for channels — they accept any C at call time.


class GenericEncode(nn.Module):
    """Encoder that adds 4 channels to whatever input it receives.

    NOTE: In practice a Conv2d has fixed in/out channels, so this is
    an idealization. The point is to test the shape flow.
    """

    def forward[B, C](self, x: Tensor[B, C]) -> Tensor[B, C + 4]: ...  # type: ignore[return-type]


class GenericDecode(nn.Module):
    """Decoder that removes 4 channels using skip connection."""

    def forward[B, C](
        self, skip: Tensor[B, C], deep: Tensor[B, C + 4]
    ) -> Tensor[B, C]: ...  # type: ignore[return-type]


class GenericBottleneck(nn.Module):
    """Shape-preserving bottleneck."""

    def forward[B, C](self, x: Tensor[B, C]) -> Tensor[B, C]: ...  # type: ignore[return-type]


def generic_one_level[B, C](
    x: Tensor[B, C],
    enc: GenericEncode,
    dec: GenericDecode,
    bottleneck: GenericBottleneck,
) -> Tensor[B, C]:
    """One level with fully generic modules."""
    skip = x
    encoded = enc(x)  # Tensor[B, C + 4]
    middle = bottleneck(encoded)  # Tensor[B, C + 4]
    decoded = dec(skip, middle)  # Tensor[B, C]
    return decoded


def test_generic_one_level():
    """Test shape-preserving one-level with generic modules."""
    enc = GenericEncode()
    dec = GenericDecode()
    bn = GenericBottleneck()

    x: Tensor[2, 8] = torch.randn(2, 8)
    y = generic_one_level(x, enc, dec, bn)
    assert_type(y, Tensor[2, 8])


# ============================================================================
# Test 5: Recursive function with generic modules (the real test)
# ============================================================================
#
# This is the key test: can a recursive function with signature
# Tensor[B, C] -> Tensor[B, C] type-check when its body does:
#   skip = x
#   encoded = enc(x)          → Tensor[B, C + 4]
#   middle = recurse(encoded) → Tensor[B, C + 4]  (by inductive hypothesis)
#   decoded = dec(middle, skip) → Tensor[B, C]
#   return decoded


class RecursiveUNet(nn.Module):
    """UNet using a recursive forward function with generic modules."""

    def __init__(self) -> None:
        super().__init__()
        self.enc = GenericEncode()
        self.dec = GenericDecode()
        self.bottleneck = GenericBottleneck()

    def recurse[I, B, C](self, x: Tensor[B, C], depth: Dim[I]) -> Tensor[B, C]:
        """Shape-preserving recursive encoder-decoder.

        The key insight: this signature (Tensor[B,C] -> Tensor[B,C]) is
        valid for ALL I. The type checker should verify:
        1. Base case (I=0): bottleneck preserves shape. Check.
        2. Inductive step: enc produces [B, C+4], recurse returns [B, C+4]
            (by its own signature), dec restores [B, C]. Check.
        """
        if depth == 0:
            return self.bottleneck(x)

        skip = x
        encoded = self.enc(x)  # Tensor[B, C + 4]
        middle = self.recurse(encoded, depth - 1)  # Tensor[B, C + 4]
        decoded = self.dec(skip, middle)  # Tensor[B, C]
        return decoded

    def forward[B](self, x: Tensor[B, 8]) -> Tensor[B, 8]:
        return self.recurse(x, 3)


def test_recursive_unet():
    """Test recursive UNet with generic modules."""
    net = RecursiveUNet()
    x: Tensor[2, 8] = torch.randn(2, 8)
    y = net(x)
    assert_type(y, Tensor[2, 8])


# ============================================================================
# Test 6: Realistic 2D UNet with spatial dimensions
# ============================================================================
#
# Real UNet encoders double channels and halve spatial dims (via stride-2 conv
# or maxpool+conv). Decoders halve channels and double spatial (via upsample
# or transposed conv) and concatenate skip connections.
#
# Encoder: Tensor[B, C, H, W] -> Tensor[B, 2*C, H//2, W//2]
# Decoder: (skip: Tensor[B, C, H, W], deep: Tensor[B, 2*C, H//2, W//2])
#          -> Tensor[B, C, H, W]


class Down2d(nn.Module):
    """Encoder block: doubles channels, halves spatial."""

    def forward[B, C, H, W](
        self, x: Tensor[B, C, H, W]
    ) -> Tensor[B, 2 * C, H // 2, W // 2]: ...  # type: ignore[return-type]


class Up2d(nn.Module):
    """Decoder block: halves channels, doubles spatial, uses skip connection.

    skip parameter comes first so C can be inferred from a direct position.
    """

    def forward[B, C, H, W](
        self, skip: Tensor[B, C, H, W], deep: Tensor[B, 2 * C, H // 2, W // 2]
    ) -> Tensor[B, C, H, W]: ...  # type: ignore[return-type]


class Bottleneck2d(nn.Module):
    """Shape-preserving bottleneck for 2D feature maps."""

    def forward[B, C, H, W](self, x: Tensor[B, C, H, W]) -> Tensor[B, C, H, W]: ...  # type: ignore[return-type]


class RecursiveUNet2d(nn.Module):
    """UNet with spatial dims: each level doubles C, halves H and W."""

    def __init__(self) -> None:
        super().__init__()
        self.down = Down2d()
        self.up = Up2d()
        self.bottleneck = Bottleneck2d()

    def recurse[I, B, C, H, W](
        self, x: Tensor[B, C, H, W], depth: Dim[I]
    ) -> Tensor[B, C, H, W]:
        """Shape-preserving recursive encoder-decoder with spatial dims.

        Each level: down (C->2C, H->H/2, W->W/2), recurse, up (2C->C, H/2->H, W/2->W).
        The recursive call at depth-1 preserves shape [B, 2C, H/2, W/2].
        """
        if depth == 0:
            return self.bottleneck(x)

        skip = x
        encoded = self.down(x)  # Tensor[B, 2*C, H//2, W//2]
        middle = self.recurse(encoded, depth - 1)  # Tensor[B, 2*C, H//2, W//2]
        decoded = self.up(skip, middle)  # Tensor[B, C, H, W]
        return decoded

    def forward[B, H, W](self, x: Tensor[B, 3, H, W]) -> Tensor[B, 3, H, W]:
        """UNet: input and output have same spatial dims and channels."""
        return self.recurse(x, 4)


def test_recursive_unet_2d():
    """Test 2D UNet: [B, 3, 256, 256] -> [B, 3, 256, 256]."""
    net = RecursiveUNet2d()
    x: Tensor[1, 3, 256, 256] = torch.randn(1, 3, 256, 256)
    y = net(x)
    assert_type(y, Tensor[1, 3, 256, 256])


def test_recursive_unet_2d_generic():
    """Test 2D UNet with generic spatial dims."""
    net = RecursiveUNet2d()
    x: Tensor[2, 3, 128, 128] = torch.randn(2, 3, 128, 128)
    y = net(x)
    assert_type(y, Tensor[2, 3, 128, 128])


# ============================================================================
# Test 7: UNet with separate input/output channels (segmentation)
# ============================================================================
#
# Real segmentation UNets have different input (e.g., 3 RGB) and output
# (e.g., num_classes) channels. The recursive core preserves shape;
# input/output projections are outside the recursion.


class SegmentationUNet(nn.Module):
    """Segmentation UNet: RGB input -> class probability maps."""

    def __init__(self) -> None:
        super().__init__()
        self.input_proj = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.down = Down2d()
        self.up = Up2d()
        self.bottleneck = Bottleneck2d()
        self.output_proj = nn.Conv2d(64, 21, kernel_size=1)

    def recurse[I, B, C, H, W](
        self, x: Tensor[B, C, H, W], depth: Dim[I]
    ) -> Tensor[B, C, H, W]:
        if depth == 0:
            return self.bottleneck(x)
        skip = x
        encoded = self.down(x)
        middle = self.recurse(encoded, depth - 1)
        decoded = self.up(skip, middle)
        return decoded

    def forward[B, H, W](self, x: Tensor[B, 3, H, W]) -> Tensor[B, 21, H, W]:
        projected = self.input_proj(x)  # Tensor[B, 64, H, W]
        features = self.recurse(projected, 4)  # Tensor[B, 64, H, W]
        return self.output_proj(features)  # Tensor[B, 21, H, W]


def test_segmentation_unet():
    """Test segmentation UNet: [B, 3, H, W] -> [B, 21, H, W]."""
    net = SegmentationUNet()
    x: Tensor[1, 3, 256, 256] = torch.randn(1, 3, 256, 256)
    y = net(x)
    assert_type(y, Tensor[1, 21, 256, 256])


# ============================================================================
# Test 8: Additive channel UNet (like Demucs)
# ============================================================================
#
# Demucs adds a fixed number of channels each level rather than doubling.
# This tests the additive pattern with spatial dims.


class DownAdd2d(nn.Module):
    """Encoder: adds K channels, halves spatial."""

    def forward[B, C, H, W](
        self, x: Tensor[B, C, H, W]
    ) -> Tensor[B, C + 64, H // 2, W // 2]: ...  # type: ignore[return-type]


class UpSub2d(nn.Module):
    """Decoder: removes K channels, doubles spatial, uses skip."""

    def forward[B, C, H, W](
        self, skip: Tensor[B, C, H, W], deep: Tensor[B, C + 64, H // 2, W // 2]
    ) -> Tensor[B, C, H, W]: ...  # type: ignore[return-type]


class AdditiveUNet(nn.Module):
    """UNet where each level adds 64 channels instead of doubling."""

    def __init__(self) -> None:
        super().__init__()
        self.down = DownAdd2d()
        self.up = UpSub2d()
        self.bottleneck = Bottleneck2d()

    def recurse[I, B, C, H, W](
        self, x: Tensor[B, C, H, W], depth: Dim[I]
    ) -> Tensor[B, C, H, W]:
        if depth == 0:
            return self.bottleneck(x)
        skip = x
        encoded = self.down(x)  # Tensor[B, C + 64, H//2, W//2]
        middle = self.recurse(encoded, depth - 1)  # Tensor[B, C + 64, H//2, W//2]
        decoded = self.up(skip, middle)  # Tensor[B, C, H, W]
        return decoded

    def forward[B, H, W](self, x: Tensor[B, 3, H, W]) -> Tensor[B, 3, H, W]:
        return self.recurse(x, 3)


def test_additive_unet():
    """Test additive channel UNet."""
    net = AdditiveUNet()
    x: Tensor[1, 3, 128, 128] = torch.randn(1, 3, 128, 128)
    y = net(x)
    assert_type(y, Tensor[1, 3, 128, 128])


# ============================================================================
# Test 9: DenseNet-style linear channel accumulation (Pattern B)
# ============================================================================
#
# Each DenseLayer concatenates GR channels to its input: C -> C + GR.
# After I layers, channels = C + I * GR.
# The recursive function's return type depends on depth I.
#
# This tests whether Pyrefly can verify the algebraic identity:
#   (C + GR) + (I - 1) * GR = C + I * GR


class GenericDenseLayer(nn.Module):
    """DenseNet layer: adds GR channels via concatenation.

    Generic in input channels — works at any level.
    """

    def forward[B, C, H, W](self, x: Tensor[B, C, H, W]) -> Tensor[B, C + 32, H, W]: ...  # type: ignore[return-type]


@overload
def dense_chain[B, C, H, W](
    x: Tensor[B, C, H, W],
    layer: GenericDenseLayer,
    depth: Dim[1],
) -> Tensor[B, C + 32, H, W]: ...


@overload
def dense_chain[I, B, C, H, W](
    x: Tensor[B, C, H, W],
    layer: GenericDenseLayer,
    depth: Dim[I],
) -> Tensor[B, C + I * 32, H, W]: ...


def dense_chain[I, B, C, H, W](
    x: Tensor[B, C, H, W],
    layer: GenericDenseLayer,
    depth: Dim[I],
) -> Tensor[B, C + 32, H, W] | Tensor[B, C + I * 32, H, W]:
    """Chain I DenseNet layers, accumulating 32 channels each.

    Uses overloads to handle the base case (depth=1) without dependent
    narrowing. The first overload gives the concrete return type when
    depth is known to be 1. The second overload handles the general case.

    Inductive step verification:
    - layer(x) : Tensor[B, C + 32, H, W]
    - dense_chain(layer(x), depth-1) : Tensor[B, (C+32) + (I-1)*32, H, W]
    - Pyrefly simplifies: (C + 32) + (I - 1) * 32 = C + I * 32. Check.
    """
    if depth == 1:
        return layer(x)
    y = layer(x)  # Tensor[B, C + 32, H, W]
    return dense_chain(y, layer, depth - 1)  # Tensor[B, (C+32) + (I-1)*32, H, W]
    # = Tensor[B, C + I*32, H, W]


def test_dense_chain():
    """Test DenseNet-style linear channel accumulation."""
    layer = GenericDenseLayer()
    x: Tensor[2, 64, 32, 32] = torch.randn(2, 64, 32, 32)
    y = dense_chain(x, layer, 6)
    assert_type(y, Tensor[2, 256, 32, 32])  # 64 + 6*32 = 256


def test_dense_chain_one():
    """Test base case: depth=1 applies one layer."""
    layer = GenericDenseLayer()
    x: Tensor[2, 64, 32, 32] = torch.randn(2, 64, 32, 32)
    y = dense_chain(x, layer, 1)
    assert_type(y, Tensor[2, 96, 32, 32])  # 64 + 32 = 96


# ============================================================================
# Test 9b: Pattern B with symbolic GR (Issue #6 — symbolic product distribution)
# ============================================================================
#
# Same as Test 9, but GR is a type variable instead of literal 32.
# Requires canonicalize_product to distribute GR * (I - 1) into GR*I - GR
# so that (C + GR) + GR*(I-1) simplifies to C + GR*I.


class SymbolicDenseLayer[GR](nn.Module):
    """DenseNet layer with symbolic growth rate."""

    def forward[B, C, H, W](self, x: Tensor[B, C, H, W]) -> Tensor[B, C + GR, H, W]: ...  # type: ignore[return-type]


@overload
def symbolic_dense_chain[GR, B, C, H, W](
    x: Tensor[B, C, H, W],
    layer: SymbolicDenseLayer[GR],
    depth: Dim[1],
) -> Tensor[B, C + GR, H, W]: ...


@overload
def symbolic_dense_chain[I, GR, B, C, H, W](
    x: Tensor[B, C, H, W],
    layer: SymbolicDenseLayer[GR],
    depth: Dim[I],
) -> Tensor[B, C + I * GR, H, W]: ...


def symbolic_dense_chain[I, GR, B, C, H, W](
    x: Tensor[B, C, H, W],
    layer: SymbolicDenseLayer[GR],
    depth: Dim[I],
) -> Tensor[B, C + GR, H, W] | Tensor[B, C + I * GR, H, W]:
    """Chain I DenseNet layers with symbolic growth rate GR.

    Inductive step: (C + GR) + (I - 1) * GR = C + I * GR
    Requires symbolic product distribution: GR * (I - 1) → GR*I - GR.
    """
    if depth == 1:
        return layer(x)
    y = layer(x)
    return symbolic_dense_chain(y, layer, depth - 1)


def test_symbolic_dense_chain():
    """Test Pattern B with symbolic GR — previously blocked by Issue #6."""
    layer = SymbolicDenseLayer[16]()
    x: Tensor[2, 64, 32, 32] = torch.randn(2, 64, 32, 32)
    y = symbolic_dense_chain(x, layer, 6)
    assert_type(y, Tensor[2, 160, 32, 32])  # 64 + 6*16 = 160


def test_symbolic_dense_chain_one():
    """Test symbolic base case: depth=1."""
    layer = SymbolicDenseLayer[16]()
    x: Tensor[2, 64, 32, 32] = torch.randn(2, 64, 32, 32)
    y = symbolic_dense_chain(x, layer, 1)
    assert_type(y, Tensor[2, 80, 32, 32])  # 64 + 16 = 80


# ============================================================================
# Test 10: Purely downsampling chain (Pattern C — exponential)
# ============================================================================
#
# ResNet outer loop: each stage doubles channels, halves spatial.
# No decoder — the shape genuinely changes with depth.
# Return type uses exponentials: Tensor[B, C * 2**I, H // 2**I, W // 2**I].
#
# Inductive step relies on:
#   channels: 2*C * 2**(I-1) = C * 2**I  (same-base Pow grouping in products)
#   spatial:  (H//2) // 2**(I-1) = H // 2**I  (nested division flattening)


class GenericDownStage(nn.Module):
    """Downsampling stage: doubles channels, halves spatial."""

    def forward[B, C, H, W](
        self, x: Tensor[B, C, H, W]
    ) -> Tensor[B, 2 * C, H // 2, W // 2]: ...  # type: ignore[return-type]


# Now that we have exponentials (SizeExpr::Pow), we CAN express the output type:
#   channels = C * 2**I
#   height   = H // 2**I
#   width    = W // 2**I


@overload
def downsample_chain[B, C, H, W](
    stage: GenericDownStage,
    x: Tensor[B, C, H, W],
    depth: Dim[1],
) -> Tensor[B, 2 * C, H // 2, W // 2]: ...


@overload
def downsample_chain[I, B, C, H, W](
    stage: GenericDownStage,
    x: Tensor[B, C, H, W],
    depth: Dim[I],
) -> Tensor[B, C * 2**I, H // 2**I, W // 2**I]: ...


def downsample_chain[I, B, C, H, W](
    stage: GenericDownStage,
    x: Tensor[B, C, H, W],
    depth: Dim[I],
) -> Tensor[B, 2 * C, H // 2, W // 2] | Tensor[B, C * 2**I, H // 2**I, W // 2**I]:
    """Chain I downsampling stages (Pattern C — exponential).

    Inductive step: After one stage we have Tensor[B, 2*C, H//2, W//2].
    Recursing with depth-1:
        channels: 2*C * 2**(I-1) = C * 2**1 * 2**(I-1) = C * 2**I  ✓
        spatial:  (H//2) // 2**(I-1) = H // (2 * 2**(I-1)) = H // 2**I  ✓
    """
    if depth == 1:
        return stage(x)
    y = stage(x)
    return downsample_chain(stage, y, depth - 1)


def test_downsample_concrete():
    """Test that two concrete downsample stages compose correctly."""
    stage = GenericDownStage()
    x: Tensor[1, 16, 64, 64] = torch.randn(1, 16, 64, 64)

    # Stage 1: [1, 16, 64, 64] -> [1, 32, 32, 32]
    y1 = stage(x)
    assert_type(y1, Tensor[1, 32, 32, 32])

    # Stage 2: [1, 32, 32, 32] -> [1, 64, 16, 16]
    y2 = stage(y1)
    assert_type(y2, Tensor[1, 64, 16, 16])


def test_downsample_chain_recursive():
    """Test Pattern C recursive chain with concrete values."""
    stage = GenericDownStage()
    x: Tensor[1, 16, 64, 64] = torch.randn(1, 16, 64, 64)

    # depth=1: [1, 16, 64, 64] -> [1, 32, 32, 32]
    y1 = downsample_chain(stage, x, 1)
    assert_type(y1, Tensor[1, 32, 32, 32])

    # depth=2: [1, 16, 64, 64] -> [1, 64, 16, 16]
    y2 = downsample_chain(stage, x, 2)
    assert_type(y2, Tensor[1, 64, 16, 16])

    # depth=3: [1, 16, 64, 64] -> [1, 128, 8, 8]
    y3 = downsample_chain(stage, x, 3)
    assert_type(y3, Tensor[1, 128, 8, 8])


# ============================================================================
# Test 11: DenseNet block+transition pattern
# ============================================================================
#
# A DenseBlock adds 6*GR channels, then a Transition halves channels and
# halves spatial. Combined: C -> (C + 6*GR) // 2, H -> H//2, W -> W//2.
#
# This is a non-linear recurrence in channels — even harder than exponential.
# Test whether a shape-preserving wrapper can help.


class GenericDenseBlock(nn.Module):
    """DenseBlock: adds 6*32=192 channels."""

    def forward[B, C, H, W](
        self, x: Tensor[B, C, H, W]
    ) -> Tensor[B, C + 192, H, W]: ...  # type: ignore[return-type]


class GenericTransition(nn.Module):
    """Transition: halves channels and spatial dims.

    Note: C // 2 requires C to be even, which we assume.
    """

    def forward[B, C, H, W](
        self, x: Tensor[B, C, H, W]
    ) -> Tensor[B, C // 2, H // 2, W // 2]: ...  # type: ignore[return-type]


class GenericDenseStage(nn.Module):
    """Combined DenseBlock + Transition.

    C -> C + 192 -> (C + 192) // 2, and H -> H//2, W -> W//2.
    """

    def __init__(self) -> None:
        super().__init__()
        self.block = GenericDenseBlock()
        self.transition = GenericTransition()

    def forward[B, C, H, W](
        self, x: Tensor[B, C, H, W]
    ) -> Tensor[B, (C + 192) // 2, H // 2, W // 2]:
        y = self.block(x)  # Tensor[B, C + 192, H, W]
        return self.transition(y)  # Tensor[B, (C + 192) // 2, H // 2, W // 2]


def test_dense_stage_concrete():
    """Test that DenseStage works with concrete values."""
    stage = GenericDenseStage()

    # Stage 1: 32 channels -> (32 + 192) // 2 = 112
    x: Tensor[1, 32, 64, 64] = torch.randn(1, 32, 64, 64)
    y1 = stage(x)
    assert_type(y1, Tensor[1, 112, 32, 32])

    # Stage 2: 112 channels -> (112 + 192) // 2 = 152
    y2 = stage(y1)
    assert_type(y2, Tensor[1, 152, 16, 16])


# The DenseStage recurrence C -> (C + 192) // 2 has no closed form.
# After I stages: C_0 = 32, C_1 = 112, C_2 = 152, C_3 = 172, ...
# Converges toward 192. Cannot express C_I as a formula of I without
# exponentials AND division. This pattern requires either:
# (a) Concrete unrolling (list the values in a tuple), or
# (b) Shape-preserving encapsulation (not possible — shape genuinely changes)
#
# Conclusion: DenseNet block+transition is NOT expressible with any of our
# recursive patterns. It requires comprehension unrolling with concrete values,
# or manual unrolling.


# ============================================================================
# Test 12: ResNet with shape-preserving "stage" abstraction
# ============================================================================
#
# Can we restructure ResNet's outer loop as shape-preserving?
# Idea: each "stage" (group + downsample) takes [B, C, H, W] and outputs
# [B, 2C, H/2, W/2]. If we pair it with an "up" that restores shape, it's
# Pattern A. But ResNet has no "up" — it's a classifier, not an autoencoder.
#
# However: what about the classification head? After all stages we apply
# AdaptiveAvgPool + Linear. The total model is:
#   input [B, 3, H, W] -> features [B, C_final, 1, 1] -> logits [B, classes]
#
# The downsampling chain IS the model's core. We can't make it shape-preserving
# because that would lose information (the whole point is to downsample).
#
# Conclusion: ResNet outer loop needs either exponentials (Pattern C) or
# concrete unrolling. With only 2-3 stages, concrete unrolling is acceptable.


# ============================================================================
# Test 13: Demucs-style additive skip (x = x + skip)
# ============================================================================
#
# Demucs uses additive skip connections instead of concatenation:
#   x = x + skip[..., :length]
# The skip has the same channels as the decoder output (no cat → project).
# This means the decoder doesn't need the skip in its forward signature —
# the addition happens outside. Test this pattern.


class DemucsEncoder(nn.Module):
    """Demucs encoder: doubles channels, halves length."""

    def forward[B, C, T](self, x: Tensor[B, C, T]) -> Tensor[B, 2 * C, T // 2]: ...  # type: ignore[return-type]


class DemucsDecoder(nn.Module):
    """Demucs decoder: takes encoded features and skip, restores shape.

    Uses skip to infer target dimensions (C and T appear in direct positions).
    In practice, Demucs trims the upsampled output to match skip length.
    """

    def forward[B, C, T](
        self, skip: Tensor[B, C, T], deep: Tensor[B, 2 * C, T // 2]
    ) -> Tensor[B, C, T]: ...  # type: ignore[return-type]


class DemucsBottleneck(nn.Module):
    """Shape-preserving bottleneck for 1D audio."""

    def forward[B, C, T](self, x: Tensor[B, C, T]) -> Tensor[B, C, T]: ...  # type: ignore[return-type]


class RecursiveDemucs(nn.Module):
    """Demucs-style model with additive skip connections."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = DemucsEncoder()
        self.decoder = DemucsDecoder()
        self.bottleneck = DemucsBottleneck()

    def recurse[I, B, C, T](self, x: Tensor[B, C, T], depth: Dim[I]) -> Tensor[B, C, T]:
        """Shape-preserving recursive encoder-decoder.

        Each level:
        1. Save skip = x                   : Tensor[B, C, T]
        2. Encode: x -> [B, 2C, T//2]
        3. Recurse: preserves [B, 2C, T//2]
        4. Decode: [B, 2C, T//2] -> [B, C, T]
        5. Add skip: result + skip         : Tensor[B, C, T]
        """
        if depth == 0:
            return self.bottleneck(x)

        skip = x
        encoded = self.encoder(x)  # Tensor[B, 2*C, T//2]
        middle = self.recurse(encoded, depth - 1)  # Tensor[B, 2*C, T//2]
        decoded = self.decoder(skip, middle)  # Tensor[B, C, T]
        result = decoded + skip  # Tensor[B, C, T]
        return result

    def forward[B, T](self, x: Tensor[B, 2, T]) -> Tensor[B, 2, T]:
        return self.recurse(x, 4)


def test_recursive_demucs():
    """Test Demucs-style model with additive skips."""
    net = RecursiveDemucs()
    x: Tensor[1, 2, 44100] = torch.randn(1, 2, 44100)
    y = net(x)
    assert_type(y, Tensor[1, 2, 44100])
