# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Demucs music source separation model from TorchBenchmark with shape annotations.

Original: pytorch/benchmark/torchbenchmark/models/demucs/demucs/model.py

Port notes:
- Demucs is a Wave-U-Net style encoder-decoder for audio source separation.
    The encoder uses Conv1d to downsample, the decoder uses ConvTranspose1d
    to upsample, with skip connections between corresponding layers.
- This port covers the encoder/decoder building blocks as generic modules
    parameterized over channels (non-GLU, rewrite=True config).
- Full Demucs forward ported: encoder → BLSTM → decoder with skip connections.
    ModuleList loop inlined into explicit enc0/enc1/enc2 + dec2/dec1/dec0 calls.
    Skip connections use center_trim (algebraic simplification resolves slice dims).
    BLSTM bottleneck: nn.LSTM with num_directions=2 (bidirectional workaround).
    GLU encoder variant also typed (EncoderBlockGLU with nn.GLU(dim=1)).
- Config used: sources=4, audio_channels=2, channels=64, depth=3,
    kernel_size=8, stride=4, growth=2, rewrite=True, glu=False, context=3

Key patterns exercised:
- Conv1d spatial formula: (L + 2*P - D*(K-1) - 1) // S + 1
- ConvTranspose1d spatial formula: (L - 1) * S - 2*P + D*(K-1) + OP + 1
- nn.Sequential chain forwarding with Conv1d modules (first 1D conv port)
- Tensor.view for reshaping final output to [B, Sources, AudioCh, T]

Now ported (previously omitted):
- downsample: x[:, :, ::stride] stride slicing works with Dim[S] step
- upsample: .size() tuple unpacking, view, ellipsis slicing, broadcasting
- center_trim: symbolic slice bounds from .size() diffs, returns Tensor[B, C, R]
"""

from typing import assert_type, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from torch import Tensor
    from torch_shapes import Dim


# ============================================================================
# Downsample (stride-based decimation)
# ============================================================================
# Original: def downsample(x, stride: int): return x[:, :, ::stride]
# Pattern: multi-dim tuple indexing with stride step.
# Stride slicing computes ceil_div(dim, step) for the strided dimension.


def downsample[B, C, T, S](
    x: Tensor[B, C, T], stride: Dim[S]
) -> Tensor[B, C, (T + S - 1) // S]:
    """Downsample x by decimation.

    Stride slicing x[:, :, ::stride] computes ceil_div(T, S) = (T + S - 1) // S
    for the strided dimension. Works with both literal and symbolic strides.
    """
    return x[:, :, ::stride]


# ============================================================================
# Upsample (linear interpolation)
# ============================================================================
# Original: def upsample(x, stride: int): ...
# Pattern: .size() tuple unpacking, torch.arange, view, ellipsis slicing,
#   broadcasting (x * weight), reshape with -1.
#
# Changes from original:
# - Type annotations added on function signature
# - stride: int (not Dim) because arange and broadcasting need runtime int
# - Variable reassignment x → x_4d (shape changes from 3D to 4D)
# - out.reshape(batch, channels, -1) uses -1 for automatic size inference


def center_trim[B, C, T, R](
    tensor: Tensor[B, C, T], reference: Tensor[B, C, R]
) -> Tensor[B, C, R]:
    """Center trim `tensor` to match `reference` along the last dimension.

    Original: demucs/utils.py center_trim
    Trims (T - R) / 2 from each side, resulting in last dim = R.

    Uses `tensor.size(-1) - (delta - delta // 2)` as stop instead of
    negation, so it works for delta == 0 (no trim) without a conditional.
    The type system resolves the slice result to R via:
    - start = delta // 2 = (T - R) // 2
    - stop = T - (delta - delta // 2) = T - ((T - R) - (T - R) // 2)
    - sub_dim(stop, start) simplifies to R (the (T-R)//2 terms cancel)
    """
    reference_val = reference.size(-1)
    delta = tensor.size(-1) - reference_val
    if delta < 0:
        raise ValueError(f"tensor must be larger than reference. Delta is {delta}.")
    return tensor[..., delta // 2 : tensor.size(-1) - (delta - delta // 2)]


def upsample[B, C, T](x: Tensor[B, C, T], stride: int) -> Tensor:
    """Linear upsampling, the output will be `stride` times longer.

    Steps:
    1. x.size() → (batch, channels, time) as Dim types
    2. torch.arange(stride) / stride → weight vector
    3. view to [B, C, T, 1] for broadcasting
    4. x[..., :-1, :] * (1 - weight) + x[..., 1:, :] * weight
    5. reshape to [B, C, (T-1)*stride]
    """
    batch, channels, time = x.size()
    weight: Tensor = torch.arange(stride, dtype=torch.float32) / stride
    x_4d = x.view(batch, channels, time, 1)
    out = x_4d[..., :-1, :] * (1 - weight) + x_4d[..., 1:, :] * weight
    return out.reshape(batch, channels, -1)


# ============================================================================
# BLSTM Bottleneck
# ============================================================================
# The original Demucs uses a bidirectional LSTM between encoder and decoder.
# BLSTM(dim) → permute → LSTM(dim, dim, bidirectional=True) → linear → permute
# Input: [B, C, T] → permute to [B, T, C] → LSTM → [B, T, 2*C] → Linear → [B, T, C]
# → permute to [B, C, T]


class BLSTM[Ch](nn.Module):
    """Bidirectional LSTM bottleneck.

    Input:  Tensor[B, Ch, T]
    Output: Tensor[B, Ch, T]

    Internally: permute → BiLSTM → Linear(2*Ch, Ch) → permute back.
    Uses num_directions=2 (temporary workaround for bidirectional=True
    not flowing through inject_module_attrs as literal bool).
    """

    def __init__(self, dim: Dim[Ch]) -> None:
        super().__init__()
        self.lstm = nn.LSTM(dim, dim, num_directions=2, batch_first=True)
        self.linear = nn.Linear(2 * dim, dim)

    def forward[B, T](self, x: Tensor[B, Ch, T]) -> Tensor[B, Ch, T]:
        # [B, Ch, T] → [B, T, Ch]
        x_perm = x.permute(0, 2, 1)
        assert_type(x_perm, Tensor[B, T, Ch])
        # BiLSTM: [B, T, Ch] → [B, T, 2*Ch]
        lstm_out, _h_n, _c_n = self.lstm(x_perm)
        assert_type(lstm_out, Tensor[B, T, 2 * Ch])
        # Linear: [B, T, 2*Ch] → [B, T, Ch]
        lin_out = self.linear(lstm_out)
        assert_type(lin_out, Tensor[B, T, Ch])
        # [B, T, Ch] → [B, Ch, T]
        out = lin_out.permute(0, 2, 1)
        assert_type(out, Tensor[B, Ch, T])
        return out


# ============================================================================
# GLU Encoder Block Variant
# ============================================================================
# With glu=True, the original uses ch_scale=2 and GLU activation:
#   Conv1d(in_ch, ch, K, stride=S) → ReLU → Conv1d(ch, 2*ch, 1) → GLU(dim=1)
# GLU halves the channel dim: 2*ch → ch


class EncoderBlockGLU(nn.Module):
    """Encoder block with GLU activation (glu=True, ch_scale=2).

    Conv1d(2, 64, 8, stride=4) → ReLU → Conv1d(64, 128, 1) → GLU(dim=1)
    GLU halves dim 1: 128 → 64
    """

    def __init__(self) -> None:
        super().__init__()
        self.encode = nn.Sequential(
            nn.Conv1d(2, 64, 8, stride=4),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.GLU(dim=1),
        )

    def forward[B, L](self, x: Tensor[B, 2, L]) -> Tensor[B, 64, (L - 8) // 4 + 1]:
        out = self.encode(x)
        assert_type(out, Tensor[B, 64, (L - 8) // 4 + 1])
        return out


# ============================================================================
# Encoder Block (generic over channels)
# ============================================================================
# In the original, each encoder layer is:
#   Conv1d(in_ch, ch, kernel_size=8, stride=4) → ReLU
#   Conv1d(ch, ch_scale*ch, 1) → activation  (rewrite=True)
# With glu=False: ch_scale=1, activation=ReLU
#
# Channel progression (depth=3, channels=64, growth=2):
#   Layer 0: 2 → 64      (audio_channels → channels)
#   Layer 1: 64 → 128     (channels → channels*growth)
#   Layer 2: 128 → 256    (channels*growth → channels*growth^2)


class EncoderBlock[InC, OutC](nn.Module):
    """Encoder block: Conv1d(InC, OutC, 8, stride=4) → ReLU → Conv1d(OutC, OutC, 1) → ReLU.

    Spatial: L → (L - 8) // 4 + 1 (1x1 conv preserves spatial).
    """

    def __init__(self, in_ch: Dim[InC], out_ch: Dim[OutC]) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 8, stride=4)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 1)

    def forward[B, L](self, x: Tensor[B, InC, L]) -> Tensor[B, OutC, (L - 8) // 4 + 1]:
        h = F.relu(self.conv1(x))
        assert_type(h, Tensor[B, OutC, (L - 8) // 4 + 1])
        out = F.relu(self.conv2(h))
        assert_type(out, Tensor[B, OutC, (L - 8) // 4 + 1])
        return out


# ============================================================================
# Decoder Block (generic over channels)
# ============================================================================
# In the original (rewrite=True, glu=False, upsample=False), each decoder layer:
#   Conv1d(ch, ch, context=3) → ReLU  (rewrite conv)
#   ConvTranspose1d(ch, out_ch, kernel_size=8, stride=4)
#   ReLU applied externally (not included here — outermost layer omits it)
#
# Conv1d(ch, ch, 3) spatial: L → L - 2
# ConvTranspose1d(ch, out, 8, 4) spatial: L-2 → (L-3)*4 + 8
# The type system canonicalizes (L-3)*4 + 8 to 4*L - 4 via distributive law.


class DecoderBlock[InC, OutC](nn.Module):
    """Decoder block: Conv1d(InC, InC, 3) → ReLU → ConvTranspose1d(InC, OutC, 8, stride=4).

    No final ReLU — caller applies it for non-outermost layers.
    Spatial: L → (L-3)*4 + 8.
    """

    def __init__(self, in_ch: Dim[InC], out_ch: Dim[OutC]) -> None:
        super().__init__()
        self.context_conv = nn.Conv1d(in_ch, in_ch, 3)
        self.deconv = nn.ConvTranspose1d(in_ch, out_ch, 8, stride=4)

    def forward[B, L](self, x: Tensor[B, InC, L]) -> Tensor[B, OutC, (L - 3) * 4 + 8]:
        h = F.relu(self.context_conv(x))
        assert_type(h, Tensor[B, InC, L - 2])
        out = self.deconv(h)
        assert_type(out, Tensor[B, OutC, (L - 3) * 4 + 8])
        return out


# ============================================================================
# Encoder-only pipeline (no skip connections)
# ============================================================================
# The full Demucs forward uses skip connections: each encoder output is saved
# and added to the corresponding decoder input after center_trim alignment.
# Since center_trim requires dynamic slicing, we only type the encoder pipeline
# and individual decoder blocks separately.


class DemucsEncoder(nn.Module):
    """Three-layer encoder pipeline (no skip connections).

    Spatial progression for input length L:
        After enc0: L0 = (L - 8) // 4 + 1 = L // 4 - 1  (S2 simplification)
        After enc1: L1 = (L0 - 8) // 4 + 1 = (L // 4 - 1) // 4 - 1
        After enc2: L2 = (L1 - 8) // 4 + 1 = ((L // 4 - 1) // 4 - 1) // 4 - 1

    For L=1024: L0=255, L1=61+1=62? No: L0=1024//4-1=255,
        L1=(255-8)//4+1=247//4+1=61+1=62, L2=(62-8)//4+1=54//4+1=13+1=14
    """

    def __init__(self) -> None:
        super().__init__()
        self.enc0 = EncoderBlock(2, 64)
        self.enc1 = EncoderBlock(64, 128)
        self.enc2 = EncoderBlock(128, 256)

    def forward[B, L](
        self, x: Tensor[B, 2, L]
    ) -> Tensor[B, 256, (((L // 4 - 1) // 4 - 1) // 4 - 1)]:
        h0 = self.enc0(x)
        assert_type(h0, Tensor[B, 64, (L // 4 - 1)])
        h1 = self.enc1(h0)
        h2 = self.enc2(h1)
        return h2


# ============================================================================
# Output reshaping
# ============================================================================
# The original's final step reshapes [B, sources*audio_channels, T]
# into [B, sources, audio_channels, T] using view.


def reshape_output[B, T](
    x: Tensor[B, 8, T],
) -> Tensor[B, 4, 2, T]:
    """Reshape decoder output to [B, sources=4, audio_channels=2, T]."""
    out = x.view(x.size(0), 4, 2, x.size(-1))
    assert_type(out, Tensor[B, 4, 2, T])
    return out


# ============================================================================
# Full Demucs: Encoder → BLSTM → Decoder with skip connections
# ============================================================================
# Original: Demucs.forward loops over self.encoder ModuleList (saving skips),
# runs BLSTM, then loops over self.decoder ModuleList (popping skips).
# Here we inline the loop for depth=3 with explicit skip variables.
#
# Changes from original:
# - ModuleList loop unrolled into explicit enc0/enc1/enc2, dec2/dec1/dec0 calls
# - saved list → explicit skip variables (s0, x0, x1, x2)
# - Variable reassignment x → unique names at each step
# - upsample mode not included (non-upsample is the default)


class Demucs(nn.Module):
    """Full Demucs model: 3-layer encoder → BLSTM → 3-layer decoder with skips.

    Input:  Tensor[B, 2, L]  (stereo audio)
    Output: Tensor[B, 4, 2, T]  (4 sources × stereo × time)

    Skip connections: each encoder output is center_trimmed to match the
    corresponding decoder output, then added before decoding.
    """

    def __init__(self) -> None:
        super().__init__()
        self.enc0 = EncoderBlock(2, 64)
        self.enc1 = EncoderBlock(64, 128)
        self.enc2 = EncoderBlock(128, 256)
        self.lstm = BLSTM(256)
        self.dec2 = DecoderBlock(256, 128)
        self.dec1 = DecoderBlock(128, 64)
        self.dec0 = DecoderBlock(64, 8)

    def forward[B, L](self, mix: Tensor[B, 2, L]):
        # Encoder: save outputs for skip connections
        # Spatial: L → L//4-1 → (L//4-1)//4-1 → ((L//4-1)//4-1)//4-1
        x0 = self.enc0(mix)
        assert_type(x0, Tensor[B, 64, L // 4 - 1])
        x1 = self.enc1(x0)
        assert_type(x1, Tensor[B, 128, (L // 4 - 1) // 4 - 1])
        x2 = self.enc2(x1)
        assert_type(x2, Tensor[B, 256, ((L // 4 - 1) // 4 - 1) // 4 - 1])

        # BLSTM bottleneck (preserves shape)
        xb = self.lstm(x2)
        assert_type(xb, Tensor[B, 256, ((L // 4 - 1) // 4 - 1) // 4 - 1])

        # Decoder with skip connections
        # Each decoder: Conv1d(ch, ch, 3) → ConvTranspose1d(ch, out, 8, stride=4)
        # Spatial: L' → (L'-3)*4+8 = 4*L'-4
        # Canonical form distributes the outermost -1 from L2 into the 4* coefficient.
        # ReLU applied after non-outermost decoders (original has ReLU inside dec2/dec1).
        skip2 = center_trim(x2, xb)
        d2 = F.relu(self.dec2(xb + skip2))
        assert_type(d2, Tensor[B, 128, 4 * (((L // 4 - 1) // 4 - 1) // 4) - 8])

        skip1 = center_trim(x1, d2)
        d1 = F.relu(self.dec1(d2 + skip1))
        assert_type(d1, Tensor[B, 64, 16 * (((L // 4 - 1) // 4 - 1) // 4) - 36])

        skip0 = center_trim(x0, d1)
        d0 = self.dec0(d1 + skip0)
        assert_type(d0, Tensor[B, 8, 64 * (((L // 4 - 1) // 4 - 1) // 4) - 148])

        # Reshape: [B, sources*audio_ch, T] → [B, sources, audio_ch, T]
        return d0.view(d0.size(0), 4, 2, d0.size(-1))


# ============================================================================
# Smoke tests
# ============================================================================


def test_encoder_block0():
    """Test first encoder: [B, 2, L] → [B, 64, (L-8)//4+1].

    With L=1024: (1024 - 8) // 4 + 1 = 1016 // 4 + 1 = 254 + 1 = 255
    """
    enc = EncoderBlock(2, 64)
    x: Tensor[2, 2, 1024] = torch.randn(2, 2, 1024)
    out = enc(x)
    assert_type(out, Tensor[2, 64, 255])


def test_encoder_block1():
    """Test second encoder: [B, 64, L] → [B, 128, (L-8)//4+1].

    With L=255: (255 - 8) // 4 + 1 = 247 // 4 + 1 = 61 + 1 = 62
    """
    enc = EncoderBlock(64, 128)
    x: Tensor[2, 64, 255] = torch.randn(2, 64, 255)
    out = enc(x)
    assert_type(out, Tensor[2, 128, 62])


def test_encoder_block2():
    """Test third encoder: [B, 128, L] → [B, 256, (L-8)//4+1].

    With L=62: (62 - 8) // 4 + 1 = 54 // 4 + 1 = 13 + 1 = 14
    """
    enc = EncoderBlock(128, 256)
    x: Tensor[2, 128, 62] = torch.randn(2, 128, 62)
    out = enc(x)
    assert_type(out, Tensor[2, 256, 14])


def test_decoder_block2():
    """Test innermost decoder: [B, 256, L] → [B, 128, (L-3)*4+8].

    With L=14: (14-3)*4 + 8 = 52
    """
    dec = DecoderBlock(256, 128)
    x: Tensor[2, 256, 14] = torch.randn(2, 256, 14)
    out = dec(x)
    assert_type(out, Tensor[2, 128, 52])


def test_decoder_block1():
    """Test middle decoder: [B, 128, L] → [B, 64, (L-3)*4+8].

    With L=52: (52-3)*4 + 8 = 204
    """
    dec = DecoderBlock(128, 64)
    x: Tensor[2, 128, 52] = torch.randn(2, 128, 52)
    out = dec(x)
    assert_type(out, Tensor[2, 64, 204])


def test_decoder_block0():
    """Test outermost decoder: [B, 64, L] → [B, 8, (L-3)*4+8].

    With L=204: (204-3)*4 + 8 = 812
    """
    dec = DecoderBlock(64, 8)
    x: Tensor[2, 64, 204] = torch.randn(2, 64, 204)
    out = dec(x)
    assert_type(out, Tensor[2, 8, 812])


def test_encoder_pipeline():
    """Test three-layer encoder: [B, 2, 1024] → [B, 256, 14]."""
    encoder = DemucsEncoder()
    x: Tensor[2, 2, 1024] = torch.randn(2, 2, 1024)
    out = encoder(x)
    assert_type(out, Tensor[2, 256, 14])


def test_reshape_output():
    """Test output reshape: [B, 8, T] → [B, 4, 2, T]."""
    x: Tensor[2, 8, 812] = torch.randn(2, 8, 812)
    out = reshape_output(x)
    assert_type(out, Tensor[2, 4, 2, 812])


def test_blstm_bottleneck():
    """Test BLSTM: [B, 256, T] → [B, 256, T] (preserves shape)."""
    blstm = BLSTM(256)
    x: Tensor[2, 256, 14] = torch.randn(2, 256, 14)
    out = blstm(x)
    assert_type(out, Tensor[2, 256, 14])


def test_encoder_block_glu():
    """Test GLU encoder: [B, 2, L] → [B, 64, (L-8)//4+1].

    With L=1024: (1024-8)//4+1 = 255
    Conv1d(2→64, K=8, S=4) → ReLU → Conv1d(64→128, K=1) → GLU(dim=1) → 64 ch
    """
    enc = EncoderBlockGLU()
    x: Tensor[2, 2, 1024] = torch.randn(2, 2, 1024)
    out = enc(x)
    assert_type(out, Tensor[2, 64, 255])


def test_downsample():
    """Test stride-based downsampling via function call.

    Original: downsample(x, stride=4)
    Pattern: Tensor[B, C, T] → Tensor[B, C, ceil_div(T, stride)]
    With T=1024, stride=4: ceil_div(1024, 4) = 256
    """
    x: Tensor[2, 64, 1024] = torch.randn(2, 64, 1024)
    # Direct stride slicing with literal step — shape is tracked
    out = x[:, :, ::4]
    assert_type(out, Tensor[2, 64, 256])
    # Function call with literal Dim stride — shape also tracked
    out2 = downsample(x, 4)
    assert_type(out2, Tensor[2, 64, 256])


def test_downsample_odd():
    """Test stride on non-divisible length: ceil_div(255, 4) = 64.

    (255 + 3) // 4 = 64
    """
    x: Tensor[2, 64, 255] = torch.randn(2, 64, 255)
    out = x[:, :, ::4]
    assert_type(out, Tensor[2, 64, 64])
    out2 = downsample(x, 4)
    assert_type(out2, Tensor[2, 64, 64])


def test_upsample():
    """Test linear upsampling.

    Original: upsample(x, stride=4)
    Pattern: .size() tuple unpacking → view → ellipsis slicing → broadcasting → reshape
    """
    x: Tensor[2, 64, 100] = torch.randn(2, 64, 100)
    out = upsample(x, 4)
    # Result is shapeless Tensor because stride is int (not Dim)
    # and broadcasting between different-rank tensors uses Self return type
    assert_type(out, Tensor)


def test_center_trim():
    """Test center trimming: align skip connection spatial dims.

    Original: center_trim(tensor, reference)
    Trims tensor's last dim to match reference's last dim.
    """
    tensor: Tensor[2, 64, 1024] = torch.randn(2, 64, 1024)
    ref: Tensor[2, 64, 1016] = torch.randn(2, 64, 1016)
    out = center_trim(tensor, ref)
    assert_type(out, Tensor[2, 64, 1016])


def test_encoder_decoder_chain():
    """Test encoder block followed by corresponding decoder block.

    EncoderBlock0: [B, 2, 1024] → [B, 64, 255]
    DecoderBlock0: [B, 64, 255] → [B, 8, (255-3)*4+8] = [B, 8, 1016]

    Note: In the real model, skip connections would add the encoder output
    to the decoder input. The spatial dimensions don't match exactly
    (1024 vs 1016) because the conv context shrinks the decoder's spatial dim.
    The original handles this with center_trim.
    """
    enc = EncoderBlock(2, 64)
    dec = DecoderBlock(64, 8)
    x: Tensor[2, 2, 1024] = torch.randn(2, 2, 1024)
    encoded = enc(x)
    assert_type(encoded, Tensor[2, 64, 255])
    decoded = dec(encoded)
    assert_type(decoded, Tensor[2, 8, 1016])


def test_demucs_full():
    """End-to-end Demucs: [B, 2, L] → [B, 4, 2, T] with skip connections.

    Input L=16384:
        enc0: [2, 2, 16384] → [2, 64, 4095]
        enc1: [2, 64, 4095] → [2, 128, 1022]
        enc2: [2, 128, 1022] → [2, 256, 254]
        BLSTM: [2, 256, 254] → [2, 256, 254]
        dec2: [2, 256, 254] → [2, 128, 1012]
        dec1: [2, 128, 1012] → [2, 64, 4044]
        dec0: [2, 64, 4044] → [2, 8, 16172]
        view: [2, 8, 16172] → [2, 4, 2, 16172]
    """
    model = Demucs()
    x: Tensor[2, 2, 16384] = torch.randn(2, 2, 16384)
    out = model(x)
    assert_type(out, Tensor[2, 4, 2, 16172])


def test_demucs_full_step_by_step():
    """Verify every intermediate shape in the Demucs pipeline."""
    model = Demucs()
    mix: Tensor[2, 2, 16384] = torch.randn(2, 2, 16384)

    # Encoder
    x0 = model.enc0(mix)
    assert_type(x0, Tensor[2, 64, 4095])
    x1 = model.enc1(x0)
    assert_type(x1, Tensor[2, 128, 1022])
    x2 = model.enc2(x1)
    assert_type(x2, Tensor[2, 256, 254])

    # BLSTM
    xb = model.lstm(x2)
    assert_type(xb, Tensor[2, 256, 254])

    # Decoder with skip connections
    skip2 = center_trim(x2, xb)
    assert_type(skip2, Tensor[2, 256, 254])
    d2 = model.dec2(xb + skip2)
    assert_type(d2, Tensor[2, 128, 1012])

    skip1 = center_trim(x1, d2)
    assert_type(skip1, Tensor[2, 128, 1012])
    d1 = model.dec1(d2 + skip1)
    assert_type(d1, Tensor[2, 64, 4044])

    skip0 = center_trim(x0, d1)
    assert_type(skip0, Tensor[2, 64, 4044])
    d0 = model.dec0(d1 + skip0)
    assert_type(d0, Tensor[2, 8, 16172])

    # Final reshape
    result = d0.view(d0.size(0), 4, 2, d0.size(-1))
    assert_type(result, Tensor[2, 4, 2, 16172])
