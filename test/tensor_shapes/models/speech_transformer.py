# Portions (c) Meta Platforms, Inc. and affiliates.
#
# This source code is adapted from pytorch/benchmark (TorchBenchmark),
# which is licensed under the BSD 3-Clause License:
# https://github.com/pytorch/benchmark/blob/main/LICENSE
#
# This adaptation adds tensor shape type annotations for pyrefly.

"""
Speech Transformer from TorchBenchmark with shape annotations.

Original: pytorch/benchmark/torchbenchmark/models/speech_transformer/
  speech_transformer/transformer/attention.py  (MultiHeadAttention, ScaledDotProductAttention)
  speech_transformer/transformer/module.py     (PositionwiseFeedForward)
  speech_transformer/transformer/encoder.py    (EncoderLayer)
  speech_transformer/transformer/decoder.py    (DecoderLayer)

Port notes:
- MultiHeadAttention parameterized as [NHead, DK] with DModel = NHead * DK derived
- PositionwiseFeedForward parameterized as [DModel, DInner]
- The full multi-head attention pattern is typed: Linear → view → permute →
  contiguous → view → bmm → view → permute → contiguous → view → Linear
- F.relu used instead of act_fn parameter
- mask operations (masked_fill, repeat) typed but mask shape
  not tracked (uses Tensor without shape params)
- PositionalEncoding ported: uses nn.Buffer (replaces register_buffer);
  Dim <: float allows math.log(10000.0) / d_model without cast;
  forward returns shapeless Tensor because symbolic slice on concrete buffer
  doesn't propagate shapes
- Encoder/Decoder ported with homogeneous ModuleList iteration:
  ModuleList[EncoderLayer[NHead, DK, DInner]] preserves Tensor[B, T, NHead*DK]
  through N layers. PositionalEncoding output skipped (shapeless).
  Mask/pad_mask args removed (mask shape not tracked).
- np.power(d_k, 0.5) replaced with d_k ** 0.5 (no numpy dependency)
"""

from typing import assert_type, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from torch import Tensor
    from torch_shapes import Dim


# ============================================================================
# Positional Encoding
# ============================================================================
# Original: speech_transformer/transformer/module.py PositionalEncoding
# Uses register_buffer to store precomputed positional encodings.
# Port: Uses nn.Buffer instead of register_buffer (same semantics, type-visible).
# Forward slices self.pe[:, :length] where length = input.size(1).
#
# Changes from original:
# - register_buffer("pe", pe) → self.pe = nn.Buffer(pe) (nn.Buffer pattern)
# - Slice assignment pe[:, 0::2] = ... works (in-place mutation during __init__)
# - math.log import added


import math


class PositionalEncoding[DModel](nn.Module):
    """Positional encoding using precomputed sin/cos embeddings.

    PE(pos, 2i)   = sin(pos / (10000^(2i/d_model)))
    PE(pos, 2i+1) = cos(pos / (10000^(2i/d_model)))

    Stores a [1, MaxLen, DModel] buffer and slices to input length.

    Changes from original:
    - register_buffer("pe", pe) → self.pe = nn.Buffer(pe) (type-visible attribute)
    - Uses .size(1) to get symbolic length for slicing
    """

    def __init__(self, d_model: Dim[DModel], max_len: int = 5000) -> None:
        super().__init__()
        pe: Tensor = torch.zeros(max_len, d_model)
        position: Tensor = torch.arange(0, max_len).unsqueeze(1).float()
        div_term: Tensor = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe_unsqueezed = pe.unsqueeze(0)
        # nn.Buffer replaces register_buffer — attribute is type-visible
        self.pe = nn.Buffer(pe_unsqueezed)

    def forward[B, T](self, input: Tensor[B, T, DModel]) -> Tensor:
        """Return positional encoding sliced to input length.

        self.pe[:, :length] slices the buffer's second dim to T elements.
        """
        length = input.size(1)
        return self.pe[:, :length]


# ============================================================================
# Scaled Dot-Product Attention
# ============================================================================


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention.

    Computes attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

    Input shapes (after multi-head reshape):
      q: Tensor[NB, Tq, DK]  (NB = NHead * B)
      k: Tensor[NB, Tk, DK]
      v: Tensor[NB, Tk, DV]

    Output: Tensor[NB, Tq, DV]
    """

    def __init__(self, temperature: float, attn_dropout: float = 0.1) -> None:
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward[NB, Tq, Tk, DK, DV](
        self,
        q: Tensor[NB, Tq, DK],
        k: Tensor[NB, Tk, DK],
        v: Tensor[NB, Tk, DV],
    ) -> tuple[Tensor[NB, Tq, DV], Tensor[NB, Tq, Tk]]:
        attn = torch.bmm(q, k.transpose(1, 2))
        assert_type(attn, Tensor[NB, Tq, Tk])
        attn_scaled = attn / self.temperature
        assert_type(attn_scaled, Tensor[NB, Tq, Tk])
        attn_normalized = self.softmax(attn_scaled)
        assert_type(attn_normalized, Tensor[NB, Tq, Tk])
        attn_dropped = self.dropout(attn_normalized)
        assert_type(attn_dropped, Tensor[NB, Tq, Tk])
        output = torch.bmm(attn_dropped, v)
        assert_type(output, Tensor[NB, Tq, DV])
        return output, attn_dropped


# ============================================================================
# Multi-Head Attention (concrete: DModel=512, NHead=8, DK=DV=64)
# ============================================================================


class MultiHeadAttention[NHead, DK](nn.Module):
    """Multi-Head Attention, generic over number of heads and key dimension.

    DModel = NHead * DK (derived, not an independent type param).

    The full reshape pattern for multi-head attention:
    1. Linear projection: [B, T, NHead*DK] → [B, T, NHead*DK]
    2. View:              [B, T, NHead*DK] → [B, T, NHead, DK]
    3. Permute:           [B, T, NHead, DK] → [NHead, B, T, DK]
    4. View:              [NHead, B, T, DK] → [NHead*B, T, DK]
    5. Attention:         [NHead*B, Tq, DK] x [NHead*B, Tk, DK] → [NHead*B, Tq, DK]
    6. View:              [NHead*B, Tq, DK] → [NHead, B, Tq, DK]
    7. Permute:           [NHead, B, Tq, DK] → [B, Tq, NHead, DK]
    8. View:              [B, Tq, NHead, DK] → [B, Tq, NHead*DK]
    9. Linear:            [B, Tq, NHead*DK] → [B, Tq, NHead*DK]
    """

    def __init__(self, n_head: Dim[NHead], d_k: Dim[DK], dropout: float = 0.1) -> None:
        super().__init__()
        d_model = n_head * d_k
        self.w_qs = nn.Linear(d_model, d_model)
        self.w_ks = nn.Linear(d_model, d_model)
        self.w_vs = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention(
            temperature=int(d_k) ** 0.5, attn_dropout=dropout
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.d_k = d_k

    def forward[B, Tq, Tk](
        self,
        q: Tensor[B, Tq, NHead * DK],
        k: Tensor[B, Tk, NHead * DK],
        v: Tensor[B, Tk, NHead * DK],
    ) -> tuple[Tensor[B, Tq, NHead * DK], Tensor]:
        residual = q
        d_model = q.size(2)

        # Project Q, K, V
        q_proj = self.w_qs(q)
        assert_type(q_proj, Tensor[B, Tq, NHead * DK])
        k_proj = self.w_ks(k)
        assert_type(k_proj, Tensor[B, Tk, NHead * DK])
        v_proj = self.w_vs(v)
        assert_type(v_proj, Tensor[B, Tk, NHead * DK])

        # Reshape to multi-head: [B, T, NHead*DK] → [B, T, NHead, DK]
        q_heads = q_proj.view(q.size(0), q.size(1), self.n_head, self.d_k)
        assert_type(q_heads, Tensor[B, Tq, NHead, DK])
        k_heads = k_proj.view(k.size(0), k.size(1), self.n_head, self.d_k)
        assert_type(k_heads, Tensor[B, Tk, NHead, DK])
        v_heads = v_proj.view(v.size(0), v.size(1), self.n_head, self.d_k)
        assert_type(v_heads, Tensor[B, Tk, NHead, DK])

        # Transpose to [NHead, B, T, DK] then flatten to [NHead*B, T, DK]
        q_flat = q_heads.permute(2, 0, 1, 3).contiguous().view(-1, q.size(1), self.d_k)
        assert_type(q_flat, Tensor[NHead * B, Tq, DK])
        k_flat = k_heads.permute(2, 0, 1, 3).contiguous().view(-1, k.size(1), self.d_k)
        assert_type(k_flat, Tensor[NHead * B, Tk, DK])
        v_flat = v_heads.permute(2, 0, 1, 3).contiguous().view(-1, v.size(1), self.d_k)
        assert_type(v_flat, Tensor[NHead * B, Tk, DK])

        # Scaled dot-product attention
        output_flat, attn = self.attention(q_flat, k_flat, v_flat)
        assert_type(output_flat, Tensor[NHead * B, Tq, DK])

        # Reshape back: [NHead*B, Tq, DK] → [NHead, B, Tq, DK]
        output_heads = output_flat.view(self.n_head, q.size(0), q.size(1), self.d_k)
        assert_type(output_heads, Tensor[NHead, B, Tq, DK])

        # Transpose and flatten: [B, Tq, NHead, DK] → [B, Tq, NHead*DK]
        output_cat = (
            output_heads.permute(1, 2, 0, 3)
            .contiguous()
            .view(q.size(0), q.size(1), d_model)
        )
        assert_type(output_cat, Tensor[B, Tq, NHead * DK])

        # Output projection + dropout + residual + layer norm
        output_proj = self.dropout(self.fc(output_cat))
        assert_type(output_proj, Tensor[B, Tq, NHead * DK])
        output_norm = self.layer_norm(output_proj + residual)
        assert_type(output_norm, Tensor[B, Tq, NHead * DK])

        return output_norm, attn


# ============================================================================
# Position-wise Feed-Forward Network
# ============================================================================


class PositionwiseFeedForward[DModel, DInner](nn.Module):
    """Position-wise feedforward sublayer.

    FFN(x) = max(0, xW1 + b1)W2 + b2
    With residual connection and layer normalization.
    """

    def __init__(
        self, d_model: Dim[DModel], d_inner: Dim[DInner], dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_inner)
        self.w_2 = nn.Linear(d_inner, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward[B, T](self, x: Tensor[B, T, DModel]) -> Tensor[B, T, DModel]:
        residual = x
        h1 = self.w_1(x)
        assert_type(h1, Tensor[B, T, DInner])
        h2 = self.w_2(F.relu(h1))
        assert_type(h2, Tensor[B, T, DModel])
        output = self.layer_norm(self.dropout(h2) + residual)
        assert_type(output, Tensor[B, T, DModel])
        return output


# ============================================================================
# Encoder Layer
# ============================================================================


class EncoderLayer[NHead, DK, DInner](nn.Module):
    """Single encoder layer: self-attention + feed-forward.

    Input:  Tensor[B, T, NHead*DK]
    Output: Tensor[B, T, NHead*DK]
    """

    def __init__(
        self,
        n_head: Dim[NHead],
        d_k: Dim[DK],
        d_inner: Dim[DInner],
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_k, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(n_head * d_k, d_inner, dropout=dropout)

    def forward[B, T](
        self, enc_input: Tensor[B, T, NHead * DK]
    ) -> tuple[Tensor[B, T, NHead * DK], Tensor]:
        enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input)
        assert_type(enc_output, Tensor[B, T, NHead * DK])
        enc_output_ffn = self.pos_ffn(enc_output)
        assert_type(enc_output_ffn, Tensor[B, T, NHead * DK])
        return enc_output_ffn, enc_slf_attn


# ============================================================================
# Decoder Layer
# ============================================================================


class DecoderLayer[NHead, DK, DInner](nn.Module):
    """Single decoder layer: self-attention + cross-attention + feed-forward.

    Input:
      dec_input: Tensor[B, Td, NHead*DK]
      enc_output: Tensor[B, Te, NHead*DK]
    Output: Tensor[B, Td, NHead*DK]
    """

    def __init__(
        self,
        n_head: Dim[NHead],
        d_k: Dim[DK],
        d_inner: Dim[DInner],
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_k, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_k, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(n_head * d_k, d_inner, dropout=dropout)

    def forward[B, Td, Te](
        self,
        dec_input: Tensor[B, Td, NHead * DK],
        enc_output: Tensor[B, Te, NHead * DK],
    ) -> tuple[Tensor[B, Td, NHead * DK], Tensor, Tensor]:
        # Self-attention (decoder attends to itself)
        dec_slf_out, dec_slf_attn = self.slf_attn(dec_input, dec_input, dec_input)
        assert_type(dec_slf_out, Tensor[B, Td, NHead * DK])

        # Cross-attention (decoder attends to encoder output)
        dec_enc_out, dec_enc_attn = self.enc_attn(dec_slf_out, enc_output, enc_output)
        assert_type(dec_enc_out, Tensor[B, Td, NHead * DK])

        # Feed-forward
        dec_ffn_out = self.pos_ffn(dec_enc_out)
        assert_type(dec_ffn_out, Tensor[B, Td, NHead * DK])

        return dec_ffn_out, dec_slf_attn, dec_enc_attn


# ============================================================================
# Encoder (top-level: PositionalEncoding + N stacked EncoderLayers)
# ============================================================================
# Original: speech_transformer/transformer/encoder.py
# Uses ModuleList iteration — each EncoderLayer preserves shape, so
# the loop is homogeneous and works directly.
#
# Changes from original:
# - Type annotations added
# - mask/pad_mask args removed (mask shape not tracked)
# - position_enc output is shapeless (concrete buffer + symbolic slice),
#   so we skip adding it and just pass through. The original adds it to src_seq.


class Encoder[NHead, DK, DInner](nn.Module):
    """Encoder: N stacked EncoderLayers.

    Input:  Tensor[B, T, NHead*DK]
    Output: Tensor[B, T, NHead*DK]
    """

    def __init__(
        self,
        n_head: Dim[NHead],
        d_k: Dim[DK],
        d_inner: Dim[DInner],
        n_layers: int = 6,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        d_model = n_head * d_k
        self.position_enc = PositionalEncoding(d_model)
        self.layer_stack = nn.ModuleList(
            [
                EncoderLayer(n_head, d_k, d_inner, dropout=dropout)
                for _ in range(n_layers)
            ]
        )

    def forward[B, T](
        self, src_seq: Tensor[B, T, NHead * DK]
    ) -> Tensor[B, T, NHead * DK]:
        enc_output = src_seq
        for layer in self.layer_stack:
            enc_output, _attn = layer(enc_output)
            assert_type(enc_output, Tensor[B, T, NHead * DK])
        return enc_output


# ============================================================================
# Decoder (top-level: PositionalEncoding + N stacked DecoderLayers)
# ============================================================================
# Original: speech_transformer/transformer/decoder.py
# Same homogeneous ModuleList iteration pattern as Encoder.
#
# Changes from original:
# - Type annotations added
# - mask/pad_mask args removed
# - position_enc skipped (same reason as Encoder)


class Decoder[NHead, DK, DInner](nn.Module):
    """Decoder: N stacked DecoderLayers.

    Input:
      tgt_seq:    Tensor[B, Td, NHead*DK]
      enc_output: Tensor[B, Te, NHead*DK]
    Output: Tensor[B, Td, NHead*DK]
    """

    def __init__(
        self,
        n_head: Dim[NHead],
        d_k: Dim[DK],
        d_inner: Dim[DInner],
        n_layers: int = 6,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        d_model = n_head * d_k
        self.position_enc = PositionalEncoding(d_model)
        self.layer_stack = nn.ModuleList(
            [
                DecoderLayer(n_head, d_k, d_inner, dropout=dropout)
                for _ in range(n_layers)
            ]
        )

    def forward[B, Td, Te](
        self,
        tgt_seq: Tensor[B, Td, NHead * DK],
        enc_output: Tensor[B, Te, NHead * DK],
    ) -> Tensor[B, Td, NHead * DK]:
        dec_output = tgt_seq
        for layer in self.layer_stack:
            dec_output, _slf_attn, _enc_attn = layer(dec_output, enc_output)
            assert_type(dec_output, Tensor[B, Td, NHead * DK])
        return dec_output


# ============================================================================
# Smoke tests
# ============================================================================


def test_scaled_dot_product_attention():
    """Test attention: [NB, Tq, DK] x [NB, Tk, DK] → [NB, Tq, DV]."""
    attn = ScaledDotProductAttention(temperature=8.0)
    q: Tensor[16, 10, 64] = torch.randn(16, 10, 64)
    k: Tensor[16, 20, 64] = torch.randn(16, 20, 64)
    v: Tensor[16, 20, 64] = torch.randn(16, 20, 64)
    output, attn_weights = attn(q, k, v)
    assert_type(output, Tensor[16, 10, 64])
    assert_type(attn_weights, Tensor[16, 10, 20])


def test_multi_head_attention_self():
    """Test self-attention: [B, T, 512] → [B, T, 512] with NHead=8, DK=64."""
    mha = MultiHeadAttention(8, 64)
    x: Tensor[2, 10, 512] = torch.randn(2, 10, 512)
    output, attn = mha(x, x, x)
    assert_type(output, Tensor[2, 10, 512])


def test_multi_head_attention_cross():
    """Test cross-attention: q=[B, Tq, 512], kv=[B, Tk, 512] → [B, Tq, 512]."""
    mha = MultiHeadAttention(8, 64)
    q: Tensor[2, 10, 512] = torch.randn(2, 10, 512)
    kv: Tensor[2, 20, 512] = torch.randn(2, 20, 512)
    output, attn = mha(q, kv, kv)
    assert_type(output, Tensor[2, 10, 512])


def test_positionwise_ffn():
    """Test FFN: [B, T, 512] → [B, T, 512]."""
    ffn = PositionwiseFeedForward(512, 2048)
    x: Tensor[2, 10, 512] = torch.randn(2, 10, 512)
    output = ffn(x)
    assert_type(output, Tensor[2, 10, 512])


def test_encoder_layer():
    """Test encoder layer: [B, T, 512] → [B, T, 512]."""
    layer = EncoderLayer(8, 64, 2048)
    x: Tensor[2, 10, 512] = torch.randn(2, 10, 512)
    output, attn = layer(x)
    assert_type(output, Tensor[2, 10, 512])


def test_decoder_layer():
    """Test decoder layer: dec=[B, Td, 512], enc=[B, Te, 512] → [B, Td, 512]."""
    layer = DecoderLayer(8, 64, 2048)
    dec: Tensor[2, 10, 512] = torch.randn(2, 10, 512)
    enc: Tensor[2, 20, 512] = torch.randn(2, 20, 512)
    output, slf_attn, enc_attn = layer(dec, enc)
    assert_type(output, Tensor[2, 10, 512])


def test_positional_encoding():
    """Test PositionalEncoding: stores precomputed pe, slices to input length.

    Original: register_buffer("pe", pe) + self.pe[:, :length]
    Port: nn.Buffer(pe) + self.pe[:, :length]
    """
    pe = PositionalEncoding(512)
    x: Tensor[2, 10, 512] = torch.randn(2, 10, 512)
    out = pe(x)
    # Result is shapeless Tensor because pe buffer is concrete-shaped
    # and slicing with symbolic length doesn't propagate
    assert_type(out, Tensor)


def test_encoder():
    """Test Encoder: [B, T, 512] -> [B, T, 512] through 6 layers via ModuleList."""
    encoder = Encoder(8, 64, 2048, n_layers=6)
    x: Tensor[2, 10, 512] = torch.randn(2, 10, 512)
    out = encoder(x)
    assert_type(out, Tensor[2, 10, 512])


def test_decoder():
    """Test Decoder: dec=[B, Td, 512], enc=[B, Te, 512] -> [B, Td, 512] through 6 layers."""
    decoder = Decoder(8, 64, 2048, n_layers=6)
    dec: Tensor[2, 10, 512] = torch.randn(2, 10, 512)
    enc: Tensor[2, 20, 512] = torch.randn(2, 20, 512)
    out = decoder(dec, enc)
    assert_type(out, Tensor[2, 10, 512])
