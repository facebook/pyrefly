# Portions (c) Meta Platforms, Inc. and affiliates.
#
# This source code is adapted from pytorch/benchmark (TorchBenchmark),
# which is licensed under the BSD 3-Clause License:
# https://github.com/pytorch/benchmark/blob/main/LICENSE
#
# This adaptation adds tensor shape type annotations for pyrefly.

"""
LLaMA decoder-only transformer from TorchBenchmark with shape annotations.

Original: pytorch/benchmark/torchbenchmark/models/llama/model.py

Port notes:
- Decoder-only transformer with RMSNorm (pre-norm), SwiGLU feedforward, and
  multi-head self-attention with rotary positional embeddings (RoPE).
- SwiGLU: w2(silu(w1(x)) * w3(x)). Two parallel Linear projections to
  HiddenDim, elementwise multiply, project back. This is the key new pattern
  vs nanogpt (which uses GELU MLP) and gptfast.
- RoPE: apply_rotary_emb uses complex tensor ops (view_as_complex, view_as_real)
  that aren't shape-tracked. Modeled as a shape-preserving operation on the
  (B, NHead, T, HeadDim) tensors. Included with annotation fallbacks.
- precompute_freqs_cis: builds complex-valued frequency tensor for RoPE.
  Uses torch.polar (shapeless) with annotation fallback.
- Causal mask: builds upper-triangular attention mask. Included.
- KV cache: inference-only optimization. Included with pre-allocated buffers
  and in-place `scatter_`-style updates via start_pos indexing. The forward
  path supports optional start_pos for cached inference.
- ModelArgs simplified to direct constructor parameters with type vars.
- Model parallelism: the original uses n_local_heads = n_heads // world_size.
  Included as a comment and default world_size=1 — the math works out to
  n_local_heads = n_heads when world_size=1.

Key patterns exercised:
- SwiGLU feedforward: silu(w1(x)) * w3(x) → w2 → output
- Pre-norm transformer blocks with RMSNorm (vs LayerNorm in nanogpt)
- Multi-head attention with reshape + transpose pattern
- Config dataclass extracting type params (like nanogpt)
- Homogeneous layer stacking via ModuleList
- RoPE (rotary positional embeddings) with complex tensor ops
- KV cache for inference
- Causal attention masking
"""

import math
from dataclasses import dataclass
from typing import Any, assert_type, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from torch import Tensor
    from torch_shapes import Dim


# ============================================================================
# Config
# ============================================================================


@dataclass
class LLaMAConfig[VocabSize, D, NHead, NLayer, HiddenDim, MaxBatch, MaxSeq]:
    """Configuration for LLaMA model, generic over key dimensions."""

    vocab_size: Dim[VocabSize]
    dim: Dim[D]
    n_heads: Dim[NHead]
    n_layers: Dim[NLayer]
    hidden_dim: Dim[HiddenDim]
    max_batch_size: Dim[MaxBatch]
    max_seq_len: Dim[MaxSeq]
    norm_eps: float = 1e-5


def compute_hidden_dim(dim: int, multiple_of: int = 256) -> int:
    """Compute SwiGLU hidden dimension rounded up to multiple_of."""
    hidden_dim = int(2 * 4 * dim / 3)
    return multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)


# ============================================================================
# RoPE (Rotary Positional Embeddings)
# ============================================================================


def precompute_freqs_cis[HeadDim, End](
    dim: Dim[HeadDim], end: Dim[End], theta: float = 10000.0
) -> Tensor[End, HeadDim // 2]:
    """Precompute complex-valued frequency tensor for RoPE.

    Original: llama/model.py precompute_freqs_cis function.

    Returns a (End, HeadDim//2) complex tensor. arange and outer are DSL-tracked.

    Args:
        dim: head dimension (frequencies use dim//2 pairs)
        end: maximum sequence length (×2 for RoPE)
        theta: base frequency
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(end)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def reshape_for_broadcast[T, HD](
    freqs_cis: Tensor[T, HD], x: Tensor
) -> Tensor[1, 1, T, HD]:
    """Reshape frequency tensor for broadcasting with multi-head tensors.

    Original: llama/model.py reshape_for_broadcast function.

    freqs_cis is (T, HeadDim//2), reshaped to (1, 1, T, HeadDim//2) to
    broadcast with (B, NHead, T, HeadDim//2).
    """
    t, hd = freqs_cis.shape
    return freqs_cis.view(1, 1, t, hd)


def apply_rotary_emb[B, NHead, T, HeadDim](
    xq: Tensor[B, NHead, T, HeadDim],
    xk: Tensor[B, NHead, T, HeadDim],
    freqs_cis: Tensor[T, HeadDim // 2],
) -> tuple[Tensor[B, NHead, T, HeadDim], Tensor[B, NHead, T, HeadDim]]:
    """Apply rotary positional embeddings to query and key tensors.

    Original: llama/model.py apply_rotary_emb function.

    Shape-preserving: (B, NHead, T, HeadDim) → (B, NHead, T, HeadDim).
    reshape + view_as_complex → (B, NHead, T, HeadDim//2)
    * freqs_cis (1, 1, T, HeadDim//2) broadcasts
    view_as_real → (B, NHead, T, HeadDim//2, 2)
    flatten(3) → (B, NHead, T, (HeadDim//2)*2) — A1 gap: can't prove = HeadDim
    """
    xq_c = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    assert_type(xq_c, Tensor[B, NHead, T, HeadDim // 2])
    xk_c = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    assert_type(xk_c, Tensor[B, NHead, T, HeadDim // 2])
    freqs_bc = reshape_for_broadcast(freqs_cis, xq_c)
    assert_type(freqs_bc, Tensor[1, 1, T, HeadDim // 2])
    xq_out = torch.view_as_real(xq_c * freqs_bc).flatten(3)
    xk_out = torch.view_as_real(xk_c * freqs_bc).flatten(3)
    # A1: (HeadDim//2)*2 can't be proven equal to HeadDim
    return xq_out.type_as(xq), xk_out.type_as(xk)  # type: ignore[bad-return]


# ============================================================================
# Causal Mask
# ============================================================================


def build_causal_mask[T](seq_len: Dim[T]) -> Tensor[T, T]:
    """Build causal (upper-triangular) attention mask.

    Original: llama/model.py, constructed in Transformer.__init__.

    Returns a (seq_len, seq_len) mask with -inf above the diagonal and
    0 on/below the diagonal, used to prevent attending to future positions.
    """
    mask: Tensor = torch.full((seq_len, seq_len), float("-inf"))
    mask = torch.triu(mask, diagonal=1)
    return mask


# ============================================================================
# RMSNorm
# ============================================================================


class RMSNorm[D](nn.Module):
    """Root Mean Square Layer Normalization.

    Shape-preserving: (*, D) → (*, D)
    """

    def __init__(self, dim: Dim[D], eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward[*Bs](self, x: Tensor[*Bs, D]) -> Tensor[*Bs, D]:
        normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return normed * self.weight


# ============================================================================
# FeedForward (SwiGLU)
# ============================================================================


class FeedForward[D, HiddenDim](nn.Module):
    """SwiGLU feedforward network.

    Architecture: silu(w1(x)) * w3(x) → w2 → output
    Two parallel projections D → HiddenDim, elementwise multiply, then
    HiddenDim → D.

    (*, D) → (*, D)
    """

    def __init__(
        self,
        config: LLaMAConfig[Any, D, Any, Any, HiddenDim, Any, Any],
    ) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.w2 = nn.Linear(config.hidden_dim, config.dim, bias=False)
        self.w3 = nn.Linear(config.dim, config.hidden_dim, bias=False)

    def forward[B, T](self, x: Tensor[B, T, D]) -> Tensor[B, T, D]:
        gate = F.silu(self.w1(x))
        assert_type(gate, Tensor[B, T, HiddenDim])
        up = self.w3(x)
        assert_type(up, Tensor[B, T, HiddenDim])
        return self.w2(gate * up)


# ============================================================================
# Attention
# ============================================================================


class Attention[D, NHead, MaxBatch, MaxSeq](nn.Module):
    """Multi-head self-attention with RoPE and KV cache.

    Shape flow:
        x (B, T, D) → wq, wk, wv → each (B, T, D)
        → reshape (B, T, NHead, HeadDim) → transpose (B, NHead, T, HeadDim)
        → [RoPE on q, k — shape-preserving]
        → KV cache: write current, read back (B, NHead, CachedT, HeadDim)
        → scores = q @ k^T / sqrt(HeadDim) → (B, NHead, T, CachedT)
        → softmax → @ v → (B, NHead, T, HeadDim)
        → transpose + reshape → (B, T, D) → wo → (B, T, D)

    (B, T, D) → (B, T, D)
    """

    def __init__(
        self,
        config: LLaMAConfig[Any, D, NHead, Any, Any, MaxBatch, MaxSeq],
    ) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.dim // config.n_heads
        self.wq = nn.Linear(config.dim, config.dim, bias=False)
        self.wk = nn.Linear(config.dim, config.dim, bias=False)
        self.wv = nn.Linear(config.dim, config.dim, bias=False)
        self.wo = nn.Linear(config.dim, config.dim, bias=False)
        # KV cache: pre-allocated buffers for inference.
        # Shape: (MaxBatch, MaxSeq, NHead, HeadDim). torch.zeros with Dim args → typed.
        self.cache_k = torch.zeros(
            (config.max_batch_size, config.max_seq_len, config.n_heads, self.head_dim)
        )
        self.cache_v = torch.zeros(
            (config.max_batch_size, config.max_seq_len, config.n_heads, self.head_dim)
        )

    def forward[B, T, SP](
        self,
        x: Tensor[B, T, D],
        start_pos: Dim[SP] | None = None,
        freqs_cis: Tensor[T, D // NHead // 2] | None = None,
        mask: Tensor[T, T] | None = None,
    ) -> Tensor[B, T, D]:
        b, t, d = x.size()
        assert_type(b, Dim[B])
        assert_type(t, Dim[T])
        assert_type(d, Dim[D])
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        assert_type(q, Tensor[B, T, D])
        assert_type(k, Tensor[B, T, D])
        assert_type(v, Tensor[B, T, D])
        # Reshape to multi-head: (B, T, D) → (B, T, NHead, HeadDim) → (B, NHead, T, HeadDim)
        xq = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        xk = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        xv = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        assert_type(xq, Tensor[B, NHead, T, D // NHead])
        assert_type(xk, Tensor[B, NHead, T, D // NHead])
        assert_type(xv, Tensor[B, NHead, T, D // NHead])
        # Apply RoPE if frequency tensor provided (shape-preserving)
        if freqs_cis is not None:
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis)
        # Attention with optional KV cache.
        # Each branch computes output independently to avoid branch join widening.
        if start_pos is not None:
            # Cached inference: write to cache, read back full range
            xk_cache = xk.transpose(1, 2)
            xv_cache = xv.transpose(1, 2)
            with torch.no_grad():
                self.cache_k[:b, start_pos : start_pos + t] = xk_cache
                self.cache_v[:b, start_pos : start_pos + t] = xv_cache
            # Keys/values: (B, NHead, SP+T, HeadDim)
            keys = self.cache_k[:b, : start_pos + t].transpose(1, 2)
            values = self.cache_v[:b, : start_pos + t].transpose(1, 2)
            # Attention: (B, NHead, T, SP+T) contracts to (B, NHead, T, HeadDim)
            scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
            scores = F.softmax(scores.float(), dim=-1)
            output = torch.matmul(scores, values)
        else:
            # No cache: keys = xk, values = xv
            scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
            assert_type(scores, Tensor[B, NHead, T, T])
            if mask is not None:
                scores = scores + mask
            scores = F.softmax(scores.float(), dim=-1)
            assert_type(scores, Tensor[B, NHead, T, T])
            output = torch.matmul(scores, xv)
        assert_type(output, Tensor[B, NHead, T, D // NHead])
        # Reassemble: (B, NHead, T, HeadDim) → (B, T, NHead, HeadDim) → (B, T, D)
        output_flat = output.transpose(1, 2).contiguous().view(b, t, d)
        assert_type(output_flat, Tensor[B, T, D])
        return self.wo(output_flat)


# ============================================================================
# TransformerBlock
# ============================================================================


class TransformerBlock[D, NHead, HiddenDim, MaxBatch, MaxSeq](nn.Module):
    """Pre-norm transformer block with RMSNorm.

    x → RMSNorm → Attention → + residual
      → RMSNorm → FeedForward → + residual

    (B, T, D) → (B, T, D)
    """

    def __init__(
        self,
        config: LLaMAConfig[Any, D, NHead, Any, HiddenDim, MaxBatch, MaxSeq],
    ) -> None:
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)

    def forward[B, T, SP](
        self,
        x: Tensor[B, T, D],
        start_pos: Dim[SP] | None = None,
        freqs_cis: Tensor[T, D // NHead // 2] | None = None,
        mask: Tensor[T, T] | None = None,
    ) -> Tensor[B, T, D]:
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        assert_type(h, Tensor[B, T, D])
        out = h + self.feed_forward(self.ffn_norm(h))
        assert_type(out, Tensor[B, T, D])
        return out


# ============================================================================
# Transformer (full model)
# ============================================================================


class Transformer[VocabSize, D, NHead, HiddenDim](nn.Module):
    """LLaMA decoder-only transformer.

    tokens (B, T) → Embedding → (B, T, D)
        → N × TransformerBlock → (B, T, D)
        → RMSNorm → (B, T, D)
        → Linear → (B, T, VocabSize)
        → last token → (B, VocabSize)

    (B, T) → (B, VocabSize)
    """

    def __init__(
        self,
        config: LLaMAConfig[VocabSize, D, NHead, Any, HiddenDim, Any, Any],
    ) -> None:
        super().__init__()
        self.head_dim = config.dim // config.n_heads
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        layers: list[TransformerBlock[D, NHead, HiddenDim, Any, Any]] = []
        for _ in range(config.n_layers):
            layers.append(TransformerBlock(config))
        self.layers = nn.ModuleList(layers)
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

    def forward[B, T, SP](
        self, tokens: Tensor[B, T], start_pos: Dim[SP] | None = None
    ) -> Tensor[B, VocabSize]:
        h = self.tok_embeddings(tokens)
        assert_type(h, Tensor[B, T, D])
        # Precompute RoPE frequencies and causal mask for this sequence
        seq_len = tokens.shape[1]
        freqs_cis = precompute_freqs_cis(self.head_dim, seq_len * 2)
        # Slice to the relevant range for this position
        if start_pos is not None:
            freqs_cis = freqs_cis[start_pos : start_pos + seq_len]
        else:
            freqs_cis = freqs_cis[:seq_len]
        # Build causal mask only for multi-token sequences (prefill).
        mask: Tensor[T, T] | None = None
        if start_pos is None and seq_len > 1:
            mask = build_causal_mask(seq_len)
        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        assert_type(h, Tensor[B, T, D])
        h = self.norm(h)
        assert_type(h, Tensor[B, T, D])
        # Only compute logits for the last token
        last = h[:, -1]
        assert_type(last, Tensor[B, D])
        return self.output(last)


# ============================================================================
# Smoke tests
# ============================================================================


def test_rmsnorm():
    """Test RMSNorm: shape-preserving."""
    norm = RMSNorm(256)
    x: Tensor[4, 16, 256] = torch.randn(4, 16, 256)
    out = norm(x)
    assert_type(out, Tensor[4, 16, 256])


def test_feedforward():
    """Test SwiGLU feedforward: (B, T, D) → (B, T, D)."""
    config = LLaMAConfig(
        vocab_size=1000,
        dim=256,
        n_heads=4,
        n_layers=2,
        hidden_dim=768,
        max_batch_size=32,
        max_seq_len=2048,
    )
    ff = FeedForward(config)
    x: Tensor[4, 16, 256] = torch.randn(4, 16, 256)
    out = ff(x)
    assert_type(out, Tensor[4, 16, 256])


def test_attention():
    """Test multi-head self-attention: (B, T, D) → (B, T, D)."""
    config = LLaMAConfig(
        vocab_size=1000,
        dim=256,
        n_heads=4,
        n_layers=2,
        hidden_dim=768,
        max_batch_size=32,
        max_seq_len=2048,
    )
    attn = Attention(config)
    x: Tensor[4, 16, 256] = torch.randn(4, 16, 256)
    out = attn(x)
    assert_type(out, Tensor[4, 16, 256])


def test_transformer_block():
    """Test transformer block: (B, T, D) → (B, T, D)."""
    config = LLaMAConfig(
        vocab_size=1000,
        dim=256,
        n_heads=4,
        n_layers=2,
        hidden_dim=768,
        max_batch_size=32,
        max_seq_len=2048,
    )
    block = TransformerBlock(config)
    x: Tensor[4, 16, 256] = torch.randn(4, 16, 256)
    out = block(x)
    assert_type(out, Tensor[4, 16, 256])


def test_transformer():
    """Test full LLaMA transformer: tokens → logits."""
    config = LLaMAConfig(
        vocab_size=1000,
        dim=256,
        n_heads=4,
        n_layers=2,
        hidden_dim=768,
        max_batch_size=32,
        max_seq_len=2048,
    )
    model = Transformer(config)
    tokens: Tensor[2, 16] = torch.randint(0, 1000, (2, 16))
    out = model(tokens)
    assert_type(out, Tensor[2, 1000])


def test_transformer_different_dims():
    """Test LLaMA with different dimensions."""
    config = LLaMAConfig(
        vocab_size=500,
        dim=128,
        n_heads=8,
        n_layers=4,
        hidden_dim=512,
        max_batch_size=32,
        max_seq_len=2048,
    )
    model = Transformer(config)
    tokens: Tensor[8, 32] = torch.randint(0, 500, (8, 32))
    out = model(tokens)
    assert_type(out, Tensor[8, 500])
