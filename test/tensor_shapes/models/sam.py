# Portions (c) Meta Platforms, Inc. and affiliates.
#
# This source code is adapted from pytorch/benchmark (TorchBenchmark),
# which is licensed under the BSD 3-Clause License:
# https://github.com/pytorch/benchmark/blob/main/LICENSE
#
# This adaptation adds tensor shape type annotations for pyrefly.

"""
SAM (Segment Anything Model) from TorchBenchmark with shape annotations.

Original: pytorch/benchmark/torchbenchmark/models/sam/
  - image_encoder.py, transformer.py, common.py, mask_decoder.py, prompt_encoder.py

Port notes:
- SAM has three components: image encoder (ViT), prompt encoder, and mask decoder.
- The image encoder operates in BHWC format (B, H, W, C), not the standard
  BCHW (B, C, H, W). This is unusual for vision models.
- Windowed attention (window_partition/unpartition) included. The partition
  merges batch and window-count dims via view(-1, ...) which is not shape-
  trackable. window_size=0 means global attention (no partition).
- Relative positional embeddings (get_rel_pos, add_decomposed_rel_pos) included.
  get_rel_pos is fully typed: F.interpolate, torch.arange, fancy indexing all
  tracked. add_decomposed_rel_pos takes independent dims first (bare Dim params)
  so the checker binds QH/QW/KH/KW before processing derived expressions in
  tensor types. ImageAttention/ViTBlock use IS (class param) for square spatial
  dims — same type var for both rel_pos (construction) and forward (usage).
  torch.einsum results are shapeless (not tracked).
- The image encoder's Attention uses a combined QKV projection (Linear(D, 3*D))
  followed by reshape+permute+reshape+unbind. This entire chain is fully
  shape-tracked through all 4 steps, producing typed q/k/v tensors.
  Attention matmul, softmax, and head reassemble are also tracked.
  `self.scale: float` annotation prevents `Any` from `Dim ** float` (typeshed gap).
- The two-way transformer's Attention (cross-attention) is cleaner: separate
  Q/K/V projections with reshape→transpose for multi-head, similar to LLaMA.
  Fully shape-tracked.
- Fixed patch_size=16 for concrete shape computation.
- PromptEncoder: multi-modal input handling (points, boxes, masks). Token count
  is data-dependent → sparse embeddings use unrefined Tensor. Dense embeddings
  from mask downsampling or learned no-mask embedding also unrefined due to
  shapeless ops (registered buffer matmul, expand, reshape).
- MaskDecoder: orchestrates TwoWayTransformer with prompt/image tokens, then
  upscales image features and applies hypernetwork MLPs per mask. Heavy dynamic
  reshaping (flatten, transpose, view, expand) → uses annotation fallbacks.
- Sam: top-level model wrapping all three components.

Key patterns exercised:
- BHWC tensor format (unusual for vision)
- Patch embedding: Conv2d with large kernel/stride → permute to BHWC
- Cross-attention: separate Q/K/V sources with different sequence lengths
- Two-way (bidirectional) attention blocks
- Neck: BHWC → permute → BCHW → 1×1 conv + 3×3 conv
- Shape-preserving transformer blocks in BHWC format
- Multi-modal prompt encoding (points, boxes, masks) with data-dependent shapes
- Hypernetwork MLPs for mask prediction
- ConvTranspose2d upscaling in mask decoder
"""

import math
from typing import Any, assert_type, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from torch import Tensor
    from torch_shapes import Dim


# ============================================================================
# Common Modules
# ============================================================================


class LayerNorm2d[C](nn.Module):
    """Channel-wise LayerNorm for (B, C, H, W) tensors.

    Normalizes along channel dim. Shape-preserving.

    (B, C, H, W) → (B, C, H, W)
    """

    def __init__(self, num_channels: Dim[C], eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward[B, H, W](self, x: Tensor[B, C, H, W]) -> Tensor[B, C, H, W]:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x_norm = (x - u) / torch.sqrt(s + self.eps)
        out = self.weight[:, None, None] * x_norm + self.bias[:, None, None]
        assert_type(out, Tensor[B, C, H, W])
        return out


class MLPBlock[D, MlpDim](nn.Module):
    """MLP block: Linear → GELU → Linear. Shape-preserving.

    (*, D) → (*, D)
    """

    def __init__(self, embedding_dim: Dim[D], mlp_dim: Dim[MlpDim]) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)

    def forward[B, N](self, x: Tensor[B, N, D]) -> Tensor[B, N, D]:
        h = nn.functional.gelu(self.lin1(x))
        assert_type(h, Tensor[B, N, MlpDim])
        return self.lin2(h)


# ============================================================================
# Windowed Attention Helpers
# ============================================================================


def window_partition[B, H, W, WS, D](
    x: Tensor[B, H, W, D], window_size: Dim[WS]
) -> Tensor[B * (H // WS) * (W // WS), WS, WS, D]:
    """Partition into non-overlapping windows.

    Original: image_encoder.py window_partition function.

    (B, H, W, D) → view (B, H//WS, WS, W//WS, WS, D)
        → permute (B, H//WS, W//WS, WS, WS, D) → view (-1, WS, WS, D)

    Assumes H, W divisible by WS (true for all SAM configs).
    """
    b, h, w, c = x.shape
    nw_h = h // window_size
    nw_w = w // window_size
    x_view = x.view(b, nw_h, window_size, nw_w, window_size, c)
    return (
        x_view.permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .view(-1, window_size, window_size, c)
    )


def window_unpartition[B, H, W, WS, D](
    window_size: Dim[WS],
    h: Dim[H],
    w: Dim[W],
    batch_size: Dim[B],
    windows: Tensor[B * (H // WS) * (W // WS), WS, WS, D],
) -> Tensor[B, H, W, D]:
    """Undo window partition: reassemble windows into spatial grid.

    Original: image_encoder.py window_unpartition function.

    (B*(H//WS)*(W//WS), WS, WS, D) → view (B, H//WS, W//WS, WS, WS, D)
        → permute → view (B, H, W, D)
    """
    nw_h = h // window_size
    nw_w = w // window_size
    c = windows.shape[3]
    x = windows.view(batch_size, nw_h, nw_w, window_size, window_size, c)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(batch_size, h, w, c)


# ============================================================================
# Relative Positional Embedding Helpers
# ============================================================================


def get_rel_pos[QS, KS, HD, RP](
    q_size: Dim[QS], k_size: Dim[KS], rel_pos: Tensor[RP, HD]
) -> Tensor[QS, KS, HD]:
    """Get relative positional embeddings for query/key size pair.

    Original: image_encoder.py get_rel_pos function.

    Uses F.interpolate to resize the learned embedding table if needed,
    then indexes by computed relative coordinates. With Dim-typed args,
    the full chain is shape-tracked: reshape, permute, interpolate,
    arange, broadcasting subtraction, and fancy indexing all produce
    typed outputs.

    (2*QS-1, HD) → (QS, KS, HD)
    """
    max_rel_dist = 2 * q_size - 1
    if rel_pos.shape[0] != max_rel_dist:
        # Resize learned embeddings to match current resolution.
        rel_pos_resized = (
            F.interpolate(
                rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
                size=max_rel_dist,
                mode="linear",
            )
            .reshape(-1, max_rel_dist)
            .permute(1, 0)
        )
    else:
        rel_pos_resized = rel_pos

    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)
    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos[B, QH, QW, KH, KW, HD, RPH, RPW](
    q_h: Dim[QH],
    q_w: Dim[QW],
    k_h: Dim[KH],
    k_w: Dim[KW],
    attn: Tensor[B, QH * QW, KH * KW],
    q: Tensor[B, QH * QW, HD],
    rel_pos_h: Tensor[RPH, HD],
    rel_pos_w: Tensor[RPW, HD],
) -> Tensor[B, QH * QW, KH * KW]:
    """Add decomposed relative position bias to attention scores.

    Original: image_encoder.py add_decomposed_rel_pos function.

    Decomposes 2D relative position into separate height and width components
    (from MViTv2). Computes q @ Rh and q @ Rw via torch.einsum, then adds
    to attention map with broadcasting.

    Independent type params (QH, QW, KH, KW) appear as bare Dim params
    first, so the checker binds them before processing derived expressions
    (QH*QW, 2*QH-1) in the tensor types.

    Shape-preserving on attn: (B, QH*QW, KH*KW) → (B, QH*QW, KH*KW)
    """
    rh = get_rel_pos(q_h, k_h, rel_pos_h)
    assert_type(rh, Tensor[QH, KH, HD])
    rw = get_rel_pos(q_w, k_w, rel_pos_w)
    assert_type(rw, Tensor[QW, KW, HD])

    b, _, dim = q.shape
    r_q = q.reshape(b, q_h, q_w, dim)
    # einsum: shapeless (not tracked), but inputs are typed
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, rw)

    attn = (
        attn.view(b, q_h, q_w, k_h, k_w)
        + rel_h[:, :, :, :, None]
        + rel_w[:, :, :, None, :]
    ).view(b, q_h * q_w, k_h * k_w)

    return attn


# ============================================================================
# Image Encoder: Attention (BHWC format)
# ============================================================================


class ImageAttention[D, NHead, IS](nn.Module):
    """Multi-head self-attention in BHWC format.

    Combined QKV projection: Linear(D, 3*D) → reshape+permute+unbind → q, k, v
    → scaled dot-product attention → view+permute+reshape → output projection.
    Fully shape-tracked through the entire chain (reshape, permute, unbind,
    matmul, view all tracked with Dim args).

    When use_rel_pos=True, learned relative position embeddings of shape
    (2*IS-1, D//NHead) are tracked via the IS type param.

    (B, H, W, D) → (B, H, W, D)
    """

    def __init__(
        self,
        dim: Dim[D],
        num_heads: Dim[NHead],
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Dim[IS] | None = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale: float = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.use_rel_pos = use_rel_pos
        if use_rel_pos:
            assert input_size is not None
            # Learned relative position embeddings for height and width
            # Shape: (2*IS-1, D//NHead). Tracked via Dim[IS] type param.
            self.rel_pos_h = nn.Parameter(
                torch.zeros(2 * input_size - 1, self.head_dim)
            )
            self.rel_pos_w = nn.Parameter(
                torch.zeros(2 * input_size - 1, self.head_dim)
            )

    def forward[B, H, W](self, x: Tensor[B, H, W, D]) -> Tensor[B, H, W, D]:
        b, h, w, d = x.size()
        # Combined QKV: (B, H, W, D) → Linear → (B, H, W, 3*D)
        qkv_out = self.qkv(x)
        assert_type(qkv_out, Tensor[B, H, W, 3 * D])
        # Reshape+permute+unbind: fully tracked through the chain
        # (B, H*W, 3, NHead, HeadDim) → (3, B, NHead, H*W, HeadDim)
        # → (3, B*NHead, H*W, HeadDim) → unbind → q, k, v
        qkv_reshaped = qkv_out.reshape(b, h * w, 3, self.num_heads, self.head_dim)
        assert_type(qkv_reshaped, Tensor[B, H * W, 3, NHead, D // NHead])
        qkv_perm = qkv_reshaped.permute(2, 0, 3, 1, 4)
        assert_type(qkv_perm, Tensor[3, B, NHead, H * W, D // NHead])
        qkv_flat = qkv_perm.reshape(3, b * self.num_heads, h * w, self.head_dim)
        assert_type(qkv_flat, Tensor[3, B * NHead, H * W, D // NHead])
        q, k, v = qkv_flat.unbind(0)
        assert_type(q, Tensor[B * NHead, H * W, D // NHead])
        # Scaled dot-product attention
        attn = (q * self.scale) @ k.transpose(-2, -1)
        assert_type(attn, Tensor[B * NHead, H * W, H * W])
        # Add relative positional embeddings if enabled
        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(
                h,
                w,
                h,
                w,
                attn,
                q,
                self.rel_pos_h,
                self.rel_pos_w,
            )
        attn = F.softmax(attn, dim=-1)
        out = attn @ v
        assert_type(out, Tensor[B * NHead, H * W, D // NHead])
        # Reassemble heads: view+permute+reshape fully tracked
        result = (
            out.view(b, self.num_heads, h, w, self.head_dim)
            .permute(0, 2, 3, 1, 4)
            .reshape(b, h, w, d)
        )
        assert_type(result, Tensor[B, H, W, D])
        return self.proj(result)


# ============================================================================
# Image Encoder: ViT Block
# ============================================================================


class ViTBlock[D, NHead, MlpDim, IS, WS](nn.Module):
    """Vision Transformer block in BHWC format.

    Supports both global attention (window_size=None) and windowed attention
    (window_size=Dim[WS]). With windowed attention, the block partitions the
    spatial grid into non-overlapping windows before attention, then unpartitions.

    (B, IS, IS, D) → (B, IS, IS, D)
    """

    def __init__(
        self,
        dim: Dim[D],
        num_heads: Dim[NHead],
        mlp_dim: Dim[MlpDim],
        use_rel_pos: bool = False,
        window_size: Dim[WS] | None = None,
        input_size: Dim[IS] | None = None,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = ImageAttention(
            dim,
            num_heads,
            use_rel_pos=use_rel_pos,
            input_size=input_size,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLPBlock(dim, mlp_dim)
        self.window_size = window_size

    def forward[B](self, x: Tensor[B, IS, IS, D]) -> Tensor[B, IS, IS, D]:
        shortcut = x
        x_normed = self.norm1(x)
        if self.window_size is not None:
            b = x.shape[0]
            h_size, w_size = x.shape[1], x.shape[2]
            # Partition: (B, IS, IS, D) → (B*(IS//WS)^2, WS, WS, D)
            x_win = window_partition(x_normed, self.window_size)
            # Attention on WS-sized windows (forward accepts any H, W)
            attn_out = self.attn(x_win)
            # Unpartition: (B*(IS//WS)^2, WS, WS, D) → (B, IS, IS, D)
            attn_result = window_unpartition(
                self.window_size, h_size, w_size, b, attn_out
            )
        else:
            attn_result = self.attn(x_normed)
        # Self-attention with residual
        h = shortcut + attn_result
        assert_type(h, Tensor[B, IS, IS, D])
        # MLP with residual
        normed = self.norm2(h)
        assert_type(normed, Tensor[B, IS, IS, D])
        # MLPBlock expects (B, N, D) — reshape (B, IS, IS, D) → (B, IS*IS, D)
        b, ht, wt, d = h.size()
        assert_type(b, Dim[B])
        assert_type(ht, Dim[IS])
        assert_type(wt, Dim[IS])
        normed_flat = normed.reshape(b, ht * wt, d)
        assert_type(normed_flat, Tensor[B, IS * IS, D])
        mlp_out_flat = self.mlp(normed_flat)
        assert_type(mlp_out_flat, Tensor[B, IS * IS, D])
        # Reshape back to (B, IS, IS, D)
        mlp_out = mlp_out_flat.view(b, ht, wt, d)
        assert_type(mlp_out, Tensor[B, IS, IS, D])
        out = h + mlp_out
        assert_type(out, Tensor[B, IS, IS, D])
        return out


# ============================================================================
# Image Encoder: PatchEmbed
# ============================================================================


class PatchEmbed[EmbDim](nn.Module):
    """Patch embedding: Conv2d with kernel_size=stride=16, then permute to BHWC.

    For square input (B, 3, S, S) where S is a multiple of 16:
    Conv2d output: (B, EmbDim, (S-16)//16+1, (S-16)//16+1) → permute → BHWC
    """

    def __init__(self, embed_dim: Dim[EmbDim]) -> None:
        super().__init__()
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=16, stride=16)

    def forward[B, S](
        self, x: Tensor[B, 3, S, S]
    ) -> Tensor[B, (S - 16) // 16 + 1, (S - 16) // 16 + 1, EmbDim]:
        h = self.proj(x)
        assert_type(h, Tensor[B, EmbDim, (S - 16) // 16 + 1, (S - 16) // 16 + 1])
        # BCHW → BHWC
        out = h.permute(0, 2, 3, 1)
        assert_type(out, Tensor[B, (S - 16) // 16 + 1, (S - 16) // 16 + 1, EmbDim])
        return out


# ============================================================================
# Image Encoder: Full ViT
# ============================================================================


class ImageEncoderViT[EmbDim, OutC](nn.Module):
    """Vision Transformer image encoder with neck (square images).

    Architecture (patch_size=16):
        PatchEmbed: (B, 3, S, S) → (B, PS, PS, EmbDim) where PS=(S-16)//16+1
        N × ViTBlock: shape-preserving in BHWC (IS×IS spatial dims)
            - Some blocks use global attention (at global_attn_indexes)
            - Others use windowed attention (window_size > 0)
            - Optionally with relative positional embeddings
        Neck: permute → BCHW → Conv2d(EmbDim, OutC, 1) → LN2d
              → Conv2d(OutC, OutC, 3, pad=1) → LN2d

    (B, 3, S, S) → (B, OutC, (S-16)//16+1, (S-16)//16+1)
    """

    def __init__(
        self,
        embed_dim: Dim[EmbDim],
        out_chans: Dim[OutC],
        depth: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_rel_pos: bool = False,
        window_size: int = 0,
        global_attn_indexes: tuple[int, ...] = (),
        img_size: int = 1024,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed(embed_dim)
        mlp_dim = int(embed_dim * mlp_ratio)
        global_input_size = img_size // 16
        layers: list[ViTBlock[EmbDim, Any, Any, Any, Any]] = []
        for i in range(depth):
            # Global attention at specified indexes, windowed elsewhere.
            # For rel_pos: global blocks see (img_size//16)² patches,
            # windowed blocks see (window_size)² patches.
            is_global = i in global_attn_indexes
            block_window_size = None if is_global else window_size
            block_input_size = global_input_size if is_global else window_size
            layers.append(
                ViTBlock(
                    embed_dim,
                    num_heads,
                    mlp_dim,
                    use_rel_pos=use_rel_pos,
                    window_size=block_window_size,
                    input_size=block_input_size if use_rel_pos else None,
                )
            )
        self.blocks = nn.ModuleList(layers)
        # Neck: BHWC → BCHW → 1×1 conv + LN2d → 3×3 conv + LN2d
        self.neck_conv1 = nn.Conv2d(embed_dim, out_chans, kernel_size=1)
        self.neck_ln1 = LayerNorm2d(out_chans)
        self.neck_conv2 = nn.Conv2d(
            out_chans, out_chans, kernel_size=3, stride=1, padding=1
        )
        self.neck_ln2 = LayerNorm2d(out_chans)

    def forward[B, S](
        self, x: Tensor[B, 3, S, S]
    ) -> Tensor[B, OutC, (S - 16) // 16 + 1, (S - 16) // 16 + 1]:
        # Patch embedding: (B, 3, S, S) → (B, PS, PS, EmbDim) in BHWC
        # where PS = (S-16)//16+1
        h = self.patch_embed(x)
        assert_type(h, Tensor[B, (S - 16) // 16 + 1, (S - 16) // 16 + 1, EmbDim])
        # Transformer blocks (shape-preserving in BHWC).
        # Each block's forward uses IS (class param) for spatial dims.
        # ModuleList stores blocks as ViTBlock[EmbDim, Any, Any, Any],
        # erasing IS to Any. Re-annotate after the loop.
        for blk in self.blocks:
            h = blk(h)
        h_blocked: Tensor[B, (S - 16) // 16 + 1, (S - 16) // 16 + 1, EmbDim] = h  # type: ignore[bad-assignment]
        # Neck: permute to BCHW
        h_bchw = h_blocked.permute(0, 3, 1, 2)
        assert_type(h_bchw, Tensor[B, EmbDim, (S - 16) // 16 + 1, (S - 16) // 16 + 1])
        n1 = self.neck_ln1(self.neck_conv1(h_bchw))
        assert_type(n1, Tensor[B, OutC, (S - 16) // 16 + 1, (S - 16) // 16 + 1])
        n2 = self.neck_ln2(self.neck_conv2(n1))
        assert_type(n2, Tensor[B, OutC, (S - 16) // 16 + 1, (S - 16) // 16 + 1])
        return n2


# ============================================================================
# Two-Way Transformer: Cross-Attention
# ============================================================================


class CrossAttention[D, IntDim, NHead](nn.Module):
    """Cross-attention with optional dimension downsampling.

    Separate Q/K/V projections: D → IntDim. Multi-head attention with
    different sequence lengths for queries and keys.

    q (B, NQ, D), k (B, NK, D), v (B, NK, D) → (B, NQ, D)
    """

    def __init__(
        self,
        embedding_dim: Dim[D],
        internal_dim: Dim[IntDim],
        num_heads: Dim[NHead],
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = internal_dim // num_heads
        self.scale: float = self.head_dim**-0.5
        self.q_proj = nn.Linear(embedding_dim, internal_dim)
        self.k_proj = nn.Linear(embedding_dim, internal_dim)
        self.v_proj = nn.Linear(embedding_dim, internal_dim)
        self.out_proj = nn.Linear(internal_dim, embedding_dim)

    def forward[B, NQ, NK](
        self, q: Tensor[B, NQ, D], k: Tensor[B, NK, D], v: Tensor[B, NK, D]
    ) -> Tensor[B, NQ, D]:
        b, nq, _ = q.size()
        assert_type(b, Dim[B])
        assert_type(nq, Dim[NQ])
        # Project
        q_proj = self.q_proj(q)
        assert_type(q_proj, Tensor[B, NQ, IntDim])
        k_proj = self.k_proj(k)
        assert_type(k_proj, Tensor[B, NK, IntDim])
        v_proj = self.v_proj(v)
        assert_type(v_proj, Tensor[B, NK, IntDim])
        # Separate heads: (B, N, IntDim) → (B, N, NHead, HeadDim) → (B, NHead, N, HeadDim)
        q_heads = q_proj.reshape(b, nq, self.num_heads, self.head_dim).transpose(1, 2)
        assert_type(q_heads, Tensor[B, NHead, NQ, IntDim // NHead])
        nk = k.size(1)
        assert_type(nk, Dim[NK])
        k_heads = k_proj.reshape(b, nk, self.num_heads, self.head_dim).transpose(1, 2)
        assert_type(k_heads, Tensor[B, NHead, NK, IntDim // NHead])
        v_heads = v_proj.reshape(b, nk, self.num_heads, self.head_dim).transpose(1, 2)
        assert_type(v_heads, Tensor[B, NHead, NK, IntDim // NHead])
        # Attention: (B, NHead, NQ, HeadDim) @ (B, NHead, HeadDim, NK) → (B, NHead, NQ, NK)
        attn = torch.matmul(q_heads * self.scale, k_heads.transpose(2, 3))
        assert_type(attn, Tensor[B, NHead, NQ, NK])
        attn = F.softmax(attn.float(), dim=-1)
        assert_type(attn, Tensor[B, NHead, NQ, NK])
        # (B, NHead, NQ, NK) @ (B, NHead, NK, HeadDim) → (B, NHead, NQ, HeadDim)
        out = torch.matmul(attn, v_heads)
        assert_type(out, Tensor[B, NHead, NQ, IntDim // NHead])
        # Recombine: (B, NHead, NQ, HeadDim) → transpose → (B, NQ, NHead, HeadDim)
        # → reshape → (B, NQ, IntDim)
        out_t = out.transpose(1, 2).contiguous()
        # reshape merges last two dims; NHead * (IntDim // NHead) = IntDim can't be proven
        out_flat: Tensor[B, NQ, IntDim] = out_t.reshape(  # type: ignore[bad-assignment]
            b, nq, self.num_heads * self.head_dim
        )
        return self.out_proj(out_flat)


# ============================================================================
# Two-Way Transformer: Attention Block
# ============================================================================


class TwoWayAttentionBlock[D, NHead, MlpDim](nn.Module):
    """Bidirectional cross-attention block.

    Four operations:
    1. Self-attention on queries
    2. Cross-attention: queries attend to keys (image)
    3. MLP on queries
    4. Cross-attention: keys attend to queries

    (B, NQ, D), (B, NK, D) → (B, NQ, D), (B, NK, D)
    """

    def __init__(
        self,
        embedding_dim: Dim[D],
        num_heads: Dim[NHead],
        mlp_dim: Dim[MlpDim],
        downsample_rate: int = 2,
    ) -> None:
        super().__init__()
        internal_dim = embedding_dim // downsample_rate
        self.self_attn = CrossAttention(embedding_dim, embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.cross_attn_token_to_image = CrossAttention(
            embedding_dim, internal_dim, num_heads
        )
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.mlp = MLPBlock(embedding_dim, mlp_dim)
        self.norm3 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = CrossAttention(
            embedding_dim, internal_dim, num_heads
        )
        self.norm4 = nn.LayerNorm(embedding_dim)

    def forward[B, NQ, NK](
        self,
        queries: Tensor[B, NQ, D],
        keys: Tensor[B, NK, D],
        query_pe: Tensor[B, NQ, D],
        key_pe: Tensor[B, NK, D],
    ) -> tuple[Tensor[B, NQ, D], Tensor[B, NK, D]]:
        # 1. Self-attention on queries
        q_pe = queries + query_pe
        attn_out = self.self_attn(q_pe, q_pe, queries)
        assert_type(attn_out, Tensor[B, NQ, D])
        queries = queries + attn_out
        queries = self.norm1(queries)
        assert_type(queries, Tensor[B, NQ, D])
        # 2. Cross-attention: queries → keys
        q = queries + query_pe
        k = keys + key_pe
        attn_out2 = self.cross_attn_token_to_image(q, k, keys)
        assert_type(attn_out2, Tensor[B, NQ, D])
        queries = queries + attn_out2
        queries = self.norm2(queries)
        assert_type(queries, Tensor[B, NQ, D])
        # 3. MLP on queries
        mlp_out = self.mlp(queries)
        assert_type(mlp_out, Tensor[B, NQ, D])
        queries = queries + mlp_out
        queries = self.norm3(queries)
        assert_type(queries, Tensor[B, NQ, D])
        # 4. Cross-attention: keys → queries
        q4 = queries + query_pe
        k4 = keys + key_pe
        attn_out3 = self.cross_attn_image_to_token(k4, q4, queries)
        assert_type(attn_out3, Tensor[B, NK, D])
        keys = keys + attn_out3
        keys = self.norm4(keys)
        assert_type(keys, Tensor[B, NK, D])
        return queries, keys


# ============================================================================
# Two-Way Transformer
# ============================================================================


class TwoWayTransformer[D, NHead, MlpDim](nn.Module):
    """Two-way transformer decoder with bidirectional cross-attention.

    Used in SAM's mask decoder: image features and prompt tokens attend to
    each other through stacked TwoWayAttentionBlocks.

    image (B, D, IH, IW), queries (B, NQ, D)
        → flatten image to (B, IH*IW, D)
        → N × TwoWayAttentionBlock
        → final cross-attention: queries → image
        → queries (B, NQ, D), image_tokens (B, IH*IW, D)

    Note: IH*IW is computed at runtime, so the flattened sequence length
    is a fresh type variable NK.
    """

    def __init__(
        self,
        embedding_dim: Dim[D],
        num_heads: Dim[NHead],
        mlp_dim: Dim[MlpDim],
        depth: int,
        downsample_rate: int = 2,
    ) -> None:
        super().__init__()
        internal_dim = embedding_dim // downsample_rate
        layers: list[TwoWayAttentionBlock[D, NHead, MlpDim]] = []
        for _ in range(depth):
            layers.append(
                TwoWayAttentionBlock(embedding_dim, num_heads, mlp_dim, downsample_rate)
            )
        self.layers = nn.ModuleList(layers)
        self.final_attn = CrossAttention(embedding_dim, internal_dim, num_heads)
        self.norm_final = nn.LayerNorm(embedding_dim)

    def forward[B, NQ, NK](
        self,
        image_tokens: Tensor[B, NK, D],
        image_pe: Tensor[B, NK, D],
        queries: Tensor[B, NQ, D],
    ) -> tuple[Tensor[B, NQ, D], Tensor[B, NK, D]]:
        # Apply two-way attention blocks
        keys = image_tokens
        for layer in self.layers:
            queries, keys = layer(queries, keys, queries, image_pe)
        assert_type(queries, Tensor[B, NQ, D])
        assert_type(keys, Tensor[B, NK, D])
        # Final cross-attention: queries attend to image
        q = queries + queries  # query + query_pe (using queries as its own PE here)
        k = keys + image_pe
        final_out = self.final_attn(q, k, keys)
        assert_type(final_out, Tensor[B, NQ, D])
        queries = queries + final_out
        queries = self.norm_final(queries)
        assert_type(queries, Tensor[B, NQ, D])
        return queries, keys


# ============================================================================
# Positional Embedding (Random Fourier Features)
# ============================================================================


class PositionalEmbeddingRandom[D](nn.Module):
    """Positional encoding using random spatial frequencies.

    Original: sam/prompt_encoder.py PositionEmbeddingRandom.

    Maps 2D coordinates to positional encodings via Gaussian random projection.
    The projection matrix is a registered buffer (not a learned parameter).

    (..., 2) → (..., 2*D) via sin/cos encoding
    """

    def __init__(self, num_pos_feats: Dim[D]) -> None:
        super().__init__()
        self.positional_encoding_gaussian_matrix: Tensor[2, D] = nn.Buffer(
            torch.randn(2, num_pos_feats), persistent=False
        )

    def _pe_encoding[*Batch](self, coords: Tensor[*Batch, 2]) -> Tensor[*Batch, 2 * D]:
        """Encode coordinates to positional features.

        coords: (*Batch, 2) normalized to [0, 1]
        Returns: (*Batch, 2*D) — sin/cos of projected coordinates.
        coords @ buffer(2, D) → (*Batch, D), then cat(sin, cos) → (*Batch, 2*D).
        """
        coords = 2 * coords - 1
        projected = coords @ self.positional_encoding_gaussian_matrix
        projected = 2 * math.pi * projected
        return torch.cat([torch.sin(projected), torch.cos(projected)], dim=-1)

    def forward(self, size: tuple[int, int]) -> Tensor:
        """Generate positional encoding grid.

        size: (H, W) spatial dimensions
        Returns: (1, 2*D, H, W) — unrefined due to buffer ops and permute.
        """
        h, w = size
        grid = torch.ones((h, w), dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w
        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1).unsqueeze(0)


# ============================================================================
# Prompt Encoder
# ============================================================================


class PromptEncoder[D, ES, MIC](nn.Module):
    """Encodes prompts (points, boxes, masks) for the mask decoder.

    Original: sam/prompt_encoder.py PromptEncoder.

    Multi-modal input handling:
    - Points (B, N, 2) coords + (B, N) labels: positional encoding + type embed
    - Boxes (B, M, 4): encoded as two corner points (top-left + bottom-right)
    - Masks (B, 1, H, W): Conv2d downsampling to image embedding resolution

    Returns:
    - sparse_embeddings: (B, NTokens, D) — NTokens data-dependent (G1)
    - dense_embeddings: (B, D, ES, ES) — from mask conv chain or learned no-mask embed

    Shapeless operations: _pe_encoding matmul (C3), data-dependent label masking (G1),
    Embedding.weight (C1). Channel dims tracked through mask Conv2d chain.
    """

    def __init__(
        self,
        embed_dim: Dim[D],
        emb_size: Dim[ES],
        input_image_size: tuple[int, int],
        mask_in_chans: Dim[MIC],
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.image_embedding_size = (emb_size, emb_size)
        self.input_image_size = input_image_size
        self.pe_layer = PositionalEmbeddingRandom(embed_dim // 2)
        # 4 point types: foreground, background, top-left corner, bottom-right corner
        self.point_embeddings = nn.ModuleList(
            [nn.Embedding(1, embed_dim) for _ in range(4)]
        )
        self.not_a_point_embed = nn.Embedding(1, embed_dim)
        # Mask downsampling: (B, 1, H, W) → (B, D, EmbH, EmbW)
        # Two stride-2 convs (4× downsample) + 1×1 conv to embed_dim
        self.mask_downscaling_conv1 = nn.Conv2d(
            1, mask_in_chans // 4, kernel_size=2, stride=2
        )
        self.mask_downscaling_ln1 = LayerNorm2d(mask_in_chans // 4)
        self.mask_downscaling_conv2 = nn.Conv2d(
            mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2
        )
        self.mask_downscaling_ln2 = LayerNorm2d(mask_in_chans)
        self.mask_downscaling_conv3 = nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1)
        self.no_mask_embed = nn.Embedding(1, embed_dim)

    def get_dense_pe(self) -> Tensor:
        """Get positional encoding for image embedding grid.

        Returns: (1, D, EmbH, EmbW) — unrefined (buffer ops).
        """
        return self.pe_layer(self.image_embedding_size)

    def _embed_points(self, points: Tensor, labels: Tensor, pad: bool) -> Tensor:
        """Embed point prompts.

        points: (B, N, 2) — point coordinates
        labels: (B, N) — point labels (0=bg, 1=fg)
        pad: whether to add a padding point (when no box prompt)

        Returns: bare Tensor — data-dependent token count (N' varies with padding),
        data-dependent label masking selects type embeddings.
        """
        points = points + 0.5
        if pad:
            # Padding changes token count (data-dependent: N → N+1)
            padding_point = torch.zeros((points.shape[0], 1, 2))
            padding_label: Tensor = -torch.ones((labels.shape[0], 1))
            points = torch.cat([points, padding_point], dim=1)
            labels = torch.cat([labels, padding_label], dim=1)
        # Positional encoding: coords @ buffer(2, D) → (..., D) → cat(sin,cos) → (..., 2*D)
        # Division by input_image_size normalizes to [0,1]; torch.tensor([...]) needs annotation
        scale: Tensor[2] = torch.tensor(list(reversed(self.input_image_size))).float()
        point_embedding = self.pe_layer._pe_encoding(points / scale)
        # Add type embeddings: labels 0/1 map to point_embeddings[0/1]
        # Data-dependent masking: (labels == 1) selects which points get which embedding
        mask_fg: Tensor = (labels == 1).unsqueeze(-1)
        mask_bg: Tensor = (labels == 0).unsqueeze(-1)
        point_embedding = (
            point_embedding
            + mask_fg * self.point_embeddings[0].weight
            + mask_bg * self.point_embeddings[1].weight
        )
        # Padding points get not_a_point_embed
        mask_pad: Tensor = (labels == -1).unsqueeze(-1)
        point_embedding = point_embedding + mask_pad * self.not_a_point_embed.weight
        return point_embedding

    def _embed_boxes(self, boxes: Tensor) -> Tensor:
        """Embed box prompts as corner point pairs.

        boxes: (B, M, 4) — box coordinates [x1, y1, x2, y2]
        Returns: bare Tensor. reshape(-1, 2, 2) merges B*M into a single dim
        (data-dependent batch), then _pe_encoding tracks but += on slices
        loses shapes. Data-dependent final shape (B, 2*M, D).
        """
        boxes = boxes + 0.5
        coords = boxes.reshape(-1, 2, 2)
        scale: Tensor[2] = torch.tensor(list(reversed(self.input_image_size))).float()
        corner_embedding: Tensor = self.pe_layer._pe_encoding(coords / scale)
        # Add corner type embeddings (top-left=idx 2, bottom-right=idx 3)
        corner_embedding[:, 0, :] += self.point_embeddings[2].weight
        corner_embedding[:, 1, :] += self.point_embeddings[3].weight
        return corner_embedding

    def _embed_masks[B](
        self, masks: Tensor[B, 1, 4 * ES, 4 * ES]
    ) -> Tensor[B, D, ES, ES]:
        """Downsample mask prompts to image embedding resolution.

        masks: (B, 1, 4*ES, 4*ES) — input at 4× embedding resolution
        Returns: (B, D, ES, ES). Conv2d chain: two stride-2 convs (4× downsample)
        + 1×1 conv. Channels: 1 → MIC//4 → MIC → D. Spatial: 4*ES → 2*ES → ES.
        """
        h = self.mask_downscaling_conv1(masks)
        h = self.mask_downscaling_ln1(h)
        h = F.gelu(h)
        h = self.mask_downscaling_conv2(h)
        h = self.mask_downscaling_ln2(h)
        h = F.gelu(h)
        return self.mask_downscaling_conv3(h)

    def forward[B, N, M](
        self,
        points: tuple[Tensor[B, N, 2], Tensor[B, N]] | None,
        boxes: Tensor[B, M, 4] | None,
        masks: Tensor[B, 1, 4 * ES, 4 * ES] | None,
    ) -> tuple[Tensor, Tensor[B, D, ES, ES]]:
        """Encode all prompt types.

        Args:
            points: (coords (B, N, 2), labels (B, N)) or None
            boxes: (B, M, 4) or None
            masks: (B, 1, 4*ES, 4*ES) or None

        Returns:
            sparse_embeddings: (B, NTokens, D) — bare Tensor (data-dependent token count)
            dense_embeddings: Tensor[B, D, ES, ES]
        """
        bs = self._get_batch_size(points, boxes, masks)
        sparse_embeddings: Tensor = torch.empty(
            (bs, 0, self.embed_dim), dtype=torch.float32
        )
        if points is not None:
            coords, labels = points
            point_embeddings = self._embed_points(coords, labels, pad=(boxes is None))
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)
        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes)
            sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)
        if masks is not None:
            dense_embeddings = self._embed_masks(masks)
        else:
            emb_h, emb_w = self.image_embedding_size
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs, -1, emb_h, emb_w
            )
        return sparse_embeddings, dense_embeddings

    def _get_batch_size[B, N, M](
        self,
        points: tuple[Tensor[B, N, 2], Tensor[B, N]] | None,
        boxes: Tensor[B, M, 4] | None,
        masks: Tensor[B, 1, 4 * ES, 4 * ES] | None,
    ) -> Dim[B]:
        """Infer batch size from whichever prompt is provided."""
        if points is not None:
            return points[0].shape[0]
        elif boxes is not None:
            return boxes.shape[0]
        elif masks is not None:
            return masks.shape[0]
        else:
            raise ValueError("At least one prompt must be provided")


# ============================================================================
# Hypernetwork MLP
# ============================================================================


class HypernetworkMLP[In, Hidden, Out](nn.Module):
    """3-layer MLP used as hypernetwork in mask decoder and for IoU prediction.

    Original: sam/common.py MLP class (with num_layers=3).

    Linear(In, Hidden) → ReLU → Linear(Hidden, Hidden) → ReLU → Linear(Hidden, Out)

    (*, In) → (*, Out)
    """

    def __init__(
        self,
        input_dim: Dim[In],
        hidden_dim: Dim[Hidden],
        output_dim: Dim[Out],
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid_output = sigmoid_output

    def forward[B](self, x: Tensor[B, In]) -> Tensor[B, Out]:
        """Forward pass: Linear→ReLU→Linear→ReLU→Linear.

        Typed like BottomMLP/TopMLP. When called with unrefined input,
        the NNModule DSL still preserves the output dim from fc3.
        """
        h = F.relu(self.fc1(x))
        assert_type(h, Tensor[B, Hidden])
        h = F.relu(self.fc2(h))
        assert_type(h, Tensor[B, Hidden])
        out = self.fc3(h)
        assert_type(out, Tensor[B, Out])
        if self.sigmoid_output:
            out = torch.sigmoid(out)
        return out


# ============================================================================
# Mask Decoder
# ============================================================================


class MaskDecoder[D, NHead, MlpDim, NumMasks](nn.Module):
    """Predicts masks and IoU scores from image and prompt embeddings.

    Original: sam/mask_decoder.py MaskDecoder.

    Architecture:
    1. Concatenate IoU token + mask tokens + sparse prompt → combined tokens
    2. Add dense prompt to image embeddings, flatten to sequence
    3. Run TwoWayTransformer on (image_tokens, image_pe, combined_tokens)
    4. Upscale image: ConvTranspose2d×2 for 4× spatial upsampling
    5. Hypernetwork: per-mask MLP on output tokens → mask weights
    6. Dot product of upscaled features with mask weights → mask predictions
    7. IoU head: MLP on IoU token output → per-mask IoU scores

    Heavy dynamic reshaping throughout — uses unrefined Tensor and annotation
    fallbacks extensively.
    """

    def __init__(
        self,
        transformer_dim: Dim[D],
        num_heads: Dim[NHead],
        mlp_dim: Dim[MlpDim],
        num_multimask_outputs: Dim[NumMasks],
        transformer_depth: int = 2,
    ) -> None:
        super().__init__()
        self.num_multimask_outputs = num_multimask_outputs
        num_mask_tokens = num_multimask_outputs + 1
        # Transformer
        self.transformer = TwoWayTransformer(
            transformer_dim, num_heads, mlp_dim, depth=transformer_depth
        )
        # Learnable tokens
        self.iou_token = nn.Embedding(1, transformer_dim)
        self.mask_tokens = nn.Embedding(num_mask_tokens, transformer_dim)
        # Upscaling: (B, D, H, W) → (B, D//4, 2H, 2W) → (B, D//8, 4H, 4W)
        self.output_upscaling_conv1 = nn.ConvTranspose2d(
            transformer_dim, transformer_dim // 4, kernel_size=2, stride=2
        )
        self.output_upscaling_ln = LayerNorm2d(transformer_dim // 4)
        self.output_upscaling_conv2 = nn.ConvTranspose2d(
            transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2
        )
        # Hypernetwork MLPs: one per mask token
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                HypernetworkMLP(transformer_dim, transformer_dim, transformer_dim // 8)
                for _ in range(num_mask_tokens)
            ]
        )
        # IoU prediction head
        self.iou_prediction_head = HypernetworkMLP(
            transformer_dim, 256, num_mask_tokens
        )

    def forward[B, IH, IW](
        self,
        image_embeddings: Tensor[B, D, IH, IW],
        image_pe: Tensor,  # batch dim may be 1; keep unrefined
        sparse_prompt_embeddings: Tensor,  # data-dependent token count
        dense_prompt_embeddings: Tensor[B, D, IH, IW],
        multimask_output: bool,
    ) -> tuple[Tensor, Tensor]:
        """Predict masks and IoU scores.

        Args:
            image_embeddings: (B, D, IH, IW) — from image encoder
            image_pe: (1, D, IH, IW) — positional encoding for image
            sparse_prompt_embeddings: (B, NTokens, D) — from prompt encoder
            dense_prompt_embeddings: (B, D, IH, IW) — from prompt encoder
            multimask_output: if True return all masks, else just the best one

        Returns:
            masks: (B, NumMasks+1, 4*IH, 4*IW) or (B, 1, 4*IH, 4*IW) — unrefined
            iou_pred: (B, NumMasks+1) or (B, 1) — unrefined
        """
        masks, iou_pred = self.predict_masks(
            image_embeddings,
            image_pe,
            sparse_prompt_embeddings,
            dense_prompt_embeddings,
        )
        # Select masks based on multimask_output
        if multimask_output:
            return masks[:, 1:, :, :], iou_pred[:, 1:]
        else:
            return masks[:, 0:1, :, :], iou_pred[:, 0:1]

    def predict_masks[B, IH, IW](
        self,
        image_embeddings: Tensor[B, D, IH, IW],
        image_pe: Tensor,
        sparse_prompt_embeddings: Tensor,
        dense_prompt_embeddings: Tensor[B, D, IH, IW],
    ) -> tuple[Tensor, Tensor]:
        """Run transformer and generate all mask predictions.

        Token prep is shapeless (Embedding.weight unrefined, cat, expand).
        Image-side params are typed; flatten(2).transpose(1,2) and ConvTranspose2d
        chain are fully shape-tracked. Shapes are lost at transformer output
        (receives shapeless tokens). Final mask generation shapeless
        (hypernetwork token slicing + matmul + view).
        """
        # Concatenate output tokens: [iou_token, mask_tokens] + sparse_prompt
        # Shapeless: Embedding.weight returns unrefined Tensor
        output_tokens: Tensor = torch.cat(
            [self.iou_token.weight, self.mask_tokens.weight], dim=0
        )
        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(0), -1, -1
        )
        tokens: Tensor = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)
        # Prepare image tokens: add dense prompt, flatten to sequence
        src = image_embeddings + dense_prompt_embeddings
        assert_type(src, Tensor[B, D, IH, IW])
        b, c, h, w = src.shape
        src_flat = src.flatten(2).transpose(1, 2)
        assert_type(src_flat, Tensor[B, IH * IW, D])
        pos_src: Tensor = image_pe.flatten(2).transpose(1, 2)
        # Expand pos encoding if needed
        if pos_src.shape[0] != b:
            pos_src = pos_src.expand(b, -1, -1)
        # Run two-way transformer
        hs, src_out = self.transformer(src_flat, pos_src, tokens)
        # src_out should be Tensor[B, IH*IW, D] if transformer tracks it
        # (src_flat is typed, but pos_src and tokens are bare)
        iou_token_out: Tensor = hs[:, 0, :]
        num_mask_tokens = self.num_multimask_outputs + 1
        mask_tokens_out: Tensor = hs[:, 1 : 1 + num_mask_tokens, :]
        # Reshape back to spatial
        src_spatial = src_out.transpose(1, 2).view(b, c, h, w)
        assert_type(src_spatial, Tensor[B, D, IH, IW])
        # Upscale: ConvTranspose2d(D, D//4, k=2, s=2) → LN → GELU
        upscaled = self.output_upscaling_conv1(src_spatial)
        assert_type(upscaled, Tensor[B, D // 4, 2 * IH, 2 * IW])
        upscaled = self.output_upscaling_ln(upscaled)
        upscaled = F.gelu(upscaled)
        # ConvTranspose2d(D//4, D//8, k=2, s=2) → GELU
        upscaled = self.output_upscaling_conv2(upscaled)
        assert_type(upscaled, Tensor[B, D // 8, 4 * IH, 4 * IW])
        upscaled = F.gelu(upscaled)
        # Generate masks via hypernetwork MLPs
        # mask_tokens_out[:, i, :] is unrefined slice; HypernetworkMLP output
        # tracks D//8 channel dim. But torch.stack + @ + view → shapeless.
        hyper_in_list: list[Tensor] = []
        for i in range(num_mask_tokens):
            hyper_in_list.append(
                self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
            )
        hyper_in: Tensor = torch.stack(hyper_in_list, dim=1)
        b_up, c_up, h_up, w_up = upscaled.shape
        masks: Tensor = (hyper_in @ upscaled.view(b_up, c_up, h_up * w_up)).view(
            b_up, -1, h_up, w_up
        )
        # IoU prediction
        iou_pred: Tensor = self.iou_prediction_head(iou_token_out)
        return masks, iou_pred


# ============================================================================
# Sam (Top-Level Model)
# ============================================================================


class Sam[EmbDim, D, NHead, MlpDim, NumMasks, ES](nn.Module):
    """Segment Anything Model — top-level orchestration.

    Original: sam/build_sam.py + sam/modeling/sam.py Sam class.

    Wraps ImageEncoderViT, PromptEncoder, and MaskDecoder.
    D is the decoder/transformer dimension, shared across all three components
    (image encoder output channels = prompt encoder embed dim = mask decoder dim).
    ES is the image embedding spatial size (= img_size // patch_size for square).
    """

    def __init__(
        self,
        image_encoder: ImageEncoderViT[EmbDim, D],
        prompt_encoder: PromptEncoder[D, ES, Any],
        mask_decoder: MaskDecoder[D, NHead, MlpDim, NumMasks],
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder

    def forward[B, S, N, M](
        self,
        images: Tensor[B, 3, S, S],
        points: tuple[Tensor[B, N, 2], Tensor[B, N]] | None = None,
        boxes: Tensor[B, M, 4] | None = None,
        masks: Tensor[B, 1, 4 * ES, 4 * ES] | None = None,
        multimask_output: bool = True,
    ) -> tuple[Tensor, Tensor]:
        """Run SAM end-to-end.

        Args:
            images: (B, 3, S, S) — input images (square)
            points: (coords (B, N, 2), labels (B, N)) or None
            boxes: (B, M, 4) or None
            masks: (B, 1, 4*ES, 4*ES) or None
            multimask_output: whether to return multiple masks

        Returns:
            pred_masks: predicted segmentation masks — unrefined
            iou_predictions: predicted IoU scores — unrefined

        Image encoder produces typed output (Tensor[B, D, PH, PH]).
        Prompt encoder dense output typed (Tensor[B, D, ES, ES]).
        Mask decoder types image-side params. Token-side and final mask
        generation remain shapeless.
        """
        image_embeddings = self.image_encoder(images)
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=points, boxes=boxes, masks=masks
        )
        image_pe = self.prompt_encoder.get_dense_pe()
        pred_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )
        return pred_masks, iou_predictions


# ============================================================================
# Smoke tests
# ============================================================================


def test_layer_norm_2d():
    """Test LayerNorm2d: shape-preserving on (B, C, H, W)."""
    ln = LayerNorm2d(64)
    x: Tensor[2, 64, 16, 16] = torch.randn(2, 64, 16, 16)
    out = ln(x)
    assert_type(out, Tensor[2, 64, 16, 16])


def test_mlp_block():
    """Test MLPBlock: (B, N, D) → (B, N, D)."""
    mlp = MLPBlock(192, 768)
    x: Tensor[2, 256, 192] = torch.randn(2, 256, 192)
    out = mlp(x)
    assert_type(out, Tensor[2, 256, 192])


def test_patch_embed():
    """Test PatchEmbed: (B, 3, 256, 256) → (B, 16, 16, 192).
    Conv2d output: (256-16)//16+1 = 16.
    """
    pe = PatchEmbed(192)
    x: Tensor[1, 3, 256, 256] = torch.randn(1, 3, 256, 256)
    out = pe(x)
    # (256-16)//16+1 = 240//16+1 = 15+1 = 16
    assert_type(out, Tensor[1, (256 - 16) // 16 + 1, (256 - 16) // 16 + 1, 192])


def test_image_attention():
    """Test ImageAttention: (B, H, W, D) → (B, H, W, D)."""
    attn = ImageAttention(192, 4)
    x: Tensor[1, 16, 16, 192] = torch.randn(1, 16, 16, 192)
    out = attn(x)
    assert_type(out, Tensor[1, 16, 16, 192])


def test_image_attention_rel_pos():
    """Test ImageAttention with relative positional embeddings."""
    attn = ImageAttention(192, 4, use_rel_pos=True, input_size=16)
    x: Tensor[1, 16, 16, 192] = torch.randn(1, 16, 16, 192)
    out = attn(x)
    assert_type(out, Tensor[1, 16, 16, 192])


def test_vit_block():
    """Test ViTBlock: (B, H, W, D) → (B, H, W, D)."""
    block = ViTBlock(192, 4, 768)
    x: Tensor[1, 16, 16, 192] = torch.randn(1, 16, 16, 192)
    out = block(x)
    assert_type(out, Tensor[1, 16, 16, 192])


def test_image_encoder():
    """Test ImageEncoderViT: (B, 3, 256, 256) → (B, 64, 16, 16)."""
    enc = ImageEncoderViT(192, 64, depth=4, num_heads=4)
    x: Tensor[1, 3, 256, 256] = torch.randn(1, 3, 256, 256)
    out = enc(x)
    assert_type(out, Tensor[1, 64, (256 - 16) // 16 + 1, (256 - 16) // 16 + 1])


def test_image_encoder_rel_pos():
    """Test ImageEncoderViT with relative positional embeddings.
    Some blocks use global attention, others windowed.
    """
    enc = ImageEncoderViT(
        192,
        64,
        depth=4,
        num_heads=4,
        use_rel_pos=True,
        window_size=8,
        global_attn_indexes=(1, 3),
    )
    x: Tensor[1, 3, 256, 256] = torch.randn(1, 3, 256, 256)
    out = enc(x)
    assert_type(out, Tensor[1, 64, (256 - 16) // 16 + 1, (256 - 16) // 16 + 1])


def test_cross_attention():
    """Test CrossAttention: different sequence lengths for Q and K."""
    attn = CrossAttention(256, 128, 8)
    q: Tensor[1, 5, 256] = torch.randn(1, 5, 256)
    k: Tensor[1, 64, 256] = torch.randn(1, 64, 256)
    v: Tensor[1, 64, 256] = torch.randn(1, 64, 256)
    out = attn(q, k, v)
    assert_type(out, Tensor[1, 5, 256])


def test_two_way_attention_block():
    """Test TwoWayAttentionBlock: bidirectional cross-attention."""
    block = TwoWayAttentionBlock(256, 8, 2048)
    queries: Tensor[1, 5, 256] = torch.randn(1, 5, 256)
    keys: Tensor[1, 64, 256] = torch.randn(1, 64, 256)
    q_pe: Tensor[1, 5, 256] = torch.randn(1, 5, 256)
    k_pe: Tensor[1, 64, 256] = torch.randn(1, 64, 256)
    out_q, out_k = block(queries, keys, q_pe, k_pe)
    assert_type(out_q, Tensor[1, 5, 256])
    assert_type(out_k, Tensor[1, 64, 256])


def test_two_way_transformer():
    """Test TwoWayTransformer: stacked bidirectional attention."""
    transformer = TwoWayTransformer(256, 8, 2048, depth=2)
    image_tokens: Tensor[1, 64, 256] = torch.randn(1, 64, 256)
    image_pe: Tensor[1, 64, 256] = torch.randn(1, 64, 256)
    queries: Tensor[1, 5, 256] = torch.randn(1, 5, 256)
    out_q, out_k = transformer(image_tokens, image_pe, queries)
    assert_type(out_q, Tensor[1, 5, 256])
    assert_type(out_k, Tensor[1, 64, 256])


def test_prompt_encoder():
    """Test PromptEncoder with point prompts."""
    pe = PromptEncoder(256, 16, input_image_size=(256, 256), mask_in_chans=16)
    # Point prompts: 2 points
    coords: Tensor[1, 2, 2] = torch.rand(1, 2, 2)
    labels: Tensor[1, 2] = torch.tensor([[1, 0]])
    sparse, dense = pe(points=(coords, labels), boxes=None, masks=None)
    # sparse: (1, N, 256) — N is data-dependent
    assert_type(dense, Tensor[1, 256, 16, 16])


def test_prompt_encoder_with_mask():
    """Test PromptEncoder with mask prompt."""
    pe = PromptEncoder(256, 16, input_image_size=(256, 256), mask_in_chans=16)
    # 4*ES = 4*16 = 64
    mask: Tensor[1, 1, 64, 64] = torch.randn(1, 1, 64, 64)
    sparse, dense = pe(points=None, boxes=None, masks=mask)
    assert_type(dense, Tensor[1, 256, 16, 16])


def test_hypernetwork_mlp():
    """Test HypernetworkMLP: (B, In) → (B, Out)."""
    mlp = HypernetworkMLP(256, 256, 32)
    x: Tensor[2, 256] = torch.randn(2, 256)
    out = mlp(x)
    assert_type(out, Tensor[2, 32])


def test_hypernetwork_mlp_sigmoid():
    """Test HypernetworkMLP with sigmoid output."""
    mlp = HypernetworkMLP(128, 64, 4, sigmoid_output=True)
    x: Tensor[1, 128] = torch.randn(1, 128)
    out = mlp(x)
    assert_type(out, Tensor[1, 4])


def test_mask_decoder():
    """Test MaskDecoder with pre-computed embeddings."""
    decoder = MaskDecoder(256, 8, 2048, 3, transformer_depth=2)
    image_emb: Tensor[1, 256, 16, 16] = torch.randn(1, 256, 16, 16)
    image_pe: Tensor[1, 256, 16, 16] = torch.randn(1, 256, 16, 16)
    sparse_prompt: Tensor[1, 2, 256] = torch.randn(1, 2, 256)
    dense_prompt: Tensor[1, 256, 16, 16] = torch.randn(1, 256, 16, 16)
    masks, iou_pred = decoder(image_emb, image_pe, sparse_prompt, dense_prompt, True)


def test_sam_end_to_end():
    """Test Sam top-level model with point prompts."""
    image_encoder = ImageEncoderViT(192, 256, depth=2, num_heads=4)
    prompt_encoder = PromptEncoder(
        256, 16, input_image_size=(256, 256), mask_in_chans=16
    )
    mask_decoder = MaskDecoder(256, 8, 2048, 3, transformer_depth=2)
    model = Sam(image_encoder, prompt_encoder, mask_decoder)
    images: Tensor[1, 3, 256, 256] = torch.randn(1, 3, 256, 256)
    coords: Tensor[1, 1, 2] = torch.rand(1, 1, 2)
    labels: Tensor[1, 1] = torch.tensor([[1]])
    pred_masks, iou_predictions = model(images, points=(coords, labels))
