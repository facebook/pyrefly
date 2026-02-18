# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import math
from dataclasses import dataclass
from typing import Any, assert_type, Optional, TYPE_CHECKING, TypedDict

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.attention.flex_attention import (
    _mask_mod_signature,
    BlockMask,
    flex_attention,
)


if TYPE_CHECKING:
    from torch import Tensor
    from torch_shapes import Dim


class RopeScalingDict(TypedDict, total=False):
    """Type for rope_scaling configuration."""

    factor: float
    low_freq_factor: float
    high_freq_factor: float
    original_max_position_embeddings: int


class ModelArgsDict[
    VocabSize,
    BlockSize,
    D,
    NHead,
    NLayer,
    IntermediateSize,
    NLocalHeads,
](TypedDict, total=False):
    """Type for transformer configuration dictionaries."""

    block_size: Dim[BlockSize]
    vocab_size: Dim[VocabSize]
    n_layer: Dim[NLayer]
    n_head: Dim[NHead]
    dim: Dim[D]
    intermediate_size: Dim[IntermediateSize]
    n_local_heads: Dim[NLocalHeads]
    head_dim: Dim[D // NHead]
    rope_base: int
    norm_eps: float
    rope_scaling: RopeScalingDict


def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)


def get_mask_mod(mask_mod: _mask_mod_signature, offset: int | Tensor):
    def _mask_mod(b, h, q, kv):
        return mask_mod(b, h, q + offset, kv)

    return _mask_mod


@dataclass
class ModelArgs[
    VocabSize,
    BlockSize,
    D,
    NHead,
    NLayer,
    IntermediateSize,
    NLocalHeads,
]:
    block_size: Dim[BlockSize] = 2048  # type: ignore[assignment]
    vocab_size: Dim[VocabSize] = 32000  # type: ignore[assignment]
    n_layer: Dim[NLayer] = 32  # type: ignore[assignment]
    n_head: Dim[NHead] = 32  # type: ignore[assignment]
    dim: Dim[D] = 4096  # type: ignore[assignment]
    intermediate_size: Dim[IntermediateSize] | None = None
    n_local_heads: Dim[NLocalHeads] = -1  # type: ignore[assignment]
    head_dim: Dim[D // NHead] = 64  # type: ignore[assignment]
    rope_base: int = 10000
    norm_eps: float = 1e-5
    rope_scaling: RopeScalingDict | None = None

    def __post_init__(self):
        if self.n_local_heads == -1:
            self.n_local_heads = self.n_head
        if self.intermediate_size is None:
            hidden_dim = 4 * self.dim
            n_hidden = int(2 * hidden_dim / 3)
            self.intermediate_size = find_multiple(n_hidden, 256)
        self.head_dim = self.dim // self.n_head

    @classmethod
    def from_name(cls, name: str):
        if name in transformer_configs:
            return cls(**transformer_configs[name])
        # fuzzy search
        config = [
            config
            for config in transformer_configs
            if config.lower() in str(name).lower()
        ]

        # We may have two or more configs matched (e.g. "7B" and "Mistral-7B"). Find the best config match,
        # take longer name (as it have more symbols matched)
        if len(config) > 1:
            config.sort(key=len, reverse=True)
            assert len(config[0]) != len(config[1]), (
                name
            )  # make sure only one 'best' match

        return cls(**transformer_configs[config[0]])


transformer_configs: dict[str, ModelArgsDict[Any, Any, Any, Any, Any, Any, Any]] = {
    "CodeLlama-7b-Python-hf": dict(
        block_size=16384, vocab_size=32000, n_layer=32, dim=4096, rope_base=1000000
    ),
    "7B": dict(n_layer=32, n_head=32, dim=4096),
    "13B": dict(n_layer=40, n_head=40, dim=5120),
    "30B": dict(n_layer=60, n_head=52, dim=6656),
    "34B": dict(
        n_layer=48,
        n_head=64,
        dim=8192,
        vocab_size=32000,
        n_local_heads=8,
        intermediate_size=22016,
        rope_base=1000000,
    ),  # CodeLlama-34B-Python-hf
    "70B": dict(
        n_layer=80, n_head=64, dim=8192, n_local_heads=8, intermediate_size=28672
    ),
    "Mistral-7B": dict(
        n_layer=32,
        n_head=32,
        n_local_heads=8,
        dim=4096,
        intermediate_size=14336,
        vocab_size=32000,
    ),
    "stories15M": dict(n_layer=6, n_head=6, dim=288),
    "stories110M": dict(n_layer=12, n_head=12, dim=768),
    "llama-3-8b": dict(
        block_size=8192,
        n_layer=32,
        n_head=32,
        n_local_heads=8,
        dim=4096,
        intermediate_size=14336,
        vocab_size=128256,
        rope_base=500000,
    ),
    "llama-3-70b": dict(
        block_size=8192,
        n_layer=80,
        n_head=64,
        n_local_heads=8,
        dim=8192,
        intermediate_size=28672,
        vocab_size=128256,
        rope_base=500000,
    ),
    "llama-3.1-8b": dict(
        block_size=131072,
        n_layer=32,
        n_head=32,
        n_local_heads=8,
        dim=4096,
        intermediate_size=14336,
        vocab_size=128256,
        rope_base=500000,
        rope_scaling=dict(
            factor=8.0,
            low_freq_factor=1.0,
            high_freq_factor=4.0,
            original_max_position_embeddings=8192,
        ),
    ),
    "llama-3.1-70b": dict(
        block_size=131072,
        n_layer=80,
        n_head=64,
        n_local_heads=8,
        dim=8192,
        intermediate_size=28672,
        vocab_size=128256,
        rope_base=500000,
        rope_scaling=dict(
            factor=8.0,
            low_freq_factor=1.0,
            high_freq_factor=4.0,
            original_max_position_embeddings=8192,
        ),
    ),
    "llama-3.1-405b": dict(
        block_size=131072,
        n_layer=126,
        n_head=128,
        n_local_heads=8,
        dim=16384,
        intermediate_size=53248,
        vocab_size=128256,
        rope_base=500000,
        rope_scaling=dict(
            factor=8.0,
            low_freq_factor=1.0,
            high_freq_factor=4.0,
            original_max_position_embeddings=8192,
        ),
    ),
}


def apply_rope_scaling[D](freqs: Tensor[D], rope_scaling: RopeScalingDict) -> Tensor[D]:
    factor = rope_scaling["factor"]
    low_freq_factor = rope_scaling["low_freq_factor"]
    high_freq_factor = rope_scaling["high_freq_factor"]
    old_context_len = rope_scaling["original_max_position_embeddings"]

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    new_freqs = []
    for freq in freqs:
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq / factor)
        else:
            assert low_freq_wavelen != high_freq_wavelen
            smooth = (old_context_len / wavelen - low_freq_factor) / (
                high_freq_factor - low_freq_factor
            )
            new_freqs.append((1 - smooth) * freq / factor + smooth * freq)
    return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)


def precompute_freqs_cis[SeqLen, HeadDim](
    seq_len: Dim[SeqLen],
    n_elem: Dim[HeadDim],
    base: int = 10000,
    dtype: torch.dtype = torch.bfloat16,
    rope_scaling: RopeScalingDict | None = None,
) -> Tensor[SeqLen, HeadDim // 2, 2]:
    freqs = 1.0 / (
        base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem)
    )
    assert_type(freqs, Tensor[HeadDim // 2])
    if rope_scaling is not None:
        freqs = apply_rope_scaling(freqs, rope_scaling)
    t = torch.arange(seq_len, device=freqs.device)
    assert_type(t, Tensor[SeqLen])
    freqs = torch.outer(t, freqs)
    assert_type(freqs, Tensor[SeqLen, HeadDim // 2])
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    assert_type(freqs_cis, Tensor[SeqLen, HeadDim // 2])
    # Use tuple instead of list for stack (meta-shape requires tuple)
    cache = torch.stack((freqs_cis.real, freqs_cis.imag), dim=-1)
    assert_type(cache, Tensor[SeqLen, HeadDim // 2, 2])
    return cache.to(dtype=dtype)


def apply_rotary_emb[B, T, NHeads, HeadDim](
    x: Tensor[B, T, NHeads, HeadDim], freqs_cis: Tensor[T, HeadDim // 2, 2]
) -> Tensor[B, T, NHeads, HeadDim]:
    xshaped = x.float().reshape(*x.size()[:-1], -1, 2)
    assert_type(xshaped, Tensor[B, T, NHeads, HeadDim // 2, 2])
    freqs_cis_reshaped = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    assert_type(freqs_cis_reshaped, Tensor[1, T, 1, HeadDim // 2, 2])
    # Use tuple instead of list so stack meta-shape can extract element shapes
    stack_input = (
        xshaped[..., 0] * freqs_cis_reshaped[..., 0]
        - xshaped[..., 1] * freqs_cis_reshaped[..., 1],
        xshaped[..., 1] * freqs_cis_reshaped[..., 0]
        + xshaped[..., 0] * freqs_cis_reshaped[..., 1],
    )
    x_out2 = torch.stack(stack_input, -1)
    assert_type(x_out2, Tensor[B, T, NHeads, HeadDim // 2, 2])

    x_out2 = x_out2.flatten(3)
    # Note: Type system computes (HeadDim // 2) * 2, which is algebraically equal to HeadDim (Issue 7)
    # reveal_type: Tensor[B, T, NHeads, ((HeadDim // 2) * 2)]
    return x_out2.type_as(x)  # type: ignore[bad-return]  # Issue 7: algebraic equivalence


class KVCache[MaxBatchSize, MaxSeqLen, NHeads, HeadDim](nn.Module):
    k_cache: Tensor[MaxBatchSize, NHeads, MaxSeqLen, HeadDim]
    v_cache: Tensor[MaxBatchSize, NHeads, MaxSeqLen, HeadDim]

    def __init__(
        self,
        max_batch_size: Dim[MaxBatchSize],
        max_seq_length: Dim[MaxSeqLen],
        n_heads: Dim[NHeads],
        head_dim: Dim[HeadDim],
        dtype=torch.bfloat16,
    ):
        super().__init__()
        cache_shape = (max_batch_size, n_heads, max_seq_length, head_dim)
        self.k_cache = nn.Buffer(torch.zeros(cache_shape, dtype=dtype))
        self.v_cache = nn.Buffer(torch.zeros(cache_shape, dtype=dtype))

    def update[B, S](
        self,
        input_pos: Tensor[S] | None,
        k_val: Tensor[B, NHeads, S, HeadDim],
        v_val: Tensor[B, NHeads, S, HeadDim],
    ) -> tuple[
        Tensor[MaxBatchSize, NHeads, MaxSeqLen, HeadDim],
        Tensor[MaxBatchSize, NHeads, MaxSeqLen, HeadDim],
    ]:
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos is not None
        assert input_pos.shape[0] == k_val.shape[2]

        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val

        return k_out, v_out


class FeedForward[D, IntermediateSize](nn.Module):
    def __init__(
        self, config: ModelArgs[Any, Any, D, Any, Any, IntermediateSize, Any]
    ) -> None:
        super().__init__()
        assert config.intermediate_size is not None
        self.w1 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        assert_type(self.w1, nn.Linear[D, IntermediateSize])
        self.w3 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        assert_type(self.w3, nn.Linear[D, IntermediateSize])
        self.w2 = nn.Linear(config.intermediate_size, config.dim, bias=False)
        assert_type(self.w2, nn.Linear[IntermediateSize, D])

    def forward[B, T](self, x: Tensor[B, T, D]) -> Tensor[B, T, D]:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RMSNorm[D](nn.Module):
    def __init__(self, dim: Dim[D], eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        assert_type(self.weight, Tensor[D])

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward[*Bs](self, x: Tensor[*Bs, D]) -> Tensor[*Bs, D]:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class Attention[D, NHead, NLocalHeads](nn.Module):
    def __init__(self, config: ModelArgs[Any, Any, D, NHead, Any, Any, NLocalHeads]):
        super().__init__()
        assert config.dim % config.n_head == 0

        total_head_dim = (config.n_head + 2 * config.n_local_heads) * config.head_dim
        # key, query, value projections for all heads, but in a batch
        self.wqkv = nn.Linear(config.dim, total_head_dim, bias=False)
        assert_type(
            self.wqkv,
            nn.Linear[D, ((NHead + 2 * NLocalHeads) * (D // NHead))],
        )
        self.wo = nn.Linear(config.dim, config.dim, bias=False)
        assert_type(self.wo, nn.Linear[D, D])
        self.kv_cache: KVCache[Any, Any, NLocalHeads, D // NHead] | None = None

        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_local_heads = config.n_local_heads
        self.dim = config.dim
        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(self, state_dict, prefix, *args):
        if prefix + "wq.weight" in state_dict:
            wq = state_dict.pop(prefix + "wq.weight")
            wk = state_dict.pop(prefix + "wk.weight")
            wv = state_dict.pop(prefix + "wv.weight")
            state_dict[prefix + "wqkv.weight"] = torch.cat([wq, wk, wv])

    def forward[B, T](
        self,
        x: Tensor[B, T, D],
        freqs_cis: Tensor[T, (D // NHead) // 2, 2],
        mask: BlockMask,
        input_pos: Tensor[T] | None = None,
    ) -> Tensor[B, T, D]:
        bsz, seqlen, _ = x.size()
        assert_type(bsz, Dim[B])
        assert_type(seqlen, Dim[T])

        kv_size = self.n_local_heads * self.head_dim
        assert_type(kv_size, Dim[NLocalHeads * (D // NHead)])
        # Using tuple instead of list to preserve individual element types for meta-shape inference
        q, k, v = self.wqkv(x).split((self.dim, kv_size, kv_size), dim=-1)
        assert_type(q, Tensor[B, T, D])
        assert_type(k, Tensor[B, T, (NLocalHeads * (D // NHead))])
        assert_type(v, Tensor[B, T, (NLocalHeads * (D // NHead))])

        q = q.view(bsz, seqlen, self.n_head, self.head_dim)
        assert_type(q, Tensor[B, T, NHead, (D // NHead)])
        k = k.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        assert_type(k, Tensor[B, T, NLocalHeads, (D // NHead)])
        v = v.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        assert_type(v, Tensor[B, T, NLocalHeads, (D // NHead)])

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        assert_type(q, Tensor[B, NHead, T, (D // NHead)])
        assert_type(k, Tensor[B, NLocalHeads, T, (D // NHead)])
        assert_type(v, Tensor[B, NLocalHeads, T, (D // NHead)])

        if self.kv_cache is not None:
            k, v = self.kv_cache.update(input_pos, k, v)

        y = flex_attention(
            q, k, v, block_mask=mask, enable_gqa=(self.n_head != self.n_local_heads)
        )
        assert_type(y, Tensor[B, NHead, T, (D // NHead)])

        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)
        assert_type(y, Tensor[B, T, D])

        y = self.wo(y)
        return y


class Transformer[
    VocabSize,
    BlockSize,
    D,
    NHead,
    NLayer,
    IntermediateSize,
    NLocalHeads,
](nn.Module):
    def __init__(
        self,
        config: ModelArgs[
            VocabSize, BlockSize, D, NHead, NLayer, IntermediateSize, NLocalHeads
        ],
    ) -> None:
        super().__init__()
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        assert_type(self.tok_embeddings, nn.Embedding[VocabSize, D])
        self.layers = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.n_layer)
        )
        assert_type(
            self.layers,
            nn.ModuleList[TransformerBlock[D, NHead, IntermediateSize, NLocalHeads]],
        )
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        assert_type(self.norm, RMSNorm[D])
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
        assert_type(self.output, nn.Linear[D, VocabSize])

        self.freqs_cis: Tensor[BlockSize, (D // NHead) // 2, 2] | None = None
        self.mask_cache: Optional[Tensor] = None
        self.max_batch_size = -1
        self.max_seq_length = -1
        self.get_mask_mod = get_mask_mod

    def setup_caches(self, max_batch_size, max_seq_length):
        if (
            self.max_seq_length >= max_seq_length
            and self.max_batch_size >= max_batch_size
        ):
            return
        head_dim = self.config.dim // self.config.n_head
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        dtype = self.output.weight.dtype
        # For quantized layers, dtype is encoded in scales
        if hasattr(self.output, "scales"):
            dtype = self.output.scales.dtype
        elif hasattr(self.output, "scales_and_zeros"):
            dtype = self.output.scales_and_zeros.dtype
        for b in self.layers:
            b.attention.kv_cache = KVCache(
                max_batch_size,
                max_seq_length,
                self.config.n_local_heads,
                head_dim,
                dtype,
            )

        self.freqs_cis = precompute_freqs_cis(
            self.config.block_size,
            self.config.dim // self.config.n_head,
            self.config.rope_base,
            dtype,
            self.config.rope_scaling,
        )

    def forward[B, T](
        self, mask: BlockMask, idx: Tensor[B, T], input_pos: Tensor[T] | None = None
    ) -> Tensor[B, T, VocabSize]:
        assert self.freqs_cis is not None, "Caches must be initialized first"
        assert input_pos is not None, "input_pos must be provided"
        assert mask.mask_mod is not None, "mask_mod must be set"
        mask.mask_mod = self.get_mask_mod(mask.mask_mod, input_pos[0])
        freqs_cis = self.freqs_cis[input_pos]
        assert_type(freqs_cis, Tensor[T, (D // NHead) // 2, 2])
        x = self.tok_embeddings(idx)
        assert_type(x, Tensor[B, T, D])

        for _i, layer in enumerate(self.layers):
            x = layer(x, input_pos, freqs_cis, mask)
        x = self.norm(x)
        assert_type(x, Tensor[B, T, D])
        logits = self.output(x)
        assert_type(logits, Tensor[B, T, VocabSize])
        return logits

    @classmethod
    def from_name(cls, name: str):
        return cls(ModelArgs.from_name(name))


class TransformerBlock[D, NHead, IntermediateSize, NLocalHeads](nn.Module):
    attention: Attention[D, NHead, NLocalHeads]
    feed_forward: FeedForward[D, IntermediateSize]
    ffn_norm: RMSNorm[D]
    attention_norm: RMSNorm[D]

    def __init__(
        self, config: ModelArgs[Any, Any, D, NHead, Any, IntermediateSize, NLocalHeads]
    ) -> None:
        super().__init__()
        self.attention = Attention(config)
        assert_type(self.attention, Attention[D, NHead, NLocalHeads])
        self.feed_forward = FeedForward(config)
        assert_type(self.feed_forward, FeedForward[D, IntermediateSize])
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        assert_type(self.ffn_norm, RMSNorm[D])
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)
        assert_type(self.attention_norm, RMSNorm[D])

    def forward[B, T](
        self,
        x: Tensor[B, T, D],
        input_pos: Tensor[T],
        freqs_cis: Tensor[T, (D // NHead) // 2, 2],
        mask: BlockMask,
    ) -> Tensor[B, T, D]:
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask, input_pos)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out
