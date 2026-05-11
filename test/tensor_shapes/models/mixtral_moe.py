# Portions (c) Meta Platforms, Inc. and affiliates.
#
# This source code is adapted from pytorch/benchmark (TorchBenchmark),
# which is licensed under the BSD 3-Clause License:
# https://github.com/pytorch/benchmark/blob/main/LICENSE
#
# Original model: pytorch-labs/gpt-fast (mixtral-moe/model.py)
#
# This adaptation adds tensor shape type annotations for pyrefly.

# ## Inventory
# - [x] find_multiple — utility, no tensors
# - [x] ModelArgs — dataclass config
# - [x] ModelArgs.__post_init__
# - [x] ModelArgs.from_name
# - [x] transformer_configs — dict constant
# - [x] KVCache.__init__ — Dims: max_batch_size, max_seq_length, n_heads, head_dim
# - [x] KVCache.update
# - [x] Transformer.__init__
# - [x] Transformer.setup_caches
# - [x] Transformer.forward
# - [x] Transformer.from_name
# - [x] TransformerBlock.__init__
# - [x] TransformerBlock.forward
# - [x] Attention.__init__
# - [x] Attention.load_hook
# - [x] Attention.forward
# - [x] ConditionalFeedForward.__init__
# - [x] ConditionalFeedForward.forward
# - [x] MOEFeedForward.__init__
# - [x] MOEFeedForward.forward
# - [x] RMSNorm.__init__
# - [x] RMSNorm._norm
# - [x] RMSNorm.forward
# - [x] precompute_freqs_cis — standalone function
# - [x] apply_rotary_emb — standalone function

from dataclasses import dataclass
from typing import Any, assert_type, TYPE_CHECKING

import torch
import torch.nn as nn
from torch.nn import functional as F

if TYPE_CHECKING:
    from torch import Tensor
    from torch_shapes import Dim


def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)


class RMSNorm[D](nn.Module):
    def __init__(self, dim: Dim[D], eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm[*Bs](self, x: Tensor[*Bs, D]) -> Tensor[*Bs, D]:
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward[*Bs](self, x: Tensor[*Bs, D]) -> Tensor[*Bs, D]:
        output = self._norm(x.float()).type_as(x)
        assert_type(output, Tensor[*Bs, D])
        result = output * self.weight
        assert_type(result, Tensor[*Bs, D])
        return result


class KVCache[B, NHead, MaxSeq, HD](nn.Module):
    def __init__(
        self,
        max_batch_size: Dim[B],
        max_seq_length: Dim[MaxSeq],
        n_heads: Dim[NHead],
        head_dim: Dim[HD],
        dtype: Any = torch.bfloat16,
    ):
        super().__init__()
        self.k_cache = nn.Buffer(
            torch.zeros(max_batch_size, n_heads, max_seq_length, head_dim, dtype=dtype)
        )
        self.v_cache = nn.Buffer(
            torch.zeros(max_batch_size, n_heads, max_seq_length, head_dim, dtype=dtype)
        )

    def update[S](
        self,
        input_pos: Tensor[S],
        k_val: Tensor[B, NHead, S, HD],
        v_val: Tensor[B, NHead, S, HD],
    ) -> tuple[Tensor[B, NHead, MaxSeq, HD], Tensor[B, NHead, MaxSeq, HD]]:
        k_out = self.k_cache
        assert_type(k_out, Tensor[B, NHead, MaxSeq, HD])
        v_out = self.v_cache
        assert_type(v_out, Tensor[B, NHead, MaxSeq, HD])
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val
        return k_out, v_out


class ConditionalFeedForward[NExp, Inter, D](nn.Module):
    def __init__(
        self,
        num_experts: Dim[NExp],
        intermediate_size: Dim[Inter],
        dim: Dim[D],
    ):
        super().__init__()
        self.w1 = nn.Parameter(torch.empty(num_experts, intermediate_size, dim))
        self.w2 = nn.Parameter(torch.empty(num_experts, dim, intermediate_size))
        self.w3 = nn.Parameter(torch.empty(num_experts, intermediate_size, dim))

    def forward[T, A](
        self, x: Tensor[T, D], expert_indices: Tensor[T, A]
    ) -> Tensor[T, A, D]:
        w1_weights = self.w1[expert_indices]
        assert_type(w1_weights, Tensor[T, A, Inter, D])
        w3_weights = self.w3[expert_indices]
        assert_type(w3_weights, Tensor[T, A, Inter, D])
        w2_weights = self.w2[expert_indices]
        assert_type(w2_weights, Tensor[T, A, D, Inter])
        x1 = F.silu(torch.einsum("ti,taoi -> tao", x, w1_weights))
        assert_type(x1, Tensor[T, A, Inter])
        x3 = torch.einsum("ti, taoi -> tao", x, w3_weights)
        assert_type(x3, Tensor[T, A, Inter])
        expert_outs = torch.einsum("tao, taio -> tai", (x1 * x3), w2_weights)
        assert_type(expert_outs, Tensor[T, A, D])
        return expert_outs


class MOEFeedForward[D, NExp, A, Inter](nn.Module):
    def __init__(
        self,
        dim: Dim[D],
        num_experts: Dim[NExp],
        num_activated_experts: Dim[A],
        intermediate_size: Dim[Inter],
    ) -> None:
        super().__init__()
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.cond_ffn = ConditionalFeedForward(num_experts, intermediate_size, dim)
        self.dim = dim
        self.num_activated_experts = num_activated_experts

    def forward[T](self, x: Tensor[T, D]) -> Tensor[T, D]:
        scores = self.gate(x)
        assert_type(scores, Tensor[T, NExp])
        expert_weights = F.softmax(scores, dim=-1)
        assert_type(expert_weights, Tensor[T, NExp])
        expert_weights, expert_indices = torch.topk(
            expert_weights, self.num_activated_experts, dim=-1
        )
        assert_type(expert_weights, Tensor[T, A])
        assert_type(expert_indices, Tensor[T, A])
        expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)
        assert_type(expert_weights, Tensor[T, A])
        expert_outs = self.cond_ffn(x, expert_indices)
        assert_type(expert_outs, Tensor[T, A, D])
        result = torch.einsum("tai,ta -> ti", expert_outs, expert_weights)
        assert_type(result, Tensor[T, D])
        return result


@dataclass
class ModelArgs:
    block_size: int = 2048
    vocab_size: int = 32000
    n_layer: int = 32
    n_head: int = 32
    dim: int = 4096
    intermediate_size: int | None = None
    n_local_heads: int = -1
    head_dim: int = 64
    rope_base: float = 10000
    norm_eps: float = 1e-5
    num_experts: int = 8
    num_activated_experts: int = 2

    def __post_init__(self) -> None:
        if self.n_local_heads == -1:
            self.n_local_heads = self.n_head
        if self.intermediate_size is None:
            hidden_dim = 4 * self.dim
            n_hidden = int(2 * hidden_dim / 3)
            self.intermediate_size = find_multiple(n_hidden, 256)
        self.head_dim = self.dim // self.n_head

    @classmethod
    def from_name(cls, name: str) -> "ModelArgs":
        if name in transformer_configs:
            return cls(**transformer_configs[name])
        config = [
            config
            for config in transformer_configs
            if config in str(name).upper() or config in str(name)
        ]
        if len(config) != 1:
            raise AssertionError(
                f"Expected exactly one config match for '{name}', but got {len(config)}: {config}"
            )
        return cls(**transformer_configs[config[0]])


transformer_configs = {
    "Mixtral-8x7B-v0.1": dict(
        block_size=32768,
        n_layer=16,
        n_head=32,
        n_local_heads=8,
        dim=4096,
        intermediate_size=14336,
        rope_base=1000000.0,
        num_experts=8,
        num_activated_experts=2,
    ),
}


def precompute_freqs_cis(seq_len: int, n_elem: int, base: float = 10000) -> Tensor:
    freqs = 1.0 / (
        base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem)
    )
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache.to(dtype=torch.bfloat16)


def apply_rotary_emb[*S](x: Tensor[*S], freqs_cis: Tensor) -> Tensor[*S]:
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)  # type: ignore[bad-argument-type]
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        -1,
    )
    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)  # type: ignore[bad-return]  # shape preserved — rotary doesn't change dims


class Attention[D, NHead, NLocalHead, HD](nn.Module):
    def __init__(
        self,
        dim: Dim[D],
        n_head: Dim[NHead],
        n_local_heads: Dim[NLocalHead],
        head_dim: Dim[HD],
    ):
        super().__init__()
        total_head_dim = (n_head + 2 * n_local_heads) * head_dim
        self.wqkv = nn.Linear(dim, total_head_dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.kv_cache: KVCache[Any, NLocalHead, Any, HD] | None = None

        self.n_head = n_head
        self.head_dim = head_dim
        self.n_local_heads = n_local_heads
        self.dim = dim
        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(self, state_dict: dict[str, Tensor], prefix: str, *args: Any) -> None:
        if prefix + "wq.weight" in state_dict:
            wq = state_dict.pop(prefix + "wq.weight")
            wk = state_dict.pop(prefix + "wk.weight")
            wv = state_dict.pop(prefix + "wv.weight")
            state_dict[prefix + "wqkv.weight"] = torch.cat([wq, wk, wv])

    def forward[B, SeqLen](
        self,
        x: Tensor[B, SeqLen, D],
        freqs_cis: Tensor,
        mask: Tensor,
        input_pos: Tensor | None = None,
    ) -> Tensor[B, SeqLen, D]:
        bsz, seqlen, _dim = x.shape

        kv_size = self.n_local_heads * self.head_dim
        q, k, v = self.wqkv(x).split((self.dim, kv_size, kv_size), dim=-1)
        assert_type(q, Tensor[B, SeqLen, D])
        assert_type(k, Tensor[B, SeqLen, HD * NLocalHead])
        assert_type(v, Tensor[B, SeqLen, HD * NLocalHead])

        q = q.view(bsz, seqlen, self.n_head, self.head_dim)
        assert_type(q, Tensor[B, SeqLen, NHead, HD])
        k = k.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        assert_type(k, Tensor[B, SeqLen, NLocalHead, HD])
        v = v.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        assert_type(v, Tensor[B, SeqLen, NLocalHead, HD])

        q = apply_rotary_emb(q, freqs_cis)
        assert_type(q, Tensor[B, SeqLen, NHead, HD])
        k = apply_rotary_emb(k, freqs_cis)
        assert_type(k, Tensor[B, SeqLen, NLocalHead, HD])

        q = q.transpose(1, 2)
        assert_type(q, Tensor[B, NHead, SeqLen, HD])
        k = k.transpose(1, 2)
        assert_type(k, Tensor[B, NLocalHead, SeqLen, HD])
        v = v.transpose(1, 2)
        assert_type(v, Tensor[B, NLocalHead, SeqLen, HD])

        if self.kv_cache is not None:
            assert input_pos is not None
            k, v = self.kv_cache.update(input_pos, k, v)

        k = k.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
        assert_type(k, Tensor)  # type: ignore  # union from kv_cache branch join
        v = v.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
        assert_type(v, Tensor)  # type: ignore  # union from kv_cache branch join
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
        assert_type(y, Tensor)  # type: ignore  # A1: NLocalHead*(NHead//NLocalHead) != NHead

        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)
        assert_type(y, Tensor[B, SeqLen, D])

        y = self.wo(y)
        assert_type(y, Tensor[B, SeqLen, D])
        return y


class TransformerBlock[D, NHead, NLocalHead, HD, NExp, A, Inter](nn.Module):
    def __init__(
        self,
        dim: Dim[D],
        n_head: Dim[NHead],
        n_local_heads: Dim[NLocalHead],
        head_dim: Dim[HD],
        num_experts: Dim[NExp],
        num_activated_experts: Dim[A],
        intermediate_size: Dim[Inter],
        norm_eps: float,
    ) -> None:
        super().__init__()
        self.attention = Attention(dim, n_head, n_local_heads, head_dim)
        self.block_sparse_moe = MOEFeedForward(
            dim, num_experts, num_activated_experts, intermediate_size
        )
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)
        self.attention_norm = RMSNorm(dim, eps=norm_eps)

    def forward[B, SeqLen](
        self,
        x: Tensor[B, SeqLen, D],
        input_pos: Tensor | None,
        freqs_cis: Tensor,
        mask: Tensor,
    ) -> Tensor[B, SeqLen, D]:
        attn_out = self.attention(self.attention_norm(x), freqs_cis, mask, input_pos)
        assert_type(attn_out, Tensor[B, SeqLen, D])
        h = x + attn_out
        assert_type(h, Tensor[B, SeqLen, D])
        h_normed = self.ffn_norm(h)
        assert_type(h_normed, Tensor[B, SeqLen, D])
        h_flat = h_normed.view(-1, h_normed.shape[-1])
        assert_type(h_flat, Tensor[B * SeqLen, D])
        ffn_out = self.block_sparse_moe(h_flat)
        assert_type(ffn_out, Tensor[B * SeqLen, D])
        ffn_out = ffn_out.view(h.shape)
        assert_type(ffn_out, Tensor[B, SeqLen, D])
        out = h + ffn_out
        assert_type(out, Tensor[B, SeqLen, D])
        return out


class Transformer(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config

        assert config.intermediate_size is not None
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(
            TransformerBlock(
                config.dim,
                config.n_head,
                config.n_local_heads,
                config.head_dim,
                config.num_experts,
                config.num_activated_experts,
                config.intermediate_size,
                config.norm_eps,
            )
            for _ in range(config.n_layer)
        )
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        self.freqs_cis: Tensor | None = None
        self.mask_cache: Tensor | None = None
        self.max_batch_size = -1
        self.max_seq_length = -1

    def setup_caches(self, max_batch_size: int, max_seq_length: int) -> None:
        if (
            self.max_seq_length >= max_seq_length
            and self.max_batch_size >= max_batch_size
        ):
            return
        head_dim = self.config.dim // self.config.n_head
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        for b in self.layers:
            b.attention.kv_cache = KVCache(
                max_batch_size, max_seq_length, self.config.n_local_heads, head_dim
            )

        self.freqs_cis = precompute_freqs_cis(
            self.config.block_size,
            self.config.dim // self.config.n_head,
            self.config.rope_base,
        )
        self.causal_mask = torch.tril(
            torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool)
        )

    def forward(self, idx: Tensor, input_pos: Tensor | None = None) -> Tensor:
        if self.freqs_cis is None:
            raise AssertionError("Caches must be initialized first")
        mask = self.causal_mask[None, None, input_pos]
        assert_type(mask, Tensor)  # bare — indexing with None/input_pos
        freqs_cis = self.freqs_cis[input_pos]
        assert_type(freqs_cis, Tensor)  # bare — indexing on bare freqs_cis
        # ModelArgs uses plain int — sub-module dims are Unknown
        x = self.tok_embeddings(idx)
        assert_type(x, Tensor)  # type: ignore  # Unknown — config dims are int not Dim

        for i, layer in enumerate(self.layers):
            x = layer(x, input_pos, freqs_cis, mask)
        assert_type(x, Tensor)  # type: ignore  # Unknown — loop widens
        x = self.norm(x)
        assert_type(x, Tensor)  # type: ignore  # Tensor[Unknown, Unknown, Unknown]
        logits = self.output(x)
        assert_type(logits, Tensor)  # type: ignore  # Tensor[Unknown, Unknown, Unknown]
        return logits

    @classmethod
    def from_name(cls, name: str) -> "Transformer":
        return cls(ModelArgs.from_name(name))


def test_rmsnorm() -> None:
    norm = RMSNorm(64)
    x: Tensor[2, 10, 64] = torch.randn(2, 10, 64)
    out = norm(x)
    assert_type(out, Tensor[2, 10, 64])


def test_conditional_feed_forward() -> None:
    cff = ConditionalFeedForward(8, 256, 64)
    x: Tensor[5, 64] = torch.randn(5, 64)
    indices: Tensor[5, 2] = torch.randint(0, 8, (5, 2))
    out = cff(x, indices)
    assert_type(out, Tensor[5, 2, 64])


def test_moe_feed_forward() -> None:
    moe = MOEFeedForward(64, 8, 2, 256)
    x: Tensor[5, 64] = torch.randn(5, 64)
    out = moe(x)
    assert_type(out, Tensor[5, 64])


def test_attention() -> None:
    attn = Attention(64, 8, 4, 8)
    x: Tensor[2, 10, 64] = torch.randn(2, 10, 64)
    freqs: Tensor = torch.randn(10, 4, 2)
    mask: Tensor = torch.ones(1, 1, 10, 10)
    out = attn(x, freqs, mask)
    assert_type(out, Tensor[2, 10, 64])


def test_transformer_block() -> None:
    block = TransformerBlock(64, 8, 4, 8, 8, 2, 256, 1e-5)
    x: Tensor[2, 10, 64] = torch.randn(2, 10, 64)
    freqs: Tensor = torch.randn(10, 4, 2)
    mask: Tensor = torch.ones(1, 1, 10, 10)
    out = block(x, None, freqs, mask)
    assert_type(out, Tensor[2, 10, 64])
