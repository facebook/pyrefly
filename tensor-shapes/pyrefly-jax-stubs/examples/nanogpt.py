# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
NanoGPT language model in JAX with shape annotations.

Equivalent to the PyTorch NanoGPT example (`karpathy/nanoGPT`), implemented
in JAX using basic pytrees, without external framework dependencies.

Adaptations relative to the PyTorch NanoGPT implementation:
- Functional PyTree Modules: Layers are immutable `@dataclass` PyTrees with static
  configuration fields marked via `jax.tree.static()`.
- Omission of Dropout: Stochastic dropout is omitted to keep the forward pass
  deterministic without threading PRNG keys through every call.
- Deterministic Weight Initialization: Parameters are initialized deterministically
  (`jnp.full`, `jnp.ones`, `jnp.zeros`) so shape and runtime tests run without PRNG keys.
- Weight Tying: The language modeling head (`lm_head`) shares the transposed token
  embedding matrix (`wte.weight.T`).
- Functional Model Surgery (`crop_block_size`): Returns a new `GPT` PyTree with
  type-level updated `NewBlockSize` rather than mutating parameters in place.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, assert_type, Literal, overload, TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax import Array
from shape_extensions import assert_shape, Elements, Int, IntTuple, IntVar

# ============================================================================
# Core Layers: Linear, LayerNorm, Embedding
# ============================================================================


@jax.tree_util.register_dataclass
@dataclass
class Linear[In: IntVar, Out: IntVar]:
    weight: Array[[In, Out]]
    bias: Array[[Out]] | None

    @classmethod
    def init(
        cls, in_features: Int[In], out_features: Int[Out], bias: bool = True
    ) -> Linear[In, Out]:
        # Deterministic initialization avoids PRNG key plumbing in shape tests.
        scale = 1.0 / math.sqrt(in_features)
        return cls(
            weight=jnp.full((in_features, out_features), scale),
            bias=jnp.zeros(out_features) if bias else None,
        )

    def __call__[Batch: IntTuple](
        self, x: Array[[*Elements[Batch], In]]
    ) -> Array[[*Elements[Batch], Out]]:
        out = jnp.matmul(x, self.weight)
        if self.bias is not None:
            out = out + self.bias
        return out


@jax.tree_util.register_dataclass
@dataclass
class LayerNorm[Features: IntVar]:
    """LayerNorm with optional bias, generic over normalized dimension size."""

    weight: Array[[Features]]
    bias: Array[[Features]] | None
    eps: float = jax.tree.static(default=1e-5)

    @classmethod
    def init(
        cls, features: Int[Features], bias: bool = True, eps: float = 1e-5
    ) -> LayerNorm[Features]:
        return cls(
            weight=jnp.ones(features),
            bias=jnp.zeros(features) if bias else None,
            eps=eps,
        )

    def __call__[Batch: IntTuple](
        self, x: Array[[*Elements[Batch], Features]]
    ) -> Array[[*Elements[Batch], Features]]:
        mean = jnp.mean(x, axis=-1, keepdims=True)
        variance = jnp.var(x, axis=-1, keepdims=True)
        out = self.weight * (x - mean) / jnp.sqrt(variance + self.eps)
        if self.bias is not None:
            out = out + self.bias
        return out


@jax.tree_util.register_dataclass
@dataclass
class Embedding[NumEmbeddings: IntVar, EmbeddingDim: IntVar]:
    weight: Array[[NumEmbeddings, EmbeddingDim]]

    @classmethod
    def init(
        cls, num_embeddings: Int[NumEmbeddings], embedding_dim: Int[EmbeddingDim]
    ) -> Embedding[NumEmbeddings, EmbeddingDim]:
        return cls(weight=jnp.ones((num_embeddings, embedding_dim)))

    def __call__[Batch: IntTuple](
        self, x: Array[Batch]
    ) -> Array[[*Elements[Batch], EmbeddingDim]]:
        return self.weight[x]


# ============================================================================
# Configuration
# ============================================================================


@dataclass(frozen=True)
class GPTConfig[
    VocabSize: IntVar,
    BlockSize: IntVar,
    NEmbedding: IntVar,
    NHead: IntVar,
    NLayer: IntVar,
]:
    """Configuration for GPT model, generic over key dimensions."""

    block_size: Int[BlockSize]
    vocab_size: Int[VocabSize]
    n_layer: Int[NLayer]
    n_head: Int[NHead]
    n_embd: Int[NEmbedding]
    bias: bool = True


# ============================================================================
# Attention, MLP, and Transformer Block
# ============================================================================


@jax.tree_util.register_dataclass
@dataclass
class CausalSelfAttention[
    NEmbedding: IntVar,
    NHead: IntVar,
]:
    """Multi-head causal self-attention."""

    c_attn: Linear[NEmbedding, 3 * NEmbedding]
    c_proj: Linear[NEmbedding, NEmbedding]
    n_head: Int[NHead] = jax.tree.static()
    n_embd: Int[NEmbedding] = jax.tree.static()

    @classmethod
    def init(
        cls, config: GPTConfig[Any, Any, NEmbedding, NHead, Any]
    ) -> CausalSelfAttention[NEmbedding, NHead]:
        assert config.n_embd % config.n_head == 0
        return cls(
            c_attn=Linear.init(config.n_embd, 3 * config.n_embd, bias=config.bias),
            c_proj=Linear.init(config.n_embd, config.n_embd, bias=config.bias),
            n_head=config.n_head,
            n_embd=config.n_embd,
        )

    def __call__[B: IntVar, T: IntVar](
        self, x: Array[[B, T, NEmbedding]]
    ) -> Array[[B, T, NEmbedding]]:
        b, t, c = x.shape
        assert_type(b, Int[B])
        assert_type(t, Int[T])
        assert_type(c, Int[NEmbedding])
        head_dim = c // self.n_head

        # Fused QKV projection: (B, T, NEmbedding) -> (B, T, 3 * NEmbedding)
        c_attn = self.c_attn(x)
        if TYPE_CHECKING:
            assert_type(c_attn, Array[[B, T, (3 * NEmbedding)]])

        # Unpack Q, K, V from fused projection
        qkv = c_attn.reshape(b, t, 3, c)
        assert_type(qkv, Array[[B, T, 3, NEmbedding]])
        q = qkv[:, :, 0]
        k = qkv[:, :, 1]
        v = qkv[:, :, 2]
        assert_type(q, Array[[B, T, NEmbedding]])
        assert_type(k, Array[[B, T, NEmbedding]])
        assert_type(v, Array[[B, T, NEmbedding]])

        # Split heads and move head dimension forward: (B, NHead, T, HeadDim)
        q_heads = q.reshape(b, t, self.n_head, head_dim).swapaxes(1, 2)
        k_heads = k.reshape(b, t, self.n_head, head_dim).swapaxes(1, 2)
        v_heads = v.reshape(b, t, self.n_head, head_dim).swapaxes(1, 2)
        if TYPE_CHECKING:
            assert_type(q_heads, Array[[B, NHead, T, (NEmbedding // NHead)]])
            assert_type(k_heads, Array[[B, NHead, T, (NEmbedding // NHead)]])
            assert_type(v_heads, Array[[B, NHead, T, (NEmbedding // NHead)]])

        # Scaled dot-product causal self-attention: (B, NHead, T, HeadDim) x (B, NHead, HeadDim, T) -> (B, NHead, T, T)
        att = jnp.matmul(q_heads, k_heads.swapaxes(-2, -1)) / math.sqrt(head_dim)
        assert_type(att, Array[[B, NHead, T, T]])
        causal_mask = jnp.tril(jnp.ones((t, t), dtype=jnp.bool_))
        assert_type(causal_mask, Array[[T, T]])
        att = jnp.where(causal_mask, att, -1e9)
        assert_type(att, Array[[B, NHead, T, T]])
        att = jax.nn.softmax(att, axis=-1)
        assert_type(att, Array[[B, NHead, T, T]])

        y = jnp.matmul(att, v_heads)
        if TYPE_CHECKING:
            assert_type(y, Array[[B, NHead, T, (NEmbedding // NHead)]])

        # Re-assemble head outputs side by side
        y_merged = y.swapaxes(1, 2).reshape(b, t, c)
        assert_type(y_merged, Array[[B, T, NEmbedding]])

        # Output projection
        out = self.c_proj(y_merged)
        assert_type(out, Array[[B, T, NEmbedding]])
        return out


@jax.tree_util.register_dataclass
@dataclass
class MLP[NEmbedding: IntVar]:
    """Two-layer feed-forward network with GELU activation."""

    c_fc: Linear[NEmbedding, 4 * NEmbedding]
    c_proj: Linear[4 * NEmbedding, NEmbedding]

    @classmethod
    def init(cls, config: GPTConfig[Any, Any, NEmbedding, Any, Any]) -> MLP[NEmbedding]:
        return cls(
            c_fc=Linear.init(config.n_embd, 4 * config.n_embd, bias=config.bias),
            c_proj=Linear.init(4 * config.n_embd, config.n_embd, bias=config.bias),
        )

    def __call__[B: IntVar, T: IntVar](
        self, x: Array[[B, T, NEmbedding]]
    ) -> Array[[B, T, NEmbedding]]:
        h = self.c_fc(x)
        if TYPE_CHECKING:
            assert_type(h, Array[[B, T, (4 * NEmbedding)]])
        h = jax.nn.gelu(h, approximate=True)
        if TYPE_CHECKING:
            assert_type(h, Array[[B, T, (4 * NEmbedding)]])
        out = self.c_proj(h)
        assert_type(out, Array[[B, T, NEmbedding]])
        return out


@jax.tree_util.register_dataclass
@dataclass
class Block[NEmbedding: IntVar, NHead: IntVar]:
    """Transformer block with Pre-LayerNorm, causal self-attention, and MLP."""

    ln_1: LayerNorm[NEmbedding]
    attn: CausalSelfAttention[NEmbedding, NHead]
    ln_2: LayerNorm[NEmbedding]
    mlp: MLP[NEmbedding]

    @classmethod
    def init(
        cls, config: GPTConfig[Any, Any, NEmbedding, NHead, Any]
    ) -> Block[NEmbedding, NHead]:
        return cls(
            ln_1=LayerNorm.init(config.n_embd, bias=config.bias),
            attn=CausalSelfAttention.init(config),
            ln_2=LayerNorm.init(config.n_embd, bias=config.bias),
            mlp=MLP.init(config),
        )

    def __call__[B: IntVar, T: IntVar](
        self, x: Array[[B, T, NEmbedding]]
    ) -> Array[[B, T, NEmbedding]]:
        x = x + self.attn(self.ln_1(x))
        assert_type(x, Array[[B, T, NEmbedding]])
        x = x + self.mlp(self.ln_2(x))
        assert_type(x, Array[[B, T, NEmbedding]])
        return x


# ============================================================================
# GPT Language Model
# ============================================================================


@jax.tree_util.register_dataclass
@dataclass
class GPT[
    VocabSize: IntVar,
    BlockSize: IntVar,
    NEmbedding: IntVar,
    NHead: IntVar,
    NLayer: IntVar,
]:
    """GPT Language Model, generic over vocabulary size, block size, embedding dim, heads, and layers."""

    wte: Embedding[VocabSize, NEmbedding]
    wpe: Embedding[BlockSize, NEmbedding]
    h: list[Block[NEmbedding, NHead]]
    ln_f: LayerNorm[NEmbedding]
    config: GPTConfig[VocabSize, BlockSize, NEmbedding, NHead, NLayer] = (
        jax.tree.static()
    )

    @classmethod
    def init(
        cls,
        config: GPTConfig[VocabSize, BlockSize, NEmbedding, NHead, NLayer],
    ) -> GPT[VocabSize, BlockSize, NEmbedding, NHead, NLayer]:
        wte = Embedding.init(config.vocab_size, config.n_embd)
        wpe = Embedding.init(config.block_size, config.n_embd)
        h = [Block.init(config) for _ in range(config.n_layer)]
        ln_f = LayerNorm.init(config.n_embd, bias=config.bias)
        return cls(
            wte=wte,
            wpe=wpe,
            h=h,
            ln_f=ln_f,
            config=config,
        )

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Return the total number of parameters in the model PyTree."""
        n_params = sum(leaf.size for leaf in jax.tree.leaves(self))
        if non_embedding:
            n_params -= self.wpe.weight.size
        return n_params

    def estimate_mfu(self, fwdbwd_per_iter: int, dt: float) -> float:
        """Estimate model flops utilization (MFU) relative to A100 bfloat16 peak FLOPS."""
        n = self.get_num_params()
        cfg = self.config
        l_layers, n_head, head_dim, seq_len = (
            cfg.n_layer,
            cfg.n_head,
            cfg.n_embd // cfg.n_head,
            cfg.block_size,
        )
        flops_per_token = 6 * n + 12 * l_layers * n_head * head_dim * seq_len
        flops_per_fwdbwd = flops_per_token * seq_len
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0 / dt)
        flops_promised = 312e12
        return flops_achieved / flops_promised

    @overload
    def __call__[B: IntVar, T: IntVar](
        self, idx: Array[[B, T]], targets: None = None
    ) -> tuple[Array[[B, 1, VocabSize]], None]: ...

    @overload
    def __call__[B: IntVar, T: IntVar](
        self, idx: Array[[B, T]], targets: Array[[B, T]]
    ) -> tuple[Array[[B, T, VocabSize]], Array[[]]]: ...

    def __call__[B: IntVar, T: IntVar](
        self, idx: Array[[B, T]], targets: Array[[B, T]] | None = None
    ) -> tuple[Any, Any]:
        """Forward pass computing logits and optional cross-entropy loss.

        When provided, `targets` must be pre-shifted by 1 token relative to `idx`
        (i.e., `targets[:, t]` is the ground-truth next token following `idx[:, :t+1]`).
        """
        b, t = idx.shape
        assert_type(b, Int[B])
        assert_type(t, Int[T])
        assert t <= self.config.block_size, (
            f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        )
        pos = jnp.arange(t, dtype=jnp.int32)
        assert_type(pos, Array[[T]])

        tok_emb = self.wte(idx)
        assert_type(tok_emb, Array[[B, T, NEmbedding]])
        pos_emb = self.wpe(pos)
        assert_type(pos_emb, Array[[T, NEmbedding]])
        x = tok_emb + pos_emb
        assert_type(x, Array[[B, T, NEmbedding]])

        for block in self.h:
            x = block(x)
        assert_type(x, Array[[B, T, NEmbedding]])

        x = self.ln_f(x)
        assert_type(x, Array[[B, T, NEmbedding]])

        # Weight tying: project hidden states to vocabulary using transposed wte.weight
        tied_weight = self.wte.weight.T
        assert_type(tied_weight, Array[[NEmbedding, VocabSize]])

        if targets is not None:
            logits = jnp.matmul(x, tied_weight)
            assert_type(logits, Array[[B, T, VocabSize]])
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            assert_type(log_probs, Array[[B, T, VocabSize]])
            target_log_probs = jnp.take_along_axis(
                log_probs, targets[:, :, None], axis=-1
            ).squeeze(-1)
            assert_type(target_log_probs, Array[[B, T]])
            loss = -jnp.mean(target_log_probs)
            assert_type(loss, Array[[]])
            return logits, loss
        else:
            # Inference-time mini-optimization: project only the last token position
            last_token_hidden = x[:, -1][:, None, :]
            assert_type(last_token_hidden, Array[[B, 1, NEmbedding]])
            logits = jnp.matmul(last_token_hidden, tied_weight)
            assert_type(logits, Array[[B, 1, VocabSize]])
            return logits, None

    def crop_block_size[NewBlockSize: IntVar](
        self, block_size: Int[NewBlockSize]
    ) -> GPT[VocabSize, NewBlockSize, NEmbedding, NHead, NLayer]:
        """Functional model surgery returning a new GPT with a smaller block size."""
        assert block_size <= self.config.block_size
        new_config = GPTConfig(
            block_size=block_size,
            vocab_size=self.config.vocab_size,
            n_layer=self.config.n_layer,
            n_head=self.config.n_head,
            n_embd=self.config.n_embd,
            bias=self.config.bias,
        )
        new_wpe = Embedding(weight=self.wpe.weight[:block_size])
        assert_type(new_wpe, Embedding[NewBlockSize, NEmbedding])
        return GPT(
            wte=self.wte,
            wpe=new_wpe,
            h=self.h,
            ln_f=self.ln_f,
            config=new_config,
        )

    @classmethod
    @overload
    def from_pretrained(
        cls, model_type: Literal["gpt2"]
    ) -> GPT[50257, 1024, 768, 12, 12]: ...

    @classmethod
    @overload
    def from_pretrained(
        cls, model_type: Literal["gpt2-medium"]
    ) -> GPT[50257, 1024, 1024, 16, 24]: ...

    @classmethod
    @overload
    def from_pretrained(
        cls, model_type: Literal["gpt2-large"]
    ) -> GPT[50257, 1024, 1280, 20, 36]: ...

    @classmethod
    @overload
    def from_pretrained(
        cls, model_type: Literal["gpt2-xl"]
    ) -> GPT[50257, 1024, 1600, 25, 48]: ...

    @classmethod
    def from_pretrained(
        cls,
        model_type: Literal["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"],
    ) -> Any:
        """Initialize standard GPT-2 architecture presets."""
        if model_type == "gpt2":
            return GPT.init(
                GPTConfig(
                    block_size=1024,
                    vocab_size=50257,
                    n_layer=12,
                    n_head=12,
                    n_embd=768,
                    bias=True,
                )
            )
        elif model_type == "gpt2-medium":
            return GPT.init(
                GPTConfig(
                    block_size=1024,
                    vocab_size=50257,
                    n_layer=24,
                    n_head=16,
                    n_embd=1024,
                    bias=True,
                )
            )
        elif model_type == "gpt2-large":
            return GPT.init(
                GPTConfig(
                    block_size=1024,
                    vocab_size=50257,
                    n_layer=36,
                    n_head=20,
                    n_embd=1280,
                    bias=True,
                )
            )
        else:
            return GPT.init(
                GPTConfig(
                    block_size=1024,
                    vocab_size=50257,
                    n_layer=48,
                    n_head=25,
                    n_embd=1600,
                    bias=True,
                )
            )

    def generate_step[B: IntVar, T: IntVar](
        self,
        idx: Array[[B, T]],
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> Array[[B, T + 1]]:
        """Single autoregressive generation step extending sequence length from T to T + 1."""
        logits_3d, _ = self(idx)
        assert_type(logits_3d, Array[[B, 1, VocabSize]])
        logits = logits_3d[:, -1, :] / temperature
        assert_type(logits, Array[[B, VocabSize]])

        if top_k is not None:
            v, _ = jnp.top_k(logits, min(top_k, logits.shape[-1]))
            threshold: Array[[B, 1]] = v[:, -1:]
            logits = jnp.where(logits < threshold, -jnp.inf, logits)
            assert_type(logits, Array[[B, VocabSize]])

        probs = jax.nn.softmax(logits, axis=-1)
        assert_type(probs, Array[[B, VocabSize]])
        idx_next = jnp.argmax(probs, axis=-1, keepdims=True)
        assert_type(idx_next, Array[[B, 1]])
        next_seq = jnp.concatenate((idx, idx_next), axis=1)
        if TYPE_CHECKING:
            assert_type(next_seq, Array[[B, T + 1]])
        return next_seq

    def generate[B: IntVar](
        self,
        idx: Array[[B, Any]],
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> Array[[B, Any]]:
        """Autoregressively generate `max_new_tokens` tokens."""
        for _ in range(max_new_tokens):
            idx_cond = (
                idx
                if idx.shape[1] <= self.config.block_size
                else idx[:, -self.config.block_size :]
            )
            assert_type(idx_cond, Array[[B, Any]])
            idx = self.generate_step(idx_cond, temperature=temperature, top_k=top_k)
            assert_type(idx, Array[[B, Any]])
        return idx


# ============================================================================
# Smoke tests
# ============================================================================


def test_nanogpt_forward_inference() -> None:
    """Test NanoGPT forward pass in inference mode (targets=None)."""
    config = GPTConfig(
        block_size=64,
        vocab_size=256,
        n_layer=2,
        n_head=4,
        n_embd=128,
        bias=True,
    )
    assert_type(config, GPTConfig[256, 64, 128, 4, 2])

    model = GPT.init(config)
    assert_type(model, GPT[256, 64, 128, 4, 2])

    idx: Array[[3, 16]] = jnp.ones((3, 16), dtype=jnp.int32)
    logits, loss = model(idx)
    assert_type(logits, Array[[3, 1, 256]])
    assert_type(loss, None)
    assert_shape(logits.shape, (3, 1, 256))
    assert loss is None


def test_nanogpt_forward_training() -> None:
    """Test NanoGPT forward pass in training mode (with targets)."""
    config = GPTConfig(
        block_size=64,
        vocab_size=256,
        n_layer=2,
        n_head=4,
        n_embd=128,
        bias=False,
    )
    model = GPT.init(config)

    idx: Array[[3, 16]] = jnp.ones((3, 16), dtype=jnp.int32)
    targets: Array[[3, 16]] = jnp.zeros((3, 16), dtype=jnp.int32)

    logits, loss = model(idx, targets)
    assert_type(logits, Array[[3, 16, 256]])
    assert_type(loss, Array[[]])
    assert_shape(logits.shape, (3, 16, 256))
    assert_shape(loss.shape, ())


def test_nanogpt_generation_and_cropping() -> None:
    """Test NanoGPT single-step generation, multi-step generation, top_k, and block size cropping."""
    config = GPTConfig(
        block_size=32,
        vocab_size=100,
        n_layer=2,
        n_head=4,
        n_embd=64,
        bias=True,
    )
    model = GPT.init(config)

    # 1) Direct top_k on logits verifies exact static top-k shape computation
    idx: Array[[2, 10]] = jnp.ones((2, 10), dtype=jnp.int32)
    logits, _ = model(idx)
    top_vals, top_indices = jnp.top_k(logits[:, -1, :], 5)
    assert_type(top_vals, Array[[2, 5]])
    assert_type(top_indices, Array[[2, 5]])
    assert_shape(top_vals.shape, (2, 5))

    # 2) Single step generation verifies exact symbolic T + 1 shape arithmetic
    step_out = model.generate_step(idx, temperature=0.8, top_k=5)
    assert_type(step_out, Array[[2, 11]])
    assert_shape(step_out.shape, (2, 11))

    # 3) Multi-step generation
    gen_out = model.generate(idx, max_new_tokens=4, temperature=1.0, top_k=5)
    assert_type(gen_out, Array[[2, Any]])
    # The loop-carried sequence length is intentionally gradual.
    assert_shape(gen_out.shape, (2, int), runtime=(2, 14))

    # 4) Functional crop_block_size updates static block size type parameter
    cropped_model = model.crop_block_size(16)
    assert_type(cropped_model, GPT[100, 16, 64, 4, 2])
    cropped_logits, _ = cropped_model(idx)
    assert_type(cropped_logits, Array[[2, 1, 100]])
    assert_shape(cropped_logits.shape, (2, 1, 100))


def test_nanogpt_pytree_transformations() -> None:
    """Test JAX PyTree transformations (tree.leaves, jit, grad, vmap) on NanoGPT."""
    config = GPTConfig(
        block_size=16,
        vocab_size=50,
        n_layer=2,
        n_head=2,
        n_embd=32,
        bias=True,
    )
    model = GPT.init(config)
    leaves = jax.tree.leaves(model)
    assert len(leaves) > 0
    assert model.get_num_params() > 0

    idx: Array[[2, 8]] = jnp.ones((2, 8), dtype=jnp.int32)
    targets: Array[[2, 8]] = jnp.zeros((2, 8), dtype=jnp.int32)

    def loss_fn(
        gpt: GPT[50, 16, 32, 2, 2], x: Array[[2, 8]], y: Array[[2, 8]]
    ) -> Array[[]]:
        _, loss = gpt(x, y)
        assert_type(loss, Array[[]])
        return loss

    jit_loss = jax.jit(loss_fn)(model, idx, targets)
    assert_type(jit_loss, Array[[]])
    assert_shape(jit_loss.shape, ())

    grads: GPT[50, 16, 32, 2, 2] = jax.grad(loss_fn)(model, idx, targets)
    grad_leaves = jax.tree.leaves(grads)
    assert len(grad_leaves) == len(leaves)
    assert_type(grads.wte.weight, Array[[50, 32]])
    assert_shape(grads.wte.weight.shape, (50, 32))
    assert model.get_num_params(non_embedding=False) == sum(p.size for p in leaves)

    # Test vmap over a batch of sequences
    batch_idx: Array[[3, 2, 8]] = jnp.ones((3, 2, 8), dtype=jnp.int32)
    vmap_logits: Array[[3, 2, 1, 50]] = jax.vmap(lambda x_i: model(x_i)[0])(batch_idx)
    assert_shape(vmap_logits.shape, (3, 2, 1, 50))
