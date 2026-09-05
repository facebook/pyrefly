# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
BERT model in JAX with shape annotations.

Equivalent to the PyTorch BERT example, implemented using dataclasses
registered as JAX PyTrees via `@jax.tree_util.register_dataclass` and
`jax.tree.static()` without external framework dependencies.

Simplifications relative to the original BERT specification (Devlin et al., 2018):
- Pre-LayerNorm: Sublayer connections apply LayerNorm before the sublayer (Pre-LN)
  without a final LayerNorm before the prediction heads, matching the PyTorch example
  (`codertimo/BERT-pytorch`) rather than original BERT's Post-LN.
- Sinusoidal Positional Embeddings: Fixed sinusoidal encodings (Vaswani et al., 2017)
  are used instead of learned positional embedding tables.
- Omission of Dropout: Stochastic dropout is omitted to keep the functional API simple
  and deterministic without explicit PRNG key threading.
- Deterministic Weight Initialization: Linear and embedding layers use deterministic
  constant weights (`jnp.full`, `jnp.ones`) so smoke tests run without PRNG keys.
- Direct NSP Head: NextSentencePrediction projects the `[CLS]` token directly to 2
  classes without the intermediate `tanh` pooling projection.
- LogSoftmax Outputs: Prediction heads return log-probabilities via `log_softmax`
  rather than unnormalized logits.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, assert_type, overload, TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax import Array
from shape_extensions import assert_shape, Int, IntVar

# ============================================================================
# Core Layers: Linear, LayerNorm, SublayerConnection, FeedForward
# ============================================================================


@jax.tree_util.register_dataclass
@dataclass
class Linear[In: IntVar, Out: IntVar]:
    weight: Array[[In, Out]]
    bias: Array[[Out]]

    @classmethod
    def init(cls, in_features: Int[In], out_features: Int[Out]) -> Linear[In, Out]:
        # Deterministic initialization avoids PRNG key plumbing in shape tests.
        scale = 1.0 / math.sqrt(int(in_features))
        return cls(
            weight=jnp.full((in_features, out_features), scale),
            bias=jnp.zeros(out_features),
        )

    @overload
    def __call__[B: IntVar](self, x: Array[[B, In]]) -> Array[[B, Out]]: ...
    @overload
    def __call__[B: IntVar, T: IntVar](
        self, x: Array[[B, T, In]]
    ) -> Array[[B, T, Out]]: ...
    def __call__(self, x: Any) -> Any:
        return jnp.matmul(x, self.weight) + self.bias


@jax.tree_util.register_dataclass
@dataclass
class LayerNorm[Features: IntVar]:
    gamma: Array[[Features]]
    beta: Array[[Features]]
    eps: float = jax.tree.static(default=1e-6)

    @classmethod
    def init(cls, features: Int[Features], eps: float = 1e-6) -> LayerNorm[Features]:
        return cls(
            gamma=jnp.ones(features),
            beta=jnp.zeros(features),
            eps=eps,
        )

    def __call__[B: IntVar, T: IntVar](
        self, x: Array[[B, T, Features]]
    ) -> Array[[B, T, Features]]:
        mean = jnp.mean(x, axis=-1, keepdims=True)
        assert_type(mean, Array[[B, T, 1]])
        variance = jnp.var(x, axis=-1, keepdims=True)
        assert_type(variance, Array[[B, T, 1]])
        return self.gamma * (x - mean) / jnp.sqrt(variance + self.eps) + self.beta


@jax.tree_util.register_dataclass
@dataclass
class SublayerConnection[Hidden: IntVar]:
    """Residual connection with Pre-LayerNorm (LayerNorm before sublayer)."""

    norm: LayerNorm[Hidden]

    @classmethod
    def init(cls, size: Int[Hidden]) -> SublayerConnection[Hidden]:
        return cls(norm=LayerNorm.init(size))

    def __call__[B: IntVar, T: IntVar](
        self,
        x: Array[[B, T, Hidden]],
        sublayer: Callable[[Array[[B, T, Hidden]]], Array[[B, T, Hidden]]],
    ) -> Array[[B, T, Hidden]]:
        """Apply residual connection to any sublayer with the same size."""
        return x + sublayer(self.norm(x))


@jax.tree_util.register_dataclass
@dataclass
class PositionwiseFeedForward[DModel: IntVar, DFF: IntVar]:
    w_1: Linear[DModel, DFF]
    w_2: Linear[DFF, DModel]

    @classmethod
    def init(
        cls, d_model: Int[DModel], d_ff: Int[DFF]
    ) -> PositionwiseFeedForward[DModel, DFF]:
        return cls(
            w_1=Linear.init(d_model, d_ff),
            w_2=Linear.init(d_ff, d_model),
        )

    def __call__[B: IntVar, T: IntVar](
        self, x: Array[[B, T, DModel]]
    ) -> Array[[B, T, DModel]]:
        h = self.w_1(x)
        assert_type(h, Array[[B, T, DFF]])
        h = jax.nn.gelu(h)
        assert_type(h, Array[[B, T, DFF]])
        out = self.w_2(h)
        assert_type(out, Array[[B, T, DModel]])
        return out


# ============================================================================
# Attention
# ============================================================================


@jax.tree_util.register_dataclass
@dataclass
class Attention:
    """Compute Scaled Dot Product Attention."""

    def __call__[B: IntVar, H: IntVar, T: IntVar, DK: IntVar](
        self,
        query: Array[[B, H, T, DK]],
        key: Array[[B, H, T, DK]],
        value: Array[[B, H, T, DK]],
        mask: Array | None = None,
    ) -> tuple[Array[[B, H, T, DK]], Array[[B, H, T, T]]]:
        d_k = query.shape[-1]
        scores = jnp.matmul(query, key.swapaxes(-2, -1)) / math.sqrt(d_k)
        assert_type(scores, Array[[B, H, T, T]])

        if mask is not None:
            scores = jnp.where(mask == 0, -1e9, scores)

        p_attn = jax.nn.softmax(scores, axis=-1)
        assert_type(p_attn, Array[[B, H, T, T]])

        out = jnp.matmul(p_attn, value)
        assert_type(out, Array[[B, H, T, DK]])
        return out, p_attn


@jax.tree_util.register_dataclass
@dataclass
class MultiHeadedAttention[DModel: IntVar, H: IntVar]:
    q_linear: Linear[DModel, DModel]
    k_linear: Linear[DModel, DModel]
    v_linear: Linear[DModel, DModel]
    output_linear: Linear[DModel, DModel]
    h: Int[H] = jax.tree.static()
    d_model: Int[DModel] = jax.tree.static()

    @classmethod
    def init(cls, h: Int[H], d_model: Int[DModel]) -> MultiHeadedAttention[DModel, H]:
        assert d_model % h == 0
        return cls(
            q_linear=Linear.init(d_model, d_model),
            k_linear=Linear.init(d_model, d_model),
            v_linear=Linear.init(d_model, d_model),
            output_linear=Linear.init(d_model, d_model),
            h=h,
            d_model=d_model,
        )

    def __call__[B: IntVar, T: IntVar](
        self,
        query: Array[[B, T, DModel]],
        key: Array[[B, T, DModel]],
        value: Array[[B, T, DModel]],
        mask: Array | None = None,
    ) -> Array[[B, T, DModel]]:
        batch_size = query.shape[0]
        seq_len = query.shape[1]
        assert_type(batch_size, Int[B])
        assert_type(seq_len, Int[T])
        d_k = self.d_model // self.h

        # 1) Linear projections and reshape to (B, H, T, d_k)
        query_p = (
            self.q_linear(query)
            .reshape(batch_size, seq_len, self.h, d_k)
            .swapaxes(1, 2)
        )
        if TYPE_CHECKING:
            assert_type(query_p, Array[[B, H, T, (DModel // H)]])
        key_p = (
            self.k_linear(key).reshape(batch_size, seq_len, self.h, d_k).swapaxes(1, 2)
        )
        value_p = (
            self.v_linear(value)
            .reshape(batch_size, seq_len, self.h, d_k)
            .swapaxes(1, 2)
        )

        # 2) Attention
        attn_out, _ = Attention()(query_p, key_p, value_p, mask=mask)
        if TYPE_CHECKING:
            assert_type(attn_out, Array[[B, H, T, (DModel // H)]])

        # 3) Concat heads and apply final linear
        x = attn_out.swapaxes(1, 2).reshape(batch_size, seq_len, self.d_model)
        assert_type(x, Array[[B, T, DModel]])
        return self.output_linear(x)


# ============================================================================
# Embeddings
# ============================================================================


@jax.tree_util.register_dataclass
@dataclass
class Embedding[VocabSize: IntVar, EmbedSize: IntVar]:
    weight: Array[[VocabSize, EmbedSize]]

    @classmethod
    def init(
        cls, vocab_size: Int[VocabSize], embed_size: Int[EmbedSize]
    ) -> Embedding[VocabSize, EmbedSize]:
        # Deterministic initialization avoids PRNG key plumbing in shape tests.
        return cls(weight=jnp.ones((vocab_size, embed_size)))

    def __call__[B: IntVar, T: IntVar](
        self, x: Array[[B, T]]
    ) -> Array[[B, T, EmbedSize]]:
        return self.weight[x]


@jax.tree_util.register_dataclass
@dataclass
class PositionalEmbedding[EmbedSize: IntVar]:
    """Sinusoidal positional encodings (Vaswani et al., 2017)."""

    pe: Array[[Any, EmbedSize]]

    @classmethod
    def init(
        cls, d_model: Int[EmbedSize], max_len: int = 512
    ) -> PositionalEmbedding[EmbedSize]:
        # Compute sinusoidal positional encodings in log space.
        position = jnp.arange(max_len)[:, None]
        div_term = jnp.exp(jnp.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        angles = position * div_term
        sin_enc = jnp.sin(angles)
        cos_enc = jnp.cos(angles)
        pe = jnp.stack([sin_enc, cos_enc], axis=2).reshape(max_len, d_model)
        return cls(pe=pe)

    def __call__[B: IntVar, T: IntVar](self, x: Array[[B, T]]) -> Array[[T, EmbedSize]]:
        seq_len = x.shape[1]
        return self.pe[:seq_len]


@jax.tree_util.register_dataclass
@dataclass
class BERTEmbedding[VocabSize: IntVar, EmbedSize: IntVar]:
    token: Embedding[VocabSize, EmbedSize]
    position: PositionalEmbedding[EmbedSize]
    segment: Embedding[3, EmbedSize]
    embed_size: Int[EmbedSize] = jax.tree.static()

    @classmethod
    def init(
        cls,
        vocab_size: Int[VocabSize],
        embed_size: Int[EmbedSize],
        max_len: int = 512,
    ) -> BERTEmbedding[VocabSize, EmbedSize]:
        return cls(
            token=Embedding.init(vocab_size, embed_size),
            position=PositionalEmbedding.init(embed_size, max_len),
            segment=Embedding.init(3, embed_size),
            embed_size=embed_size,
        )

    def __call__[B: IntVar, T: IntVar](
        self, sequence: Array[[B, T]], segment_label: Array[[B, T]]
    ) -> Array[[B, T, EmbedSize]]:
        x = self.token(sequence) + self.position(sequence) + self.segment(segment_label)
        return x


# ============================================================================
# Transformer Block
# ============================================================================


@jax.tree_util.register_dataclass
@dataclass
class TransformerBlock[Hidden: IntVar, H: IntVar]:
    attention: MultiHeadedAttention[Hidden, H]
    feed_forward: PositionwiseFeedForward[Hidden, Any]
    input_sublayer: SublayerConnection[Hidden]
    output_sublayer: SublayerConnection[Hidden]

    @classmethod
    def init(
        cls,
        hidden: Int[Hidden],
        attn_heads: Int[H],
        feed_forward_hidden: int,
    ) -> TransformerBlock[Hidden, H]:
        return cls(
            attention=MultiHeadedAttention.init(attn_heads, hidden),
            feed_forward=PositionwiseFeedForward.init(hidden, feed_forward_hidden),
            input_sublayer=SublayerConnection.init(size=hidden),
            output_sublayer=SublayerConnection.init(size=hidden),
        )

    def __call__[B: IntVar, T: IntVar](
        self, x: Array[[B, T, Hidden]], mask: Array | None = None
    ) -> Array[[B, T, Hidden]]:
        x = self.input_sublayer(x, lambda x_: self.attention(x_, x_, x_, mask=mask))
        assert_type(x, Array[[B, T, Hidden]])
        x = self.output_sublayer(x, self.feed_forward)
        assert_type(x, Array[[B, T, Hidden]])
        return x


# ============================================================================
# BERT Model
# ============================================================================


@jax.tree_util.register_dataclass
@dataclass
class BERT[VocabSize: IntVar, Hidden: IntVar = 768, H: IntVar = 12]:
    """BERT model: Bidirectional Encoder Representations from Transformers."""

    embedding: BERTEmbedding[VocabSize, Hidden]
    transformer_blocks: list[TransformerBlock[Hidden, H]]
    hidden: Int[Hidden] = jax.tree.static(default=768)
    attn_heads: Int[H] = jax.tree.static(default=12)

    @classmethod
    def init[V: IntVar, Hid: IntVar = 768, Heads: IntVar = 12](
        cls,
        vocab_size: Int[V],
        hidden: Int[Hid] = 768,
        n_layers: int = 12,
        attn_heads: Int[Heads] = 12,
    ) -> BERT[V, Hid, Heads]:
        feed_forward_hidden = hidden * 4
        embedding = BERTEmbedding.init(vocab_size, hidden)
        transformer_blocks = [
            TransformerBlock.init(hidden, attn_heads, feed_forward_hidden)
            for _ in range(n_layers)
        ]
        return cls(
            embedding=embedding,
            transformer_blocks=transformer_blocks,
            hidden=hidden,
            attn_heads=attn_heads,
        )

    def __call__[B: IntVar, T: IntVar](
        self, x: Array[[B, T]], segment_info: Array[[B, T]]
    ) -> Array[[B, T, Hidden]]:
        mask = (x > 0)[:, None, None, :]
        x_emb = self.embedding(x, segment_info)
        assert_type(x_emb, Array[[B, T, Hidden]])

        for transformer in self.transformer_blocks:
            x_emb = transformer(x_emb, mask)
        assert_type(x_emb, Array[[B, T, Hidden]])
        return x_emb


# ============================================================================
# Language Model Heads
# ============================================================================


@jax.tree_util.register_dataclass
@dataclass
class NextSentencePrediction[Hidden: IntVar]:
    """Binary classifier: is_next vs. is_not_next.

    Directly projects the [CLS] token representation to 2 classes without
    the intermediate tanh pooling layer used in the original BERT paper.
    """

    linear: Linear[Hidden, 2]

    @classmethod
    def init(cls, hidden: Int[Hidden]) -> NextSentencePrediction[Hidden]:
        return cls(linear=Linear.init(hidden, 2))

    def __call__[B: IntVar, T: IntVar](self, x: Array[[B, T, Hidden]]) -> Array[[B, 2]]:
        first_token = x[:, 0]
        assert_type(first_token, Array[[B, Hidden]])
        # Returns log-probabilities over the 2 classes.
        return jax.nn.log_softmax(self.linear(first_token), axis=-1)


@jax.tree_util.register_dataclass
@dataclass
class MaskedLanguageModel[Hidden: IntVar, VocabSize: IntVar]:
    """Masked language modeling head predicting tokens over the vocabulary."""

    linear: Linear[Hidden, VocabSize]

    @classmethod
    def init(
        cls, hidden: Int[Hidden], vocab_size: Int[VocabSize]
    ) -> MaskedLanguageModel[Hidden, VocabSize]:
        return cls(linear=Linear.init(hidden, vocab_size))

    def __call__[B: IntVar, T: IntVar](
        self, x: Array[[B, T, Hidden]]
    ) -> Array[[B, T, VocabSize]]:
        # Returns log-probabilities over the vocabulary.
        return jax.nn.log_softmax(self.linear(x), axis=-1)


@jax.tree_util.register_dataclass
@dataclass
class BERTLM[VocabSize: IntVar, Hidden: IntVar, H: IntVar]:
    bert: BERT[VocabSize, Hidden, H]
    next_sentence: NextSentencePrediction[Hidden]
    mask_lm: MaskedLanguageModel[Hidden, VocabSize]

    @classmethod
    def init[V: IntVar, Hid: IntVar, Heads: IntVar](
        cls, bert: BERT[V, Hid, Heads], vocab_size: Int[V]
    ) -> BERTLM[V, Hid, Heads]:
        return cls(
            bert=bert,
            next_sentence=NextSentencePrediction.init(bert.hidden),
            mask_lm=MaskedLanguageModel.init(bert.hidden, vocab_size),
        )

    def __call__[B: IntVar, T: IntVar](
        self, x: Array[[B, T]], segment_label: Array[[B, T]]
    ) -> tuple[Array[[B, 2]], Array[[B, T, VocabSize]]]:
        x_out = self.bert(x, segment_label)
        assert_type(x_out, Array[[B, T, Hidden]])
        nsp = self.next_sentence(x_out)
        assert_type(nsp, Array[[B, 2]])
        mlm = self.mask_lm(x_out)
        assert_type(mlm, Array[[B, T, VocabSize]])
        return nsp, mlm


# ============================================================================
# Smoke tests
# ============================================================================


def test_bert_model() -> None:
    """Test BERT encoder produces correct output shape."""
    bert = BERT.init(vocab_size=30522, hidden=256, n_layers=2, attn_heads=8)
    assert_type(bert, BERT[30522, 256, 8])

    x: Array[[4, 128]] = jnp.ones((4, 128), dtype=jnp.int32)
    segment: Array[[4, 128]] = jnp.zeros((4, 128), dtype=jnp.int32)

    out = bert(x, segment)
    assert_shape(out, (4, 128, 256))
    assert_type(out, Array[[4, 128, 256]])


def test_bert_default_hidden() -> None:
    """Test BERT with default hidden=768 and attn_heads=12."""
    bert = BERT.init(vocab_size=30522, n_layers=2, attn_heads=12)
    assert_type(bert, BERT[30522, 768, 12])

    x: Array[[4, 128]] = jnp.ones((4, 128), dtype=jnp.int32)
    segment: Array[[4, 128]] = jnp.zeros((4, 128), dtype=jnp.int32)

    out = bert(x, segment)
    assert_shape(out, (4, 128, 768))
    assert_type(out, Array[[4, 128, 768]])


def test_bert_lm() -> None:
    """Test BERT Language Model with both heads."""
    bert = BERT.init(vocab_size=30522, hidden=256, n_layers=2, attn_heads=8)
    model = BERTLM.init(bert, vocab_size=30522)

    x: Array[[4, 128]] = jnp.ones((4, 128), dtype=jnp.int32)
    segment: Array[[4, 128]] = jnp.zeros((4, 128), dtype=jnp.int32)

    nsp_out, mlm_out = model(x, segment)
    assert_shape(nsp_out, (4, 2))
    assert_type(nsp_out, Array[[4, 2]])
    assert_shape(mlm_out, (4, 128, 30522))
    assert_type(mlm_out, Array[[4, 128, 30522]])


def test_bert_pytree_transformations() -> None:
    """Test JAX PyTree transformations (tree_leaves, jit, grad, vmap)."""
    bert = BERT.init(vocab_size=1000, hidden=64, n_layers=2, attn_heads=4)
    leaves = jax.tree.leaves(bert)
    assert len(leaves) > 0

    x: Array[[2, 16]] = jnp.ones((2, 16), dtype=jnp.int32)
    segment: Array[[2, 16]] = jnp.zeros((2, 16), dtype=jnp.int32)

    def loss_fn(
        model: BERT[1000, 64, 4], x_: Array[[2, 16]], seg_: Array[[2, 16]]
    ) -> Array[[]]:
        loss = jnp.sum(model(x_, seg_))
        assert_type(loss, Array[[]])
        return loss

    jit_loss = jax.jit(loss_fn)(bert, x, segment)
    assert_shape(jit_loss, ())

    grad = jax.grad(loss_fn)(bert, x, segment)
    grad_leaves = jax.tree.leaves(grad)
    assert len(grad_leaves) == len(leaves)

    # Test vmap over batch of inputs with model as static unmapped argument
    batch_x = jnp.ones((3, 2, 16), dtype=jnp.int32)
    batch_seg = jnp.zeros((3, 2, 16), dtype=jnp.int32)
    vmap_out: Array[[3, 2, 16, 64]] = jax.vmap(lambda x_i, s_i: bert(x_i, s_i))(
        batch_x, batch_seg
    )
    assert_shape(vmap_out, (3, 2, 16, 64))
