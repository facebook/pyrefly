# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
BERT from TorchBenchmark with shape annotations.

Original: pytorch/benchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/
See model_port_changes.md for full change analysis.
"""

import math
from collections.abc import Callable
from typing import assert_type, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from torch import Tensor
    from torch_shapes import Dim


# ============================================================================
# Utils: LayerNorm, SublayerConnection, PositionwiseFeedForward
# ============================================================================


class LayerNorm[Features](nn.Module):
    """Construct a layernorm module (See citation for details)."""

    a_2: Tensor[Features]
    b_2: Tensor[Features]

    def __init__(self, features: Dim[Features], eps: float = 1e-6) -> None:
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward[*Bs](self, x: Tensor[*Bs, Features]) -> Tensor[*Bs, Features]:
        mean = x.mean(-1, keepdim=True)
        assert_type(mean, Tensor[*Bs, 1])
        std = x.std(-1, keepdim=True)
        assert_type(std, Tensor[*Bs, 1])
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection[Hidden](nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size: Dim[Hidden], dropout: float) -> None:
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward[B, T](
        self,
        x: Tensor[B, T, Hidden],
        sublayer: Callable[[Tensor[B, T, Hidden]], Tensor[B, T, Hidden]],
    ) -> Tensor[B, T, Hidden]:
        """Apply residual connection to any sublayer with the same size."""
        return x + self.dropout(sublayer(self.norm(x)))


class PositionwiseFeedForward[DModel, DFF](nn.Module):
    """Implements FFN equation."""

    def __init__(
        self, d_model: Dim[DModel], d_ff: Dim[DFF], dropout: float = 0.1
    ) -> None:
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward[B, T](self, x: Tensor[B, T, DModel]) -> Tensor[B, T, DModel]:
        h = self.w_1(x)
        assert_type(h, Tensor[B, T, DFF])
        h = self.activation(h)
        assert_type(h, Tensor[B, T, DFF])
        h = self.dropout(h)
        h = self.w_2(h)
        assert_type(h, Tensor[B, T, DModel])
        return h


# ============================================================================
# Attention
# ============================================================================


class Attention(nn.Module):
    """Compute 'Scaled Dot Product Attention'"""

    def forward[B, H, T, DK](
        self,
        query: Tensor[B, H, T, DK],
        key: Tensor[B, H, T, DK],
        value: Tensor[B, H, T, DK],
        mask: Tensor | None = None,
        dropout: nn.Dropout | None = None,
    ) -> tuple[Tensor[B, H, T, DK], Tensor[B, H, T, T]]:
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention[DModel, H](nn.Module):
    """Take in model size and number of heads."""

    def __init__(self, h: Dim[H], d_model: Dim[DModel], dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.d_model = d_model

        self.linear_layers = nn.ModuleList(
            [nn.Linear(d_model, d_model) for _ in range(3)]
        )
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()
        self.dropout = nn.Dropout(p=dropout)

    def forward[B, T](
        self,
        query: Tensor[B, T, DModel],
        key: Tensor[B, T, DModel],
        value: Tensor[B, T, DModel],
        mask: Tensor | None = None,
    ) -> Tensor[B, T, DModel]:
        batch_size = query.size(0)
        seq_len = query.size(1)
        assert_type(batch_size, Dim[B])
        assert_type(seq_len, Dim[T])

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query_p = (
            self.linear_layers[0](query)
            .view(batch_size, seq_len, self.h, self.d_k)
            .transpose(1, 2)
        )
        assert_type(query_p, Tensor[B, H, T, (DModel // H)])
        key_p = (
            self.linear_layers[1](key)
            .view(batch_size, seq_len, self.h, self.d_k)
            .transpose(1, 2)
        )
        value_p = (
            self.linear_layers[2](value)
            .view(batch_size, seq_len, self.h, self.d_k)
            .transpose(1, 2)
        )

        # 2) Apply attention on all the projected vectors in batch.
        attn_out, attn = self.attention(
            query_p, key_p, value_p, mask=mask, dropout=self.dropout
        )
        assert_type(attn_out, Tensor[B, H, T, (DModel // H)])

        # 3) "Concat" using a view and apply a final linear.
        x = (
            attn_out.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.d_model)
        )
        assert_type(x, Tensor[B, T, DModel])

        return self.output_linear(x)


# ============================================================================
# Embeddings
# ============================================================================


class TokenEmbedding[VocabSize, EmbedSize = 512](nn.Embedding[VocabSize, EmbedSize]):
    def __init__(
        self, vocab_size: Dim[VocabSize], embed_size: Dim[EmbedSize] = 512
    ) -> None:
        super().__init__(vocab_size, embed_size, padding_idx=0)


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()

        # Compute the positional encodings once in log space.
        pe: Tensor = torch.zeros(max_len, d_model).float()

        position: Tensor = torch.arange(0, max_len).float().unsqueeze(1)
        div_term: Tensor = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.pe = nn.Buffer(pe)

    def forward(self, x: Tensor) -> Tensor:
        return self.pe[:, : x.size(1)]


class SegmentEmbedding(nn.Embedding):
    def __init__(self, embed_size: int = 512) -> None:
        super().__init__(3, embed_size, padding_idx=0)


class BERTEmbedding[VocabSize, EmbedSize](nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)

        sum of all these features are output of BERTEmbedding
    """

    def __init__(
        self,
        vocab_size: Dim[VocabSize],
        embed_size: Dim[EmbedSize],
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        self.position = PositionalEmbedding(d_model=embed_size)
        self.segment = SegmentEmbedding(embed_size=embed_size)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def forward[B, T](
        self, sequence: Tensor[B, T], segment_label: Tensor[B, T]
    ) -> Tensor[B, T, EmbedSize]:
        x = self.token(sequence) + self.position(sequence) + self.segment(segment_label)
        return self.dropout(x)


# ============================================================================
# Transformer Block
# ============================================================================


class SelfAttentionWrapper[DModel, H](nn.Module):
    """Wraps MultiHeadedAttention for self-attention (Q=K=V=x).

    The original uses a LambdaModule (TorchScript JIT wrapper) to convert the
    multi-arg attention call into a single-arg callable for SublayerConnection.
    We use a simple module wrapper instead.
    """

    def __init__(self, attention: MultiHeadedAttention[DModel, H]) -> None:
        super().__init__()
        self.attention = attention
        self.mask: Tensor | None = None

    def set_mask(self, mask: Tensor) -> None:
        self.mask = mask

    def forward[B, T](self, x: Tensor[B, T, DModel]) -> Tensor[B, T, DModel]:
        return self.attention(x, x, x, mask=self.mask)


class TransformerBlock[Hidden, H](nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(
        self,
        hidden: Dim[Hidden],
        attn_heads: Dim[H],
        feed_forward_hidden: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.self_attn = SelfAttentionWrapper(self.attention)
        self.feed_forward = PositionwiseFeedForward(
            d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout
        )
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward[B, T](
        self, x: Tensor[B, T, Hidden], mask: Tensor
    ) -> Tensor[B, T, Hidden]:
        self.self_attn.set_mask(mask)
        x = self.input_sublayer(x, self.self_attn)
        assert_type(x, Tensor[B, T, Hidden])
        x = self.output_sublayer(x, self.feed_forward)
        assert_type(x, Tensor[B, T, Hidden])
        return self.dropout(x)


# ============================================================================
# BERT Model
# ============================================================================


class BERT[VocabSize, Hidden = 768, H = 12](nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(
        self,
        vocab_size: Dim[VocabSize],
        hidden: Dim[Hidden] = 768,
        n_layers: int = 12,
        attn_heads: Dim[H] = 12,
        dropout: float = 0.1,
    ) -> None:
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """
        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden * 4

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=hidden)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(hidden, attn_heads, hidden * 4, dropout)
                for _ in range(n_layers)
            ]
        )

    def forward[B, T](
        self, x: Tensor[B, T], segment_info: Tensor[B, T]
    ) -> Tensor[B, T, Hidden]:
        # attention masking for padded token
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
        mask: Tensor = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        x_emb = self.embedding(x, segment_info)
        assert_type(x_emb, Tensor[B, T, Hidden])

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x_emb = transformer(x_emb, mask)
        assert_type(x_emb, Tensor[B, T, Hidden])

        return x_emb


# ============================================================================
# Language Model Heads
# ============================================================================


class NextSentencePrediction[Hidden](nn.Module):
    """
    2-class classification model : is_next, is_not_next
    """

    def __init__(self, hidden: Dim[Hidden]) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden, 2)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward[B, T](self, x: Tensor[B, T, Hidden]) -> Tensor[B, 2]:
        return self.softmax(self.linear(x[:, 0]))


class MaskedLanguageModel[Hidden, VocabSize](nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """

    def __init__(self, hidden: Dim[Hidden], vocab_size: Dim[VocabSize]) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward[B, T](self, x: Tensor[B, T, Hidden]) -> Tensor[B, T, VocabSize]:
        return self.softmax(self.linear(x))


class BERTLM[VocabSize, Hidden, H](nn.Module):
    """
    BERT Language Model
    Next Sentence Prediction Model + Masked Language Model
    """

    def __init__(
        self, bert: BERT[VocabSize, Hidden, H], vocab_size: Dim[VocabSize]
    ) -> None:
        super().__init__()
        self.bert = bert
        self.next_sentence = NextSentencePrediction(bert.hidden)
        self.mask_lm = MaskedLanguageModel(bert.hidden, vocab_size)

    def forward[B, T](
        self, x: Tensor[B, T], segment_label: Tensor[B, T]
    ) -> tuple[Tensor[B, 2], Tensor[B, T, VocabSize]]:
        x_out = self.bert(x, segment_label)
        assert_type(x_out, Tensor[B, T, Hidden])
        nsp = self.next_sentence(x_out)
        assert_type(nsp, Tensor[B, 2])
        mlm = self.mask_lm(x_out)
        assert_type(mlm, Tensor[B, T, VocabSize])
        return nsp, mlm


# ============================================================================
# Smoke tests
# ============================================================================


def test_bert_model():
    """Test BERT encoder produces correct output shape."""
    bert = BERT(vocab_size=30522, hidden=256, n_layers=2, attn_heads=8)
    assert_type(bert, BERT[30522, 256, 8])

    x: Tensor[4, 128] = torch.randint(0, 30522, (4, 128))
    segment: Tensor[4, 128] = torch.zeros(4, 128).long()

    out = bert(x, segment)
    assert_type(out, Tensor[4, 128, 256])


def test_bert_default_hidden():
    """Test BERT with default hidden=768 (PEP 696 TypeVar default)."""
    bert = BERT(vocab_size=30522, n_layers=2, attn_heads=12)
    assert_type(bert, BERT[30522, 768])

    x: Tensor[4, 128] = torch.randint(0, 30522, (4, 128))
    segment: Tensor[4, 128] = torch.zeros(4, 128).long()

    out = bert(x, segment)
    assert_type(out, Tensor[4, 128, 768])


def test_bert_lm():
    """Test BERT Language Model with both heads."""
    bert = BERT(vocab_size=30522, hidden=256, n_layers=2, attn_heads=8)
    model = BERTLM(bert, vocab_size=30522)

    x: Tensor[4, 128] = torch.randint(0, 30522, (4, 128))
    segment: Tensor[4, 128] = torch.zeros(4, 128).long()

    nsp_out, mlm_out = model(x, segment)
    assert_type(nsp_out, Tensor[4, 2])
    assert_type(mlm_out, Tensor[4, 128, 30522])
