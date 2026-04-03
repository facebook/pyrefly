# Portions (c) Meta Platforms, Inc. and affiliates.
#
# This source code is adapted from pytorch/benchmark (TorchBenchmark),
# which is licensed under the BSD 3-Clause License:
# https://github.com/pytorch/benchmark/blob/main/LICENSE
#
# This adaptation adds tensor shape type annotations for pyrefly.

"""
DLRM (Deep Learning Recommendation Model) from TorchBenchmark with shape annotations.

Original: ai_codesign/benchmarks/dlrm/dlrm_s_pytorch.py

Port notes:
- DLRM processes dense (continuous) and sparse (categorical) features separately,
  then combines them via feature interaction.
- Dense features go through a "bottom MLP": (B, DenseDim) → (B, D).
- Sparse features go through N EmbeddingBag tables, each producing (B, D).
- Feature interaction: stack dense + sparse → (B, N+1, D) → BMM → (B, N+1, N+1)
  → extract upper triangle → (B, NumInt) → concat with dense → (B, D + NumInt)
  → top MLP → (B, 1) → sigmoid.
- The original constructs MLPs dynamically from layer-size arrays. This port
  spells out layers explicitly for a concrete configuration:
  - DenseDim=13, D=64, 3 embedding tables
  - Bottom MLP: 13 → 512 → 256 → 64
  - Top MLP: 70 → 512 → 256 → 1  (70 = 64 + 6, where 6 = (3+1)*3//2)
- EmbeddingBag returns unrefined Tensor (batch dim from offsets is data-dependent).
  Annotated at use sites.
- Upper triangle extraction uses fancy indexing → shapeless fallback, annotated.
- QR (Quotient-Remainder) embedding trick: compresses large vocabulary embedding
  tables by decomposing index i into quotient (i // q_factor) and remainder
  (i % q_factor), looking up two smaller tables, and combining.
- PR (Pruned-Row) embedding trick: uses a hashing function to map original
  indices to a smaller table, trading accuracy for memory.
- Multi-device parallelism: scatter embedding tables across devices, gather results.
  Uses unrefined Tensor since device placement is runtime-only.
- Quantization: quantized embedding lookup for inference. Uses torch quantization
  APIs which return unrefined Tensor.

Key patterns exercised:
- EmbeddingBag for sparse feature lookup
- Feature interaction via BMM (batch matrix multiply)
- Shapeless fallback for triangle extraction + annotation
- Multiple MLPs with different roles (bottom/top)
- Concrete MLP layers (no dynamic construction)
- Embedding compression tricks (QR, PR)
- Multi-device scatter/gather for distributed inference
- Quantized embeddings for memory-efficient inference
"""

from typing import Any, assert_type, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.quantization

if TYPE_CHECKING:
    from torch import Tensor
    from torch_shapes import Dim


# ============================================================================
# Bottom MLP
# ============================================================================


class BottomMLP[DenseDim, D](nn.Module):
    """Bottom MLP: maps dense features to embedding space.

    Architecture: DenseDim → 512 → 256 → D

    (B, DenseDim) → (B, D)
    """

    def __init__(self, dense_dim: Dim[DenseDim], embed_dim: Dim[D]) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dense_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, embed_dim)

    def forward[B](self, x: Tensor[B, DenseDim]) -> Tensor[B, D]:
        h = nn.functional.relu(self.fc1(x))
        assert_type(h, Tensor[B, 512])
        h = nn.functional.relu(self.fc2(h))
        assert_type(h, Tensor[B, 256])
        return self.fc3(h)


# ============================================================================
# Top MLP
# ============================================================================


class TopMLP[TopIn](nn.Module):
    """Top MLP: maps interaction features to prediction.

    Architecture: TopIn → 512 → 256 → 1

    (B, TopIn) → (B, 1)
    """

    def __init__(self, top_in: Dim[TopIn]) -> None:
        super().__init__()
        self.fc1 = nn.Linear(top_in, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward[B](self, x: Tensor[B, TopIn]) -> Tensor[B, 1]:
        h = nn.functional.relu(self.fc1(x))
        assert_type(h, Tensor[B, 512])
        h = nn.functional.relu(self.fc2(h))
        assert_type(h, Tensor[B, 256])
        return torch.sigmoid(self.fc3(h))


# ============================================================================
# DLRM (concrete config: 3 embedding tables, D=64, DenseDim=13)
# ============================================================================


class DLRM[DenseDim, D](nn.Module):
    """DLRM recommendation model.

    Concrete configuration for shape tracking:
    - 3 embedding tables (different vocab sizes, same embedding dim D)
    - Bottom MLP: DenseDim → 512 → 256 → D
    - Feature interaction: 4 vectors (1 dense + 3 sparse) of dim D
      → BMM → upper triangle: (4*3)//2 = 6 interactions
    - Top MLP: (D + 6) → 512 → 256 → 1

    (B, DenseDim), list of sparse inputs → (B, 1)
    """

    def __init__(
        self,
        dense_dim: Dim[DenseDim],
        embed_dim: Dim[D],
        vocab1: int,
        vocab2: int,
        vocab3: int,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        # Bottom MLP
        self.bot_mlp = BottomMLP(dense_dim, embed_dim)
        # Embedding tables (3 tables, each producing (B, D))
        self.emb1 = nn.EmbeddingBag(vocab1, embed_dim, mode="sum")
        self.emb2 = nn.EmbeddingBag(vocab2, embed_dim, mode="sum")
        self.emb3 = nn.EmbeddingBag(vocab3, embed_dim, mode="sum")
        # Top MLP: D + 6 interaction features
        self.top_mlp = TopMLP(embed_dim + 6)

    def interact_features[B](
        self,
        dense: Tensor[B, D],
        sparse1: Tensor[B, D],
        sparse2: Tensor[B, D],
        sparse3: Tensor[B, D],
    ) -> Tensor[B, D + 6]:
        """Feature interaction via dot product.

        Stack 4 vectors → (B, 4, D) → BMM with transpose → (B, 4, 4)
        → extract upper triangle (6 elements) → concat with dense → (B, D+6)
        """
        b = dense.size(0)
        assert_type(b, Dim[B])
        # Stack: (B, 4, D)
        T = torch.stack((dense, sparse1, sparse2, sparse3), dim=1)
        assert_type(T, Tensor[B, 4, D])
        # Pairwise dot products: (B, 4, D) @ (B, D, 4) → (B, 4, 4)
        Z = torch.bmm(T, T.transpose(1, 2))
        assert_type(Z, Tensor[B, 4, 4])
        # Extract upper triangle (without diagonal): 6 elements
        # Annotation fallback: torch.tensor() returns bare Tensor
        li: Tensor[6] = torch.tensor([0, 0, 0, 1, 1, 2])
        lj: Tensor[6] = torch.tensor([1, 2, 3, 2, 3, 3])
        interactions = Z[:, li, lj]
        assert_type(interactions, Tensor[B, 6])
        # Concat dense features with interactions
        result = torch.cat((dense, interactions), dim=1)
        assert_type(result, Tensor[B, D + 6])
        return result

    def forward[B](
        self,
        dense_x: Tensor[B, DenseDim],
        idx1: Tensor,
        off1: Tensor,
        idx2: Tensor,
        off2: Tensor,
        idx3: Tensor,
        off3: Tensor,
    ) -> Tensor[B, 1]:
        # Bottom MLP on dense features
        x = self.bot_mlp(dense_x)
        assert_type(x, Tensor[B, D])
        # Embedding lookups (EmbeddingBag returns unrefined Tensor)
        b = dense_x.size(0)
        assert_type(b, Dim[B])
        e1: Tensor[B, D] = self.emb1(idx1, off1)
        e2: Tensor[B, D] = self.emb2(idx2, off2)
        e3: Tensor[B, D] = self.emb3(idx3, off3)
        # Feature interaction
        z = self.interact_features(x, e1, e2, e3)
        assert_type(z, Tensor[B, D + 6])
        # Top MLP
        return self.top_mlp(z)


# ============================================================================
# QR Embedding (Quotient-Remainder Trick)
# ============================================================================


class QREmbeddingBag[D](nn.Module):
    """Quotient-Remainder embedding compression.

    Original: dlrm_s_pytorch.py QREmbeddingBag.

    Compresses a large embedding table (num_categories × D) into two smaller
    tables: quotient table (num_categories // q_factor × D) and remainder
    table (q_factor × D). The embedding for index i is:
        emb_q[i // q_factor] + emb_r[i % q_factor]

    This reduces memory from O(V × D) to O((V/q + q) × D).

    Forward takes indices and offsets (same interface as nn.EmbeddingBag).
    Returns unrefined Tensor since EmbeddingBag output batch dim is
    data-dependent (determined by offsets).
    """

    def __init__(
        self,
        num_categories: int,
        embedding_dim: Dim[D],
        q_factor: int,
    ) -> None:
        super().__init__()
        self.q_factor = q_factor
        q_num = (num_categories + q_factor - 1) // q_factor
        self.emb_q = nn.EmbeddingBag(q_num, embedding_dim, mode="sum")
        self.emb_r = nn.EmbeddingBag(q_factor, embedding_dim, mode="sum")

    def forward(self, indices: Tensor, offsets: Tensor) -> Tensor:
        """Look up compressed embeddings.

        indices: (N,) — categorical indices
        offsets: (B+1,) or (B,) — bag boundaries
        Returns: (B, D) — unrefined due to EmbeddingBag
        """
        q_indices: Tensor = torch.div(indices, self.q_factor, rounding_mode="floor")
        r_indices: Tensor = torch.remainder(indices, self.q_factor)
        emb_q: Tensor = self.emb_q(q_indices, offsets)
        emb_r: Tensor = self.emb_r(r_indices, offsets)
        return emb_q + emb_r


# ============================================================================
# PR Embedding (Pruned-Row Trick)
# ============================================================================


class PREmbeddingBag[D](nn.Module):
    """Pruned-Row embedding compression via hashing.

    Original: dlrm_s_pytorch.py PREmbeddingBag.

    Reduces vocabulary size by hashing original indices into a smaller table.
    Uses a simple modulo hash: hash(i) = i % num_rows.

    Memory savings: O(num_rows × D) instead of O(V × D) where num_rows << V.
    Trades accuracy for memory — hash collisions cause different categories
    to share embeddings.

    Forward takes indices and offsets (same interface as nn.EmbeddingBag).
    """

    def __init__(
        self,
        num_rows: int,
        embedding_dim: Dim[D],
    ) -> None:
        super().__init__()
        self.num_rows = num_rows
        self.emb = nn.EmbeddingBag(num_rows, embedding_dim, mode="sum")

    def forward(self, indices: Tensor, offsets: Tensor) -> Tensor:
        """Look up hashed embeddings.

        indices: (N,) — categorical indices (hashed to [0, num_rows))
        offsets: (B+1,) or (B,) — bag boundaries
        Returns: (B, D) — unrefined due to EmbeddingBag
        """
        hashed_indices: Tensor = torch.remainder(indices, self.num_rows)
        return self.emb(hashed_indices, offsets)


# ============================================================================
# Multi-Device Parallelism
# ============================================================================


def scatter_emb_tables(
    emb_tables: list[nn.EmbeddingBag],
    devices: list[torch.device],
) -> list[nn.EmbeddingBag]:
    """Scatter embedding tables across multiple devices.

    Original: dlrm_s_pytorch.py — embedding table parallelism logic.

    In the original DLRM, embedding tables are distributed across GPUs
    to handle large vocabulary sizes that don't fit in single-GPU memory.
    Each device gets a subset of tables; gather_emb_results recombines.

    Args:
        emb_tables: list of EmbeddingBag modules
        devices: target devices, one per table (or round-robin)

    Returns:
        list of EmbeddingBag modules moved to their assigned devices.
        Types are preserved — device placement is runtime-only.
    """
    distributed: list[nn.EmbeddingBag] = []
    for i, table in enumerate(emb_tables):
        device = devices[i % len(devices)]
        distributed.append(table.to(device))
    return distributed


def gather_emb_results(
    emb_outputs: list[Tensor],
    target_device: torch.device,
) -> list[Tensor]:
    """Gather embedding outputs from multiple devices to a single device.

    Original: dlrm_s_pytorch.py — gather logic after parallel embedding lookup.

    Args:
        emb_outputs: list of (B, D) tensors on various devices
        target_device: device to gather results to

    Returns:
        list of (B, D) tensors all on target_device. Unrefined.
    """
    return [e.to(target_device) for e in emb_outputs]


# ============================================================================
# Quantized Embedding
# ============================================================================


class QuantizedEmbeddingBag[D](nn.Module):
    """Quantized embedding lookup for inference.

    Original: dlrm_s_pytorch.py — quantization support.

    Wraps nn.EmbeddingBag with int8 quantization for reduced memory footprint
    during inference. The embedding weights are quantized; lookups dequantize
    on the fly.

    Uses torch.quantization APIs which return unrefined Tensor.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: Dim[D],
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        # Create a standard EmbeddingBag and prepare for quantization
        self.emb = nn.EmbeddingBag(num_embeddings, embedding_dim, mode="sum")

    def quantize(self) -> None:
        """Quantize the embedding table to int8.

        Uses torch.quantization.quantize_dynamic, same as original.
        Shape-wise this is a no-op — the output dimensions are unchanged.
        quantize_dynamic returns Module (not EmbeddingBag), so we need
        a type: ignore for the assignment.
        """
        self.emb = torch.quantization.quantize_dynamic(  # type: ignore[bad-assignment]
            self.emb, {nn.EmbeddingBag}, dtype=torch.qint8
        )

    def forward(self, indices: Tensor, offsets: Tensor) -> Tensor:
        """Look up embeddings (quantized or fp32).

        indices: (N,) — category indices
        offsets: (B+1,) or (B,) — bag boundaries
        Returns: (B, D) — unrefined due to EmbeddingBag
        """
        return self.emb(indices, offsets)


# ============================================================================
# Smoke tests
# ============================================================================


def test_bottom_mlp():
    """Test bottom MLP: (B, 13) → (B, 64)."""
    mlp = BottomMLP(13, 64)
    x: Tensor[4, 13] = torch.randn(4, 13)
    out = mlp(x)
    assert_type(out, Tensor[4, 64])


def test_top_mlp():
    """Test top MLP: (B, 70) → (B, 1)."""
    mlp = TopMLP(70)
    x: Tensor[4, 70] = torch.randn(4, 70)
    out = mlp(x)
    assert_type(out, Tensor[4, 1])


def test_interact_features():
    """Test feature interaction: 4 vectors → (B, D+6)."""
    model = DLRM(13, 64, 100, 200, 300)
    dense: Tensor[4, 64] = torch.randn(4, 64)
    s1: Tensor[4, 64] = torch.randn(4, 64)
    s2: Tensor[4, 64] = torch.randn(4, 64)
    s3: Tensor[4, 64] = torch.randn(4, 64)
    out = model.interact_features(dense, s1, s2, s3)
    assert_type(out, Tensor[4, 70])


def test_dlrm():
    """Test full DLRM: dense + sparse → prediction."""
    model = DLRM(13, 64, 100, 200, 300)
    dense: Tensor[32, 13] = torch.randn(32, 13)
    # Sparse inputs: indices and offsets for each table
    idx1: Tensor = torch.randint(0, 100, (64,))
    off1: Tensor = torch.arange(0, 65, 2)
    idx2: Tensor = torch.randint(0, 200, (96,))
    off2: Tensor = torch.arange(0, 97, 3)
    idx3: Tensor = torch.randint(0, 300, (64,))
    off3: Tensor = torch.arange(0, 65, 2)
    out = model(dense, idx1, off1, idx2, off2, idx3, off3)
    assert_type(out, Tensor[32, 1])


def test_dlrm_different_dims():
    """Test DLRM with different dense/embed dimensions."""
    model = DLRM(26, 32, 500, 1000, 2000)
    dense: Tensor[8, 26] = torch.randn(8, 26)
    idx1: Tensor = torch.randint(0, 500, (16,))
    off1: Tensor = torch.arange(0, 17, 2)
    idx2: Tensor = torch.randint(0, 1000, (24,))
    off2: Tensor = torch.arange(0, 25, 3)
    idx3: Tensor = torch.randint(0, 2000, (16,))
    off3: Tensor = torch.arange(0, 17, 2)
    out = model(dense, idx1, off1, idx2, off2, idx3, off3)
    assert_type(out, Tensor[8, 1])


def test_qr_embedding_bag():
    """Test QR embedding: compressed lookup via quotient-remainder."""
    qr_emb = QREmbeddingBag(10000, 64, q_factor=50)
    indices: Tensor = torch.randint(0, 10000, (32,))
    offsets: Tensor = torch.arange(0, 33, 4)
    _out = qr_emb(indices, offsets)
    # _out: (B, 64) — unrefined due to EmbeddingBag


def test_pr_embedding_bag():
    """Test PR embedding: compressed lookup via hashing."""
    pr_emb = PREmbeddingBag(1000, 64)
    indices: Tensor = torch.randint(0, 50000, (32,))
    offsets: Tensor = torch.arange(0, 33, 4)
    _out = pr_emb(indices, offsets)
    # _out: (B, 64) — unrefined due to EmbeddingBag


def test_quantized_embedding_bag():
    """Test quantized embedding: int8 quantized lookup."""
    q_emb = QuantizedEmbeddingBag(10000, 64)
    q_emb.quantize()
    indices: Tensor = torch.randint(0, 10000, (32,))
    offsets: Tensor = torch.arange(0, 33, 4)
    _out = q_emb(indices, offsets)
    # _out: (B, 64) — unrefined due to EmbeddingBag
