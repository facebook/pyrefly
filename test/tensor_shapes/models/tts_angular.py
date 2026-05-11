# Portions (c) Meta Platforms, Inc. and affiliates.
#
# This source code is adapted from pytorch/benchmark (TorchBenchmark),
# which is licensed under the BSD 3-Clause License:
# https://github.com/pytorch/benchmark/blob/main/LICENSE
#
# This adaptation adds tensor shape type annotations for pyrefly.

"""
TTS Angular speaker encoder from TorchBenchmark with shape annotations.

Original: pytorch/benchmark/torchbenchmark/models/tts_angular/model.py

Port notes:
- SpeakerEncoder with stacked LSTM layers, each with linear projection.
- LSTMWithProjection: LSTM(input, hidden) → Linear(hidden, proj). Each layer
    maps (B, T, input) → (B, T, proj), so stacking preserves the signature
    after the first layer: (B, T, proj) → (B, T, proj).
- SpeakerEncoder: N stacked LSTMWithProjection layers → take last timestep
    → L2-normalize → speaker embedding (B, ProjDim).
- Original builds all layers in one list and wraps with nn.Sequential.
  Port uses nn.ModuleList with list[LSTMWithProjection[Any, Any, Any]] and
  narrows the first element (InDim→ProjDim) at the call site. Remaining
  layers are shape-preserving (ProjDim→ProjDim).
- Loss modules (GE2ELoss, AngleProtoLoss): compute cosine similarity matrices
    for speaker verification training. Matrix dimensions are data-dependent
    (N speakers × M utterances) — annotated with unrefined Tensor where needed.
- compute_embedding / batch_compute_embedding: inference helpers for extracting
    speaker embeddings from variable-length utterances with sliding windows.
- LSTMWithoutProjection: uses multi-layer LSTM and extracts hidden[-1] (last
    layer hidden state). Our LSTM stub returns (output, h_n, c_n) where h_n is
    Tensor[NumLayers, B, Hidden]. Indexing h_n[-1] to get last layer is
    unrefined (negative indexing on first dim). The output is annotated as bare
    Tensor and narrowed to Tensor[B, ProjDim] at the call site.
- SpeakerEncoder supports both use_lstm_with_projection=True (stacked
    LSTMWithProjection layers) and False (single LSTMWithoutProjection).
- nn.LSTM uses num_directions=1 (unidirectional) with batch_first=True.
    Output shape: (B, T, hidden_size).

Key patterns exercised:
- list[Foo[Any]] + narrowing pattern for heterogeneous ModuleList
- Stacked recurrent layers (homogeneous after first)
- Multi-layer LSTM hidden state extraction: hidden[-1] (LSTMWithoutProjection)
- Last timestep extraction: output[:, -1] on sequence dimension
- L2 normalization: F.normalize(x, p=2, dim=1) — shape-preserving
- Cosine similarity loss (GE2E, AngleProto) for speaker verification
- Sliding window embedding extraction for inference
"""

from typing import Any, assert_type, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from torch import Tensor
    from torch_shapes import Dim


# ============================================================================
# LSTM with Projection
# ============================================================================


class LSTMWithProjection[InSize, Hidden, Proj](nn.Module):
    """LSTM followed by linear projection.

    (B, T, InSize) → LSTM → (B, T, Hidden) → Linear → (B, T, Proj)
    """

    def __init__(
        self,
        input_size: Dim[InSize],
        hidden_size: Dim[Hidden],
        proj_size: Dim[Proj],
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, proj_size, bias=False)

    def forward[B, T](self, x: Tensor[B, T, InSize]) -> Tensor[B, T, Proj]:
        self.lstm.flatten_parameters()
        o, _h_n, _c_n = self.lstm(x)
        assert_type(o, Tensor[B, T, Hidden])
        return self.linear(o)


# ============================================================================
# LSTM without Projection
# ============================================================================


class LSTMWithoutProjection[InSize, Hidden, Proj](nn.Module):
    """Multi-layer LSTM that extracts last layer's hidden state, then projects.

    Original: tts_angular/model.py LSTMWithoutProjection class.

    Uses multi-layer LSTM (num_lstm_layers stacked), takes hidden[-1]
    (last layer's hidden state), then Linear → ReLU.

    (B, T, InSize) → LSTM(num_layers) → hidden[-1] → (B, Hidden)
        → Linear → (B, Proj) → ReLU → (B, Proj)
    """

    def __init__(
        self,
        input_dim: Dim[InSize],
        lstm_dim: Dim[Hidden],
        proj_dim: Dim[Proj],
        num_lstm_layers: int,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, lstm_dim, num_layers=num_lstm_layers, batch_first=True
        )
        self.linear = nn.Linear(lstm_dim, proj_dim)
        self.relu = nn.ReLU()

    def forward[B, T](self, x: Tensor[B, T, InSize]) -> Tensor[B, Proj]:
        self.lstm.flatten_parameters()
        _output, h_n, _c_n = self.lstm(x)
        # h_n: (num_layers, B, Hidden) — h_n[-1] returns unrefined (negative
        # indexing on first dim not tracked), but nn.Linear still tracks output
        # dim from unrefined input, so shapes flow through
        last_hidden: Tensor = h_n[-1]
        return self.relu(self.linear(last_hidden))


# ============================================================================
# Speaker Encoder
# ============================================================================


class SpeakerEncoder[InDim, ProjDim](nn.Module):
    """Speaker verification encoder.

    Two modes (original: tts_angular/model.py SpeakerEncoder):

    use_lstm_with_projection=True (default):
        Stacks N LSTMWithProjection layers, takes last timestep, L2-normalizes.
        (B, T, InDim) → N × LSTMWithProjection → (B, T, ProjDim)
            → last timestep → (B, ProjDim) → L2 normalize → (B, ProjDim)

    use_lstm_with_projection=False:
        Single LSTMWithoutProjection (multi-layer LSTM + Linear + ReLU).
        (B, T, InDim) → LSTMWithoutProjection → (B, ProjDim)
            → L2 normalize → (B, ProjDim)
    """

    def __init__(
        self,
        input_dim: Dim[InDim],
        proj_dim: Dim[ProjDim],
        lstm_dim: int = 768,
        num_lstm_layers: int = 3,
        use_lstm_with_projection: bool = True,
    ) -> None:
        super().__init__()
        self.use_lstm_with_projection = use_lstm_with_projection

        if use_lstm_with_projection:
            # Stacked single-layer LSTMs with projection (original pattern)
            # First layer: InDim → ProjDim, rest: ProjDim → ProjDim
            layers: list[LSTMWithProjection[Any, Any, Any]] = [
                LSTMWithProjection(input_dim, lstm_dim, proj_dim),
            ]
            for _ in range(num_lstm_layers - 1):
                layers.append(LSTMWithProjection(proj_dim, lstm_dim, proj_dim))
            self.network = nn.ModuleList(layers)
        else:
            # Single multi-layer LSTM without projection
            self.lstm_no_proj = LSTMWithoutProjection(
                input_dim, lstm_dim, proj_dim, num_lstm_layers
            )

    def forward[B, T](self, x: Tensor[B, T, InDim]) -> Tensor[B, ProjDim]:
        if self.use_lstm_with_projection:
            # First layer: (B, T, InDim) → (B, T, ProjDim)
            first: LSTMWithProjection[InDim, int, ProjDim] = self.network[0]
            h = first(x)
            assert_type(h, Tensor[B, T, ProjDim])
            # Remaining layers: (B, T, ProjDim) → (B, T, ProjDim)
            for i in range(1, len(self.network)):
                layer: LSTMWithProjection[ProjDim, int, ProjDim] = self.network[i]
                h = layer(h)
            assert_type(h, Tensor[B, T, ProjDim])
            # Take last timestep and L2-normalize
            last = h[:, -1]
            assert_type(last, Tensor[B, ProjDim])
            return F.normalize(last, p=2, dim=1)
        else:
            # LSTMWithoutProjection already extracts last hidden state
            d = self.lstm_no_proj(x)
            return F.normalize(d, p=2, dim=1)


# ============================================================================
# Loss Modules
# ============================================================================


class GE2ELoss(nn.Module):
    """Generalized end-to-end loss for speaker verification.

    Original: tts_angular/model.py GE2ELoss class.

    Computes a cosine similarity matrix between speaker embeddings and
    per-speaker centroids. The matrix has shape (N*M, N) where N = number
    of speakers, M = number of utterances per speaker.
    Dimensions are data-dependent (N and M vary per batch).
    """

    def __init__(self, init_w: float = 10.0, init_b: float = -5.0) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.tensor(init_w))
        self.b = nn.Parameter(torch.tensor(init_b))
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, embeddings: Tensor, n_speakers: int, n_utterances: int) -> Tensor:
        """Compute GE2E loss.

        Args:
            embeddings: (N*M, D) speaker embeddings, reshaped to (N, M, D) internally
            n_speakers: N — number of speakers in the batch
            n_utterances: M — number of utterances per speaker
        """
        # Reshape to (N, M, D)
        dvecs: Tensor = embeddings.view(n_speakers, n_utterances, -1)
        # Compute centroids per speaker: (N, D)
        centroids: Tensor = dvecs.mean(dim=1)
        # Cosine similarity matrix: (N*M, N)
        cos_sim: Tensor = F.cosine_similarity(
            embeddings.unsqueeze(1), centroids.unsqueeze(0), dim=2
        )
        sim_matrix: Tensor = self.w * cos_sim + self.b
        # Target: each utterance should match its speaker
        targets: Tensor = torch.arange(n_speakers).repeat_interleave(n_utterances)
        return self.criterion(sim_matrix, targets)


class AngleProtoLoss(nn.Module):
    """Angular prototypical loss for speaker verification.

    Original: tts_angular/model.py AngleProtoLoss class.

    Similar to GE2ELoss but uses angular (cosine) distance with a learned
    scale and bias. The first utterance is used as the "prototype" and the
    rest are compared against it.
    """

    def __init__(self, init_w: float = 10.0, init_b: float = -5.0) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.tensor(init_w))
        self.b = nn.Parameter(torch.tensor(init_b))
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, embeddings: Tensor, n_speakers: int, n_utterances: int) -> Tensor:
        """Compute angular prototypical loss.

        Args:
            embeddings: (N*M, D) speaker embeddings
            n_speakers: N
            n_utterances: M
        """
        dvecs: Tensor = embeddings.view(n_speakers, n_utterances, -1)
        # Use first utterance as prototype, rest as queries
        prototypes: Tensor = dvecs[:, 0, :]  # (N, D)
        queries: Tensor = dvecs[:, 1:, :].reshape(-1, dvecs.size(2))  # (N*(M-1), D)
        # Cosine similarity: (N*(M-1), N)
        cos_sim: Tensor = F.cosine_similarity(
            queries.unsqueeze(1), prototypes.unsqueeze(0), dim=2
        )
        sim_matrix: Tensor = self.w * cos_sim + self.b
        targets: Tensor = torch.arange(n_speakers).repeat_interleave(n_utterances - 1)
        return self.criterion(sim_matrix, targets)


# ============================================================================
# Inference Helpers
# ============================================================================


def compute_embedding[ProjDim](
    encoder: SpeakerEncoder[int, ProjDim],
    utterance: Tensor,
    num_eval: int = 10,
    partial_n_frames: int = 160,
) -> Tensor:
    """Extract speaker embedding from a single utterance using sliding windows.

    Original: tts_angular/model.py SpeakerEncoder.compute_embedding (standalone version).

    Splits the utterance into overlapping windows of partial_n_frames,
    encodes each window, and averages the embeddings.

    Args:
        encoder: trained SpeakerEncoder
        utterance: (T, InDim) mel spectrogram for one utterance (no batch dim)
        num_eval: number of windows to extract
        partial_n_frames: frames per window

    Returns:
        (ProjDim,) averaged speaker embedding
    """
    max_len = utterance.size(0)
    if max_len < partial_n_frames:
        # Pad short utterances
        pad_size = partial_n_frames - max_len
        utterance = F.pad(utterance, (0, 0, 0, pad_size))
        max_len = partial_n_frames

    # Extract evenly-spaced windows
    offsets: Tensor = torch.linspace(
        0, max_len - partial_n_frames, steps=num_eval
    ).long()
    frames: list[Tensor] = []
    for offset in offsets:
        start: int = offset.item()  # type: ignore[assignment]
        window: Tensor = utterance[start : start + partial_n_frames]
        frames.append(window)
    windows: Tensor = torch.stack(frames)  # (num_eval, partial_n_frames, InDim)
    # Encode all windows at once
    with torch.no_grad():
        embeddings: Tensor = encoder(windows)  # (num_eval, ProjDim)
    # Average across windows
    return embeddings.mean(dim=0)  # (ProjDim,)


def batch_compute_embedding[ProjDim](
    encoder: SpeakerEncoder[int, ProjDim],
    utterances: list[Tensor],
    num_eval: int = 10,
    partial_n_frames: int = 160,
) -> Tensor:
    """Extract speaker embeddings for a batch of utterances.

    Original: tts_angular/model.py SpeakerEncoder.batch_compute_embedding.

    Args:
        encoder: trained SpeakerEncoder
        utterances: list of (T_i, InDim) mel spectrograms (variable length)
        num_eval: number of windows per utterance
        partial_n_frames: frames per window

    Returns:
        (len(utterances), ProjDim) stacked embeddings
    """
    embeddings: list[Tensor] = [
        compute_embedding(encoder, utt, num_eval, partial_n_frames)
        for utt in utterances
    ]
    return torch.stack(embeddings)  # (N, ProjDim)


# ============================================================================
# Smoke tests
# ============================================================================


def test_lstm_with_projection():
    """Test single LSTM + projection layer."""
    layer = LSTMWithProjection(40, 768, 256)
    x: Tensor[8, 100, 40] = torch.randn(8, 100, 40)
    out = layer(x)
    assert_type(out, Tensor[8, 100, 256])


def test_lstm_with_projection_identity_dims():
    """Test when input_size == proj_size (used for layers 2..N)."""
    layer = LSTMWithProjection(256, 768, 256)
    x: Tensor[4, 50, 256] = torch.randn(4, 50, 256)
    out = layer(x)
    assert_type(out, Tensor[4, 50, 256])


def test_speaker_encoder():
    """Test full speaker encoder: mel features → speaker embedding."""
    enc = SpeakerEncoder(40, 256, lstm_dim=768, num_lstm_layers=3)
    x: Tensor[8, 100, 40] = torch.randn(8, 100, 40)
    out = enc(x)
    assert_type(out, Tensor[8, 256])


def test_speaker_encoder_different_dims():
    """Test with different input/projection dimensions."""
    enc = SpeakerEncoder(80, 128, lstm_dim=512, num_lstm_layers=2)
    x: Tensor[4, 200, 80] = torch.randn(4, 200, 80)
    out = enc(x)
    assert_type(out, Tensor[4, 128])


def test_lstm_without_projection():
    """Test LSTMWithoutProjection: multi-layer LSTM → hidden[-1] → Linear → ReLU."""
    layer = LSTMWithoutProjection(40, 768, 256, num_lstm_layers=3)
    x: Tensor[8, 100, 40] = torch.randn(8, 100, 40)
    out = layer(x)
    assert_type(out, Tensor[8, 256])


def test_speaker_encoder_without_projection():
    """Test SpeakerEncoder with use_lstm_with_projection=False."""
    enc = SpeakerEncoder(
        40, 256, lstm_dim=768, num_lstm_layers=3, use_lstm_with_projection=False
    )
    x: Tensor[8, 100, 40] = torch.randn(8, 100, 40)
    out = enc(x)
    assert_type(out, Tensor[8, 256])
