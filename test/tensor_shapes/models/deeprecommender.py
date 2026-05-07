# Portions (c) Meta Platforms, Inc. and affiliates.
#
# This source code is adapted from pytorch/benchmark (TorchBenchmark),
# which is licensed under the BSD 3-Clause License:
# https://github.com/pytorch/benchmark/blob/main/LICENSE
#
# This adaptation adds tensor shape type annotations for pyrefly.

"""
NVIDIA Deep Recommender autoencoder from TorchBenchmark with shape annotations.

Original: pytorch/benchmark/torchbenchmark/models/nvidia_deeprecommender/
    reco_encoder/model/model.py

Port notes:
- Symmetric autoencoder for collaborative filtering. The original uses
    nn.ParameterList with F.linear for manual weight management. This port
    uses nn.Linear modules for better shape tracking.
- The original supports dynamic layer_sizes. This port uses a 3-layer
    architecture parameterized as AutoEncoder[InDim, H1, H2] where:
    Encoder: InDim → H1 → H2 (bottleneck)
    Decoder: H2 → H1 → InDim
- Activation function parameterized as ShapePreservingActivation (SELU default).
- Dropout applied only on the bottleneck (code) layer.
- The constrained (weight-tied) decoder path is dead code in the original:
    line 140 reads `if False:  # self.is_constrained:`. The else branch always
    executes (unconstrained decoder). Our port faithfully omits it.
- MSEloss: masked MSE loss — only computes error on non-zero entries (since
    ratings matrices are sparse in collaborative filtering).

Key patterns exercised:
- Symmetric autoencoder: encoder and decoder are mirror images
- Linear pipeline with activation between each layer
- Dropout on bottleneck only
- Masked loss computation on sparse data
"""

from typing import assert_type, TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from torch import Tensor
    from torch_shapes import Dim

# Activation function type — all are shape-preserving: Tensor[*S] -> Tensor[*S]
ShapePreservingActivation = (
    type[nn.ReLU]
    | type[nn.SELU]
    | type[nn.ELU]
    | type[nn.LeakyReLU]
    | type[nn.Sigmoid]
    | type[nn.Tanh]
)


# ============================================================================
# AutoEncoder
# ============================================================================


class AutoEncoder[InDim, H1, H2](nn.Module):
    """Symmetric autoencoder for collaborative filtering.

    Architecture (3-layer):
        Encoder: InDim → H1 → H2 (bottleneck, dropout applied here)
        Decoder: H2 → H1 → InDim

    Forward: (B, InDim) → (B, InDim)
    """

    def __init__(
        self,
        in_dim: Dim[InDim],
        h1: Dim[H1],
        h2: Dim[H2],
        act_fn: ShapePreservingActivation = nn.SELU,
        dp_drop_prob: float = 0.0,
    ) -> None:
        super().__init__()
        self.act = act_fn()
        self.dp_drop_prob = dp_drop_prob
        if dp_drop_prob > 0:
            self.drop = nn.Dropout(dp_drop_prob)

        # Encoder layers
        self.enc1 = nn.Linear(in_dim, h1)
        self.enc2 = nn.Linear(h1, h2)

        # Decoder layers (mirror of encoder)
        self.dec1 = nn.Linear(h2, h1)
        self.dec2 = nn.Linear(h1, in_dim)

    def encode[B](self, x: Tensor[B, InDim]) -> Tensor[B, H2]:
        h = self.act(self.enc1(x))
        assert_type(h, Tensor[B, H1])
        z = self.act(self.enc2(h))
        assert_type(z, Tensor[B, H2])
        if self.dp_drop_prob > 0:
            z = self.drop(z)
        return z

    def decode[B](self, z: Tensor[B, H2]) -> Tensor[B, InDim]:
        h = self.act(self.dec1(z))
        assert_type(h, Tensor[B, H1])
        out = self.act(self.dec2(h))
        assert_type(out, Tensor[B, InDim])
        return out

    def forward[B](self, x: Tensor[B, InDim]) -> Tensor[B, InDim]:
        return self.decode(self.encode(x))


# ============================================================================
# Loss
# ============================================================================


def MSEloss[B, InDim](inputs: Tensor[B, InDim], targets: Tensor[B, InDim]) -> Tensor:
    """Masked MSE loss — only computes error on non-zero target entries.

    In collaborative filtering, the ratings matrix is sparse: most entries are
    zero (unobserved). This loss masks out the zeros so the model is only
    penalized on observed ratings. Returns scalar loss.

    All intermediate shapes tracked: mask (B, InDim), mask.float() (B, InDim),
    products (B, InDim). MSELoss(reduction="sum") returns scalar.

    Original: reco_encoder/model/model.py MSEloss function.
    """
    mask = targets != 0
    num_ratings = torch.sum(mask.float())
    criterion = nn.MSELoss(reduction="sum")
    mse = criterion(inputs * mask.float(), targets * mask.float())
    return mse / (num_ratings + 1e-8)


# ============================================================================
# Smoke tests
# ============================================================================


def test_autoencoder_encode():
    """Test encoder: (B, InDim) → (B, H2)."""
    ae = AutoEncoder(10000, 1024, 512)
    x: Tensor[32, 10000] = torch.randn(32, 10000)
    z = ae.encode(x)
    assert_type(z, Tensor[32, 512])


def test_autoencoder_decode():
    """Test decoder: (B, H2) → (B, InDim)."""
    ae = AutoEncoder(10000, 1024, 512)
    z: Tensor[32, 512] = torch.randn(32, 512)
    out = ae.decode(z)
    assert_type(out, Tensor[32, 10000])


def test_autoencoder_roundtrip():
    """Test full autoencoder: (B, InDim) → (B, InDim)."""
    ae = AutoEncoder(10000, 1024, 512)
    x: Tensor[16, 10000] = torch.randn(16, 10000)
    out = ae(x)
    assert_type(out, Tensor[16, 10000])


def test_autoencoder_relu():
    """Test with ReLU activation instead of default SELU."""
    ae = AutoEncoder(5000, 512, 256, act_fn=nn.ReLU)
    x: Tensor[8, 5000] = torch.randn(8, 5000)
    out = ae(x)
    assert_type(out, Tensor[8, 5000])


def test_autoencoder_with_dropout():
    """Test with dropout on bottleneck."""
    ae = AutoEncoder(5000, 512, 256, dp_drop_prob=0.5)
    x: Tensor[8, 5000] = torch.randn(8, 5000)
    out = ae(x)
    assert_type(out, Tensor[8, 5000])


def test_mse_loss():
    """Test masked MSE loss on autoencoder output."""
    ae = AutoEncoder(10000, 1024, 512)
    x: Tensor[32, 10000] = torch.randn(32, 10000)
    out = ae(x)
    loss = MSEloss(out, x)
    assert_type(loss, Tensor)
