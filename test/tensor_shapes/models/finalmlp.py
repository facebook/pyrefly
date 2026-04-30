# Portions (c) Meta Platforms, Inc. and affiliates.
#
# This source code is adapted from pytorch/benchmark (TorchBenchmark),
# which is licensed under the BSD 3-Clause License:
# https://github.com/pytorch/benchmark/blob/main/LICENSE
#
# Original model: reczoo/FuxiCTR (Apache 2.0 License)
# Reference: K. Mao et al., "FinalMLP: An Enhanced Two-Stream MLP Model
# for CTR Prediction," AAAI 2023.
#
# This adaptation adds tensor shape type annotations for pyrefly.

from dataclasses import dataclass, field
# ## Inventory
# - [x] FinalMLPConfig — dataclass; Dims: mlp1_output_dim, mlp2_output_dim, num_heads, num_output_features; int: num_layers
# - [x] MLP.__init__ — Dims: input_dim, output_dim; int: hidden_units (list)
# - [x] MLP.forward
# - [x] InteractionAggregation.__init__ — Dims: x_dim, y_dim, output_dim, num_heads
# - [x] InteractionAggregation.forward
# - [x] FinalMLPLayer.__init__ — Dims: num_features, emb_dim, num_output_features, output_emb_dim; config provides MLP dims
# - [x] FinalMLPLayer.forward
# - [x] FinalMLPBackbone.__init__ — Dims: num_features, emb_dim; config provides rest
# - [x] FinalMLPBackbone.output_dim — property
# - [x] FinalMLPBackbone.forward

from typing import Any, assert_type, TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from torch import Tensor
    from torch_shapes import Dim


@dataclass
class FinalMLPConfig[M1Out = 256, M2Out = 256, NH = 1, K = 16]:
    mlp1_hidden_units: list[int] = field(default_factory=lambda: [512, 256])
    mlp1_output_dim: Dim[M1Out] = 256  # type: ignore[bad-assignment]
    mlp1_activation: str = "ReLU"
    mlp1_dropout: float = 0.0
    mlp1_batch_norm: bool = False
    mlp2_hidden_units: list[int] = field(default_factory=lambda: [512, 256])
    mlp2_output_dim: Dim[M2Out] = 256  # type: ignore[bad-assignment]
    mlp2_activation: str = "ReLU"
    mlp2_dropout: float = 0.0
    mlp2_batch_norm: bool = False
    num_heads: Dim[NH] = 1  # type: ignore[bad-assignment]
    num_layers: int = 1
    num_output_features: Dim[K] = 16  # type: ignore[bad-assignment]


class MLP[InD, OutD](nn.Module):
    """Multi-layer perceptron with shape-preserving layers built from Sequential(*list).

    Internal shapes are bare because Sequential(*list_var) erases module types.
    Bridge dims InD and OutD provide typed input/output interface.
    """

    output_dim: Dim[OutD]

    def __init__(
        self,
        input_dim: Dim[InD],
        hidden_units: list[int],
        output_dim: Dim[OutD],
        activation: str = "ReLU",
        dropout: float = 0.0,
        batch_norm: bool = False,
    ) -> None:
        super().__init__()
        layer_blocks: list[nn.Module] = []
        in_dim = input_dim
        for out_dim in hidden_units:
            block: list[nn.Module] = []
            block.append(nn.Linear(in_dim, out_dim))
            if batch_norm:
                block.append(nn.BatchNorm1d(out_dim))
            block.append(getattr(nn, activation)())
            if dropout > 0:
                block.append(nn.Dropout(dropout))
            layer_blocks.append(nn.Sequential(*block))
        self.layers = nn.ModuleList(layer_blocks)
        self.output_dim = output_dim

    def forward[B](self, x: Tensor[B, InD]) -> Tensor[B, OutD]:
        for layer in self.layers:
            x = layer(x)
        # typed interface: Sequential(*list) + ModuleList[nn.Module] loop erases shapes
        result: Tensor[B, OutD] = x  # type: ignore[bad-assignment]
        assert_type(result, Tensor[B, OutD])
        return result


class InteractionAggregation[XD, YD, OutD, NH](nn.Module):
    """Bilinear interaction aggregation for fusing two stream outputs.

    Computes: w_x(x) + w_y(y) + sum_h(head_x_h @ W_xy_h @ head_y_h)
    """

    def __init__(
        self,
        x_dim: Dim[XD],
        y_dim: Dim[YD],
        output_dim: Dim[OutD],
        num_heads: Dim[NH],
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.output_dim = output_dim

        self.w_x = nn.Linear(x_dim, output_dim)
        self.w_y = nn.Linear(y_dim, output_dim)

        head_x_dim = x_dim // num_heads
        head_y_dim = y_dim // num_heads
        self.bilinear_W = nn.ParameterList(
            [
                nn.Parameter(torch.randn(head_x_dim, head_y_dim) * 0.01)
                for _ in range(num_heads)
            ]
        )
        self.bilinear_out = nn.Linear(num_heads, output_dim)

    def forward[B](self, x: Tensor[B, XD], y: Tensor[B, YD]) -> Tensor[B, OutD]:
        out = self.w_x(x) + self.w_y(y)
        assert_type(out, Tensor[B, OutD])

        head_x_dim = x.size(1) // self.num_heads
        head_y_dim = y.size(1) // self.num_heads
        bilinear_terms: list[Tensor[B]] = []
        for i in range(self.num_heads):
            x_h = x[:, i * head_x_dim : (i + 1) * head_x_dim]
            assert_type(x_h, Tensor[B, Any])  # slicing with int indices loses head dim
            y_h = y[:, i * head_y_dim : (i + 1) * head_y_dim]
            assert_type(y_h, Tensor[B, Any])  # slicing with int indices loses head dim
            interaction = (x_h @ self.bilinear_W[i] * y_h).sum(dim=-1)
            assert_type(interaction, Tensor[B])
            bilinear_terms.append(interaction)

        # annotation fallback: stack from dynamic loop can't infer collection size
        bilinear_out: Tensor[B, NH] = torch.stack(bilinear_terms, dim=-1)
        assert_type(bilinear_out, Tensor[B, NH])
        projected = self.bilinear_out(bilinear_out)
        assert_type(projected, Tensor[B, OutD])
        out = out + projected
        assert_type(out, Tensor[B, OutD])

        return out


class FinalMLPLayer[F, D, K, M1Out, M2Out, NH](nn.Module):
    """Single FinalMLP interaction layer: [B, F, D] -> [B, K, D].

    Runs two parallel MLPs on the flattened input, fuses via bilinear
    interaction aggregation, and projects the output to (batch, K, D)
    via a lazy linear layer.
    """

    def __init__(
        self,
        num_features: Dim[F],
        emb_dim: Dim[D],
        num_output_features: Dim[K],
        output_emb_dim: Dim[D],
        config: FinalMLPConfig[M1Out, M2Out, NH, K],
    ) -> None:
        super().__init__()
        self.num_output_features = num_output_features
        self.output_emb_dim = output_emb_dim

        input_dim = num_features * emb_dim

        self.mlp1 = MLP(
            input_dim=input_dim,
            hidden_units=config.mlp1_hidden_units,
            output_dim=config.mlp1_output_dim,
            activation=config.mlp1_activation,
            dropout=config.mlp1_dropout,
            batch_norm=config.mlp1_batch_norm,
        )
        self.mlp2 = MLP(
            input_dim=input_dim,
            hidden_units=config.mlp2_hidden_units,
            output_dim=config.mlp2_output_dim,
            activation=config.mlp2_activation,
            dropout=config.mlp2_dropout,
            batch_norm=config.mlp2_batch_norm,
        )

        self.fusion = InteractionAggregation(
            x_dim=config.mlp1_output_dim,
            y_dim=config.mlp2_output_dim,
            output_dim=config.mlp1_output_dim,
            num_heads=config.num_heads,
        )

        self.projector = nn.LazyLinear(num_output_features * output_emb_dim)
        self.layer_norm = nn.LayerNorm(output_emb_dim)

    def forward[B](self, input_embs: Tensor[B, F, D]) -> Tensor[B, K, D]:
        flat = input_embs.flatten(start_dim=1)
        assert_type(flat, Tensor[B, D * F])
        mlp1_out = self.mlp1(flat)
        assert_type(mlp1_out, Tensor[B, M1Out])
        mlp2_out = self.mlp2(flat)
        assert_type(mlp2_out, Tensor[B, M2Out])
        fused = self.fusion(mlp1_out, mlp2_out)
        assert_type(fused, Tensor[B, M1Out])
        projected = self.projector(fused)
        assert_type(projected, Tensor[B, D * K])
        out = projected.view(-1, self.num_output_features, self.output_emb_dim)
        assert_type(out, Tensor[B, K, D])
        result = self.layer_norm(out)
        assert_type(result, Tensor[B, K, D])
        return result


class FinalMLPBackbone[F, D, M1Out, M2Out, NH, K](nn.Module):
    """Multi-layer FinalMLP backbone: stacks FinalMLPLayers for iterative interaction.

    Each layer runs dual-stream MLPs with bilinear fusion and projects output to
    [B, K, D], which serves as input features for the next layer.  The final
    layer's 3D output is flattened to 2D.

    First layer is separated from rest layers because the first transforms
    [B, F, D] -> [B, K, D] while subsequent layers preserve [B, K, D].
    """

    def __init__(
        self,
        num_features: Dim[F],
        emb_dim: Dim[D],
        config: FinalMLPConfig[M1Out, M2Out, NH, K] | None = None,
    ) -> None:
        super().__init__()
        if config is None:
            config = FinalMLPConfig()

        K_dim = config.num_output_features

        self.first_layer = FinalMLPLayer(num_features, emb_dim, K_dim, emb_dim, config)
        rest: list[FinalMLPLayer[K, D, K, M1Out, M2Out, NH]] = []
        for _ in range(config.num_layers - 1):
            rest.append(FinalMLPLayer(K_dim, emb_dim, K_dim, emb_dim, config))
        self.rest_layers = nn.ModuleList(rest)

        self._output_dim = K_dim * emb_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward[B](self, input_embs: Tensor[B, F, D]) -> Tensor[B, K * D]:
        x = self.first_layer(input_embs)
        assert_type(x, Tensor[B, K, D])
        for layer in self.rest_layers:
            x = layer(x)
        assert_type(x, Tensor[B, K, D])
        result = x.flatten(1)
        assert_type(result, Tensor[B, D * K])
        return result


def test_interaction_aggregation():
    """Test bilinear interaction aggregation: (x_dim=128, y_dim=64) -> output_dim=128."""
    module = InteractionAggregation(128, 64, 128, 1)
    x: Tensor[4, 128] = torch.randn(4, 128)
    y: Tensor[4, 64] = torch.randn(4, 64)
    out = module(x, y)
    assert_type(out, Tensor[4, 128])


def test_finalmlp_layer():
    """Test single FinalMLP layer: [B, 32, 8] -> [B, 16, 8]."""
    config = FinalMLPConfig(
        mlp1_hidden_units=[64, 32],
        mlp1_output_dim=32,
        mlp2_hidden_units=[64, 32],
        mlp2_output_dim=32,
        num_heads=1,
        num_output_features=16,
    )
    layer = FinalMLPLayer(32, 8, 16, 8, config)
    x: Tensor[4, 32, 8] = torch.randn(4, 32, 8)
    out = layer(x)
    assert_type(out, Tensor[4, 16, 8])


def test_finalmlp_backbone():
    """Test end-to-end FinalMLP backbone: [B, 32, 8] -> [B, 128]."""
    config = FinalMLPConfig(
        mlp1_hidden_units=[64, 32],
        mlp1_output_dim=32,
        mlp2_hidden_units=[64, 32],
        mlp2_output_dim=32,
        num_heads=1,
        num_layers=2,
        num_output_features=16,
    )
    backbone = FinalMLPBackbone(32, 8, config)
    x: Tensor[2, 32, 8] = torch.randn(2, 32, 8)
    out = backbone(x)
    assert_type(out, Tensor[2, 128])
