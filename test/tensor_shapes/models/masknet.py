# Portions (c) Meta Platforms, Inc. and affiliates.
#
# This source code is adapted from pytorch/benchmark (TorchBenchmark),
# which is licensed under the BSD 3-Clause License:
# https://github.com/pytorch/benchmark/blob/main/LICENSE
#
# Original model: reczoo/FuxiCTR (Apache 2.0 License)
# Reference: Z. Wang et al., "MaskNet: Introducing Feature-Wise
# Multiplication to CTR Ranking Models by Instance-Guided Mask,"
# DLP-KDD 2021.
#
# This adaptation adds tensor shape type annotations for pyrefly.

# ## Inventory
# - [x] MaskNetConfig — dataclass, no tensors
# - [x] MaskBlock.__init__ — Dims: input_dim, hidden_dim, output_dim; float: reduction_ratio; float: dropout_rate; bool: layer_norm
# - [x] MaskBlock.forward
# - [x] SerialMaskNet.__init__ — Dims: input_dim, last_hidden; int: none (hidden_units extracted as Dims)
# - [x] SerialMaskNet.forward
# - [x] ParallelMaskNet.__init__ — Dims: input_dim, block_dim; int: num_blocks
# - [x] ParallelMaskNet.forward
# - [x] MaskNetBackbone.__init__ — Dims: num_features, emb_dim, output_dim; int: compression_num (from config)
# - [x] MaskNetBackbone.output_dim (property)
# - [x] MaskNetBackbone.forward

from __future__ import annotations

from dataclasses import dataclass, field
from typing import assert_type, TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from torch import Tensor
    from torch_shapes import Dim


@dataclass
class MaskNetConfig:
    model_type: str = "SerialMaskNet"
    hidden_units: list[int] = field(default_factory=lambda: [512, 256])
    hidden_activation: str = "ReLU"
    reduction_ratio: float = 1.0
    dropout: float = 0.0
    layer_norm: bool = True
    num_blocks: int = 1
    block_dim: int = 64
    compression_num: int = 0


class MaskBlock[InD, HidD, RedD, OutD](nn.Module):
    """Core building block of MaskNet.

    Generates an instance-guided mask from V_emb, applies it to V_hidden,
    then projects through a hidden layer.
    """

    def __init__(
        self,
        input_dim: Dim[InD],
        hidden_dim: Dim[HidD],
        reduced_dim: Dim[RedD],
        output_dim: Dim[OutD],
        hidden_activation: str = "ReLU",
        dropout_rate: float = 0.0,
        layer_norm: bool = True,
    ) -> None:
        super().__init__()
        self.mask_layer = nn.Sequential(
            nn.Linear(input_dim, reduced_dim),
            nn.ReLU(),
            nn.Linear(reduced_dim, hidden_dim),
        )
        self.hidden_linear = nn.Linear(hidden_dim, output_dim, bias=False)
        self.hidden_norm = nn.LayerNorm(output_dim) if layer_norm else None
        self.hidden_act: nn.Module = getattr(nn, hidden_activation)()
        self.hidden_dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None

    def forward[B](
        self, V_emb: Tensor[B, InD], V_hidden: Tensor[B, HidD]
    ) -> Tensor[B, OutD]:
        V_mask = self.mask_layer(V_emb)
        assert_type(V_mask, Tensor[B, HidD])
        masked = V_mask * V_hidden
        assert_type(masked, Tensor[B, HidD])
        v_out = self.hidden_linear(masked)
        assert_type(v_out, Tensor[B, OutD])
        if self.hidden_norm is not None:
            v_out = self.hidden_norm(v_out)
            assert_type(v_out, Tensor[B, OutD])
        # getattr(nn, activation)() returns nn.Module — forward returns Any
        v_out: Tensor[B, OutD] = self.hidden_act(v_out)
        assert_type(v_out, Tensor[B, OutD])
        if self.hidden_dropout is not None:
            v_out = self.hidden_dropout(v_out)
            assert_type(v_out, Tensor[B, OutD])
        return v_out


class SerialMaskNet[InD, OutD](nn.Module):
    """Serial MaskNet: chain of MaskBlocks where each block's output feeds the next."""

    def __init__(
        self,
        input_dim: Dim[InD],
        output_dim: Dim[OutD],
        hidden_units: list[int],
        hidden_activation: str = "ReLU",
        reduction_ratio: float = 1.0,
        dropout_rate: float = 0.0,
        layer_norm: bool = True,
    ) -> None:
        super().__init__()
        dims = [int(input_dim)] + hidden_units
        self.mask_blocks = nn.ModuleList()
        for idx in range(len(dims) - 1):
            self.mask_blocks.append(
                MaskBlock(
                    input_dim,
                    dims[idx],
                    int(dims[idx] * reduction_ratio),
                    dims[idx + 1],
                    hidden_activation,
                    dropout_rate,
                    layer_norm,
                )
            )
        self.output_dim = output_dim

    def forward[B](
        self, V_emb: Tensor[B, InD], V_hidden: Tensor[B, InD]
    ) -> Tensor[B, OutD]:
        v_out: Tensor = V_hidden
        assert_type(v_out, Tensor)
        for block in self.mask_blocks:
            # ModuleList iteration — blocks constructed with int from list, returns Unknown
            v_out = block(V_emb, v_out)  # type: ignore[assignment]
        # Annotation fallback: last block outputs Tensor[B, OutD]
        result: Tensor[B, OutD] = v_out  # type: ignore[assignment]
        assert_type(result, Tensor[B, OutD])
        return result


class ParallelMaskNet[InD, BlkD = 64, OutD = 64](nn.Module):
    """Parallel MaskNet: multiple independent MaskBlocks, outputs concatenated."""

    def __init__(
        self,
        input_dim: Dim[InD],
        output_dim: Dim[OutD],
        num_blocks: int = 1,
        block_dim: Dim[BlkD] = 64,
        dnn_hidden_units: list[int] | None = None,
        hidden_activation: str = "ReLU",
        reduction_ratio: float = 1.0,
        dropout_rate: float = 0.0,
        layer_norm: bool = True,
    ) -> None:
        super().__init__()
        self.num_blocks = num_blocks
        self.mask_blocks = nn.ModuleList(
            [
                MaskBlock(
                    input_dim,
                    input_dim,
                    int(input_dim * reduction_ratio),
                    block_dim,
                    hidden_activation,
                    dropout_rate,
                    layer_norm,
                )
                for _ in range(num_blocks)
            ]
        )

        # DNN after concatenation — Sequential(*list) erases types
        dnn_layers: list[nn.Module] = []
        in_dim = int(block_dim) * num_blocks
        if dnn_hidden_units:
            for out_dim in dnn_hidden_units:
                dnn_layers.append(nn.Linear(in_dim, out_dim))
                dnn_layers.append(getattr(nn, hidden_activation)())
                if dropout_rate > 0:
                    dnn_layers.append(nn.Dropout(dropout_rate))
                in_dim = out_dim
        self.dnn = nn.Sequential(*dnn_layers) if dnn_layers else nn.Identity()
        self.output_dim = output_dim

    def forward[B](
        self, V_emb: Tensor[B, InD], V_hidden: Tensor[B, InD]
    ) -> Tensor[B, OutD]:
        block_out = [
            self.mask_blocks[i](V_emb, V_hidden) for i in range(self.num_blocks)
        ]
        assert_type(block_out, list[Tensor[B, BlkD]])
        # torch.cat with list — DSL can't infer size from dynamic list
        concat_out: Tensor = torch.cat(block_out, dim=-1)
        assert_type(concat_out, Tensor)
        # Sequential(*list) | Identity — both erase types
        result: Tensor[B, OutD] = self.dnn(concat_out)  # type: ignore[assignment]
        assert_type(result, Tensor[B, OutD])
        return result


class MaskNetBackbone[F, D, OutD](nn.Module):
    """MaskNet backbone: instance-guided feature masking.

    Args:
        num_features: Number of feature fields (F).
        emb_dim: Embedding dimension per feature (D).
        output_dim: Output dimension of the backbone.
        config: MaskNetConfig with architecture hyperparameters.
    """

    def __init__(
        self,
        num_features: Dim[F],
        emb_dim: Dim[D],
        output_dim: Dim[OutD],
        config: MaskNetConfig | None = None,
    ) -> None:
        super().__init__()
        if config is None:
            config = MaskNetConfig()

        # LCE compression: [B, F, D] -> [B, compression_num, D] via linear on feature dim
        self.compression_num = config.compression_num
        self.lce: nn.Linear | None = None
        if config.compression_num > 0:
            self.lce = nn.Linear(num_features, config.compression_num)
            effective_num_features = config.compression_num
        else:
            effective_num_features = int(num_features)

        input_dim = effective_num_features * int(emb_dim)

        # mask_net typed as nn.Module — branch creates union of Serial/Parallel
        self.mask_net: nn.Module
        if config.model_type == "SerialMaskNet":
            self.mask_net = SerialMaskNet(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_units=config.hidden_units,
                hidden_activation=config.hidden_activation,
                reduction_ratio=config.reduction_ratio,
                dropout_rate=config.dropout,
                layer_norm=config.layer_norm,
            )
        elif config.model_type == "ParallelMaskNet":
            self.mask_net = ParallelMaskNet(
                input_dim=input_dim,
                output_dim=output_dim,
                num_blocks=config.num_blocks,
                block_dim=config.block_dim,
                dnn_hidden_units=config.hidden_units,
                hidden_activation=config.hidden_activation,
                reduction_ratio=config.reduction_ratio,
                dropout_rate=config.dropout,
                layer_norm=config.layer_norm,
            )
        else:
            raise ValueError(
                f"model_type={config.model_type} not supported. "
                "Choose from: SerialMaskNet, ParallelMaskNet"
            )

        self._output_dim = output_dim

        # Per-field LayerNorm for V_hidden
        self.num_features = effective_num_features
        self.emb_dim = emb_dim
        self.emb_norm = nn.ModuleList(
            nn.LayerNorm(int(emb_dim)) for _ in range(effective_num_features)
        )

    @property
    def output_dim(self) -> Dim[OutD]:
        return self._output_dim

    def forward[B](self, input_embs: Tensor[B, F, D]) -> Tensor[B, OutD]:
        # LCE compression: [B, F, D] -> [B, compression_num, D]
        if self.lce is not None:
            # lce compresses F to compression_num (int from config — Unknown)
            input_embs = self.lce(input_embs.transpose(1, 2)).transpose(1, 2)

        # Per-field LayerNorm to produce V_hidden — cat with list from unbind loses shapes
        feat_list = input_embs.unbind(dim=1)
        assert_type(feat_list, tuple[Tensor[B, D], ...])
        V_hidden: Tensor = torch.cat(
            [self.emb_norm[i](feat) for i, feat in enumerate(feat_list)], dim=1
        )
        assert_type(V_hidden, Tensor)

        # flatten: D*F (non-LCE) or unknown (LCE) — union type from if/else branch
        V_emb = input_embs.flatten(start_dim=1)

        # mask_net is nn.Module — forward returns Any
        result: Tensor[B, OutD] = self.mask_net(V_emb, V_hidden)  # type: ignore[assignment]
        assert_type(result, Tensor[B, OutD])
        return result


# ============================================================================
# Smoke Tests
# ============================================================================


def test_mask_block():
    """Test MaskBlock: input(128) x hidden(128) -> output(64)."""
    block = MaskBlock(128, 128, 64, 64)
    v_emb: Tensor[4, 128] = torch.randn(4, 128)
    v_hidden: Tensor[4, 128] = torch.randn(4, 128)
    out = block(v_emb, v_hidden)
    assert_type(out, Tensor[4, 64])


def test_serial_masknet():
    """Test SerialMaskNet: chain of MaskBlocks."""
    net = SerialMaskNet(
        input_dim=128,
        output_dim=64,
        hidden_units=[128, 64],
    )
    v_emb: Tensor[4, 128] = torch.randn(4, 128)
    v_hidden: Tensor[4, 128] = torch.randn(4, 128)
    out = net(v_emb, v_hidden)
    assert_type(out, Tensor[4, 64])


def test_parallel_masknet():
    """Test ParallelMaskNet: parallel MaskBlocks + DNN."""
    net = ParallelMaskNet(
        input_dim=128,
        output_dim=32,
        num_blocks=3,
        block_dim=64,
        dnn_hidden_units=[128, 32],
    )
    v_emb: Tensor[4, 128] = torch.randn(4, 128)
    v_hidden: Tensor[4, 128] = torch.randn(4, 128)
    out = net(v_emb, v_hidden)
    assert_type(out, Tensor[4, 32])


def test_masknet_backbone_serial():
    """End-to-end: SerialMaskNet backbone [B, 10, 16] -> [B, 256]."""
    config = MaskNetConfig(model_type="SerialMaskNet", hidden_units=[512, 256])
    model = MaskNetBackbone(num_features=10, emb_dim=16, output_dim=256, config=config)
    x: Tensor[4, 10, 16] = torch.randn(4, 10, 16)
    out = model(x)
    assert_type(out, Tensor[4, 256])


def test_masknet_backbone_parallel():
    """End-to-end: ParallelMaskNet backbone [B, 10, 16] -> [B, 64]."""
    config = MaskNetConfig(
        model_type="ParallelMaskNet",
        num_blocks=2,
        block_dim=32,
        hidden_units=[64],
    )
    model = MaskNetBackbone(num_features=10, emb_dim=16, output_dim=64, config=config)
    x: Tensor[4, 10, 16] = torch.randn(4, 10, 16)
    out = model(x)
    assert_type(out, Tensor[4, 64])
