# Portions (c) Meta Platforms, Inc. and affiliates.
#
# This source code is adapted from pytorch/benchmark (TorchBenchmark),
# which is licensed under the BSD 3-Clause License:
# https://github.com/pytorch/benchmark/blob/main/LICENSE
#
# Original model: reczoo/FuxiCTR (Apache 2.0 License)
# Reference: Y. Yan et al., "APG: Adaptive Parameter Generation Network
# for Click-Through Rate Prediction," NeurIPS 2023.
#
# This adaptation adds tensor shape type annotations for pyrefly.

# ## Inventory
# - [x] HyperNet.__init__ — Dims: input_dim, output_dim; int: hidden_units (list), activation (str), dropout (float)
# - [x] HyperNet.forward
# - [x] APGLinear.__init__ — Dims: input_dim, output_dim, condition_dim; Dim|None: rank_k, overparam_p; int: bias (bool), generate_bias (bool)
# - [x] APGLinear.forward
# - [x] APGMLP.__init__ — Dims: input_dim, output_dim (bridge); int: hidden_units (list), hidden_activation (str), condition_mode (str)
# - [x] APGMLP.forward
# - [x] APGBackbone.__init__ — Dims: num_features, emb_dim, output_dim (bridge); int: config params
# - [x] APGBackbone.output_dim (property)
# - [x] APGBackbone.forward

from typing import assert_type, TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from torch import Tensor
    from torch_shapes import Dim


class HyperNet[IN, OUT](nn.Module):
    """Small MLP that generates parameters for APG_Linear."""

    def __init__(
        self,
        input_dim: Dim[IN],
        output_dim: Dim[OUT],
        hidden_units: list[int],
        activation: str = "ReLU",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        for h_dim in hidden_units:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(getattr(nn, activation)())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward[B](self, x: Tensor[B, IN]) -> Tensor[B, OUT]:
        # Sequential(*list) returns bare Tensor
        out: Tensor[B, OUT] = self.net(x)
        assert_type(out, Tensor[B, OUT])
        return out


class APGLinear[IN, OUT, CD, RK, OP](nn.Module):
    """Linear layer with adaptively generated weights.

    A hypernet takes a conditioning signal and generates the weight matrix
    (or a low-rank core S for U @ S @ V decomposition).
    """

    def __init__(
        self,
        input_dim: Dim[IN],
        output_dim: Dim[OUT],
        condition_dim: Dim[CD],
        bias: bool = True,
        rank_k: Dim[RK] | None = None,
        overparam_p: Dim[OP] | None = None,
        generate_bias: bool = False,
        hypernet_hidden_units: list[int] | tuple[()] = (),
        hypernet_activation: str = "ReLU",
        hypernet_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.generate_bias = generate_bias
        self.rank_k = rank_k
        self.overparam_p = overparam_p
        self.use_bias = bias

        if rank_k is not None:
            assert rank_k <= min(input_dim, output_dim), (
                f"rank_k={rank_k} must be <= min({input_dim}, {output_dim})"
            )
            if overparam_p is not None:
                assert overparam_p >= rank_k, (
                    f"overparam_p={overparam_p} must be >= rank_k={rank_k}"
                )
                self.U_l = nn.Parameter(
                    nn.init.xavier_normal_(torch.empty(input_dim, overparam_p))
                )
                self.U_r = nn.Parameter(
                    nn.init.xavier_normal_(torch.empty(overparam_p, rank_k))
                )
                self.V_l = nn.Parameter(
                    nn.init.xavier_normal_(torch.empty(rank_k, overparam_p))
                )
                self.V_r = nn.Parameter(
                    nn.init.xavier_normal_(torch.empty(overparam_p, output_dim))
                )
            else:
                self.U = nn.Parameter(
                    nn.init.xavier_normal_(torch.empty(input_dim, rank_k))
                )
                self.V = nn.Parameter(
                    nn.init.xavier_normal_(torch.empty(rank_k, output_dim))
                )
            # hypernet_output_dim is an internal size, not a meaningful tensor dim
            hypernet_output_dim = rank_k * rank_k + int(generate_bias) * output_dim
        else:
            hypernet_output_dim = (
                input_dim * output_dim + int(generate_bias) * output_dim
            )

        self.hypernet = HyperNet(
            input_dim=condition_dim,
            output_dim=hypernet_output_dim,
            hidden_units=list(hypernet_hidden_units),
            activation=hypernet_activation,
            dropout=hypernet_dropout,
        )

        self.bias: Tensor[1, OUT] | None
        if self.use_bias and not self.generate_bias:
            self.bias = nn.Parameter(torch.zeros(1, output_dim))
        else:
            self.bias = None

    def forward[B](
        self, input_h: Tensor[B, IN], condition_z: Tensor[B, CD]
    ) -> Tensor[B, OUT]:
        # weight_S: Tensor[B, Unknown] — int() arithmetic in hypernet_output_dim
        weight_S = self.hypernet(condition_z)
        bias = self.bias
        assert_type(bias, Tensor[1, OUT] | None)

        if self.generate_bias:
            if self.use_bias:
                bias = weight_S[:, : self.output_dim]
                assert_type(bias, Tensor[B, OUT])
            # weight_S: Tensor[B, Any] — upstream Unknown contagion
            weight_S = weight_S[:, self.output_dim :]

        if self.rank_k is not None:
            # weight_S: Tensor[Any, RK, RK] — batch dim Any from upstream
            weight_S = weight_S.reshape(-1, self.rank_k, self.rank_k)
            if self.overparam_p is not None:
                U = torch.matmul(self.U_l, self.U_r)
                assert_type(U, Tensor[IN, RK])
                V = torch.matmul(self.V_l, self.V_r)
                assert_type(V, Tensor[RK, OUT])
            else:
                U = self.U
                assert_type(U, Tensor[IN, RK])
                V = self.V
                assert_type(V, Tensor[RK, OUT])
            h = torch.matmul(input_h, U)
            assert_type(h, Tensor[B, RK])
            h = torch.bmm(h.unsqueeze(1), weight_S).squeeze(1)
            assert_type(h, Tensor[B, RK])
            out = torch.matmul(h, V)
            assert_type(out, Tensor[B, OUT])
        else:
            # weight_S: Tensor[Any, IN, OUT] — batch dim Any from upstream
            weight_S = weight_S.reshape(-1, self.input_dim, self.output_dim)
            out = torch.bmm(input_h.unsqueeze(1), weight_S).squeeze(1)
            assert_type(out, Tensor[B, OUT])

        if bias is not None:
            out = out + bias
            assert_type(out, Tensor[B, OUT])
        return out


class APGMLP[InDim, OutDim](nn.Module):
    """Multi-layer perceptron with APG (Adaptive Parameter Generation) layers.

    Each linear layer's weights are dynamically generated by a hypernet
    conditioned on the input itself (self-wise) or an external signal.
    """

    def __init__(
        self,
        input_dim: Dim[InDim],
        output_dim: Dim[OutDim],
        hidden_units: list[int],
        hidden_activation: str = "ReLU",
        dropout: float = 0.0,
        batch_norm: bool = False,
        condition_mode: str = "self-wise",
        condition_dim: int | None = None,
        rank_k: int | None = None,
        overparam_p: int | None = None,
        generate_bias: bool = True,
        hypernet_hidden_units: list[int] | tuple[()] = (),
        hypernet_activation: str = "ReLU",
        hypernet_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.condition_mode = condition_mode
        self.num_layers = len(hidden_units)
        dims = [input_dim] + hidden_units

        self.apg_linears = nn.ModuleList[nn.Module]()
        self.bns = nn.ModuleList[nn.Module | None]()
        self.acts = nn.ModuleList[nn.Module]()
        self.drops = nn.ModuleList[nn.Module | None]()

        for idx in range(self.num_layers):
            if condition_mode == "self-wise":
                cond_dim = dims[idx]
            else:
                assert condition_dim is not None, (
                    "condition_dim required for group-wise mode"
                )
                cond_dim = condition_dim

            self.apg_linears.append(
                APGLinear(
                    input_dim=dims[idx],
                    output_dim=dims[idx + 1],
                    condition_dim=cond_dim,
                    bias=True,
                    rank_k=rank_k,
                    overparam_p=overparam_p,
                    generate_bias=generate_bias,
                    hypernet_hidden_units=hypernet_hidden_units,
                    hypernet_activation=hypernet_activation,
                    hypernet_dropout=hypernet_dropout,
                )
            )
            if batch_norm:
                self.bns.append(nn.BatchNorm1d(dims[idx + 1]))
            else:
                self.bns.append(None)
            self.acts.append(getattr(nn, hidden_activation)())
            if dropout > 0:
                self.drops.append(nn.Dropout(dropout))
            else:
                self.drops.append(None)

        self.output_dim = output_dim

    def forward[B](
        self, x: Tensor[B, InDim], condition_z: Tensor | None = None
    ) -> Tensor[B, OutDim]:
        # ModuleList iteration with heterogeneous modules → Any
        h: Tensor = x
        for idx in range(self.num_layers):
            if self.condition_mode == "self-wise":
                h = self.apg_linears[idx](h, h)
            else:
                h = self.apg_linears[idx](h, condition_z)
            bn = self.bns[idx]
            if bn is not None:
                h = bn(h)
            h = self.acts[idx](h)
            drop = self.drops[idx]
            if drop is not None:
                h = drop(h)
        out: Tensor[B, OutDim] = h  # annotation fallback
        assert_type(out, Tensor[B, OutDim])
        return out


class APGBackbone[F, D, OutD](nn.Module):
    """APG backbone: MLP with adaptively generated weights.

    Takes [B, F, D] embedding tensor, flattens to [B, F*D], processes
    through APG_MLP with instance-wise weight generation.

    Supports multi-layer stacking: each layer processes [B, F, D] -> [B, out_dim],
    then reshapes to [B, K, D'] for the next layer.
    """

    def __init__(
        self,
        num_features: Dim[F],
        emb_dim: Dim[D],
        output_dim: Dim[OutD],
        config: dict | None = None,
        hidden_units: list[int] | None = None,
        hidden_activation: str = "ReLU",
        dropout: float = 0.0,
        batch_norm: bool = False,
        condition_mode: str = "self-wise",
        rank_k: int | None = None,
        overparam_p: int | None = None,
        generate_bias: bool = True,
        hypernet_hidden_units: list[int] | tuple[()] = (),
        hypernet_activation: str = "ReLU",
        hypernet_dropout: float = 0.0,
        num_layers: int = 1,
        num_output_features: int = 16,
    ) -> None:
        super().__init__()
        if hidden_units is None:
            hidden_units = [512, 256]
        self.num_stacked_layers = num_layers
        self.num_output_features = num_output_features

        self.apg_layers = nn.ModuleList[nn.Module]()
        self.reshape_projs = nn.ModuleList[nn.Module]()
        self.layer_norms = nn.ModuleList[nn.Module]()

        cur_num_features: int = num_features
        cur_emb_dim: int = emb_dim

        for i in range(self.num_stacked_layers):
            input_dim = cur_num_features * cur_emb_dim

            last_hidden = hidden_units[-1]
            apg_mlp = APGMLP(
                input_dim=input_dim,
                output_dim=last_hidden,
                hidden_units=hidden_units,
                hidden_activation=hidden_activation,
                dropout=dropout,
                batch_norm=batch_norm,
                condition_mode=condition_mode,
                condition_dim=(input_dim if condition_mode != "self-wise" else None),
                rank_k=rank_k,
                overparam_p=overparam_p,
                generate_bias=generate_bias,
                hypernet_hidden_units=hypernet_hidden_units,
                hypernet_activation=hypernet_activation,
                hypernet_dropout=hypernet_dropout,
            )
            self.apg_layers.append(apg_mlp)

            if i < self.num_stacked_layers - 1:
                out_dim = last_hidden
                K = num_output_features
                target_dim = K * cur_emb_dim
                if out_dim != target_dim:
                    self.reshape_projs.append(nn.Linear(out_dim, target_dim))
                else:
                    self.reshape_projs.append(nn.Identity())
                self.layer_norms.append(nn.LayerNorm(cur_emb_dim))
                cur_num_features = K

        self._output_dim = output_dim

    @property
    def output_dim(self) -> Dim[OutD]:
        return self._output_dim

    def forward[B](self, input_embs: Tensor[B, F, D]) -> Tensor[B, OutD]:
        # ModuleList iteration in loop → All locals Any
        x: Tensor = input_embs
        for i in range(self.num_stacked_layers):
            flat = x.flatten(start_dim=1)
            out = self.apg_layers[i](flat)

            if i < self.num_stacked_layers - 1:
                out = self.reshape_projs[i](out)
                x = out.view(out.size(0), self.num_output_features, -1)
                x = self.layer_norms[i](x)
            else:
                x = out

        result: Tensor[B, OutD] = x  # annotation fallback
        assert_type(result, Tensor[B, OutD])
        return result


def test_hypernet() -> None:
    net = HyperNet(input_dim=32, output_dim=64, hidden_units=[128])
    x = torch.randn(8, 32)
    out = net(x)
    assert_type(out, Tensor[8, 64])


def test_apg_linear_full_rank() -> None:
    layer = APGLinear(input_dim=16, output_dim=8, condition_dim=16)
    h = torch.randn(4, 16)
    z = torch.randn(4, 16)
    out = layer(h, z)
    assert_type(out, Tensor[4, 8])


def test_apg_linear_low_rank() -> None:
    layer = APGLinear(
        input_dim=16,
        output_dim=8,
        condition_dim=16,
        rank_k=4,
        generate_bias=True,
    )
    h = torch.randn(4, 16)
    z = torch.randn(4, 16)
    out = layer(h, z)
    assert_type(out, Tensor[4, 8])


def test_apg_backbone() -> None:
    backbone = APGBackbone(
        num_features=10,
        emb_dim=32,
        output_dim=256,
        hidden_units=[512, 256],
        num_layers=1,
    )
    embs = torch.randn(4, 10, 32)
    out = backbone(embs)
    assert_type(out, Tensor[4, 256])
