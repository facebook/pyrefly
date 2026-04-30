# Portions (c) Meta Platforms, Inc. and affiliates.
#
# This source code is adapted from pytorch/benchmark (TorchBenchmark),
# which is licensed under the BSD 3-Clause License:
# https://github.com/pytorch/benchmark/blob/main/LICENSE
#
# Original model: pytorch/vision (torchvision/models/mobilenetv2.py)
# Reference: Sandler et al., "MobileNetV2: Inverted Residuals and Linear
# Bottlenecks," CVPR 2018 (arXiv 1801.04381).
#
# This adaptation adds tensor shape type annotations for pyrefly.

# ## Inventory
# - [x] _make_divisible — utility, no tensors
# - [x] InvertedResidual.__init__ — Dims: inp(Inp), oup(Oup), expand_ratio(ER), stride(S); dropped norm_layer (Callable erases types)
# - [x] InvertedResidual.forward
# - [x] MobileNetV2.__init__ — Dims: num_classes(NC), last_channel(LC bridge); int: round_nearest; float: width_mult, dropout; dropped: block, norm_layer (Callable erases types)
# - [x] MobileNetV2._forward_impl
# - [x] MobileNetV2.forward

from typing import assert_type, TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from torch import Tensor
    from torch_shapes import Dim


def _make_divisible(v: float, divisor: int, min_value: int | None = None) -> int:
    """Round a value to the nearest multiple of divisor."""
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_value < 0.9 * v:
        new_value += divisor
    return new_value


class InvertedResidual[Inp, Oup, ER, S](nn.Module):
    """MobileNetV2 inverted residual block.

    Restructured from nn.Sequential(*layers) to individual nn.Sequential
    attributes with direct args, enabling shape tracking through each stage.
    """

    def __init__(
        self,
        inp: Dim[Inp],
        oup: Dim[Oup],
        stride: Dim[S],
        expand_ratio: Dim[ER],
    ) -> None:
        super().__init__()
        if stride not in [1, 2]:
            raise ValueError(f"stride should be 1 or 2 instead of {stride}")

        # removed int(round(...)) — no-op on ints, kills Dim tracking
        hidden_dim = inp * expand_ratio
        self.use_res_connect: bool = stride == 1 and inp == oup
        self.expand_ratio = expand_ratio

        # pw expansion: 1x1 conv to expand channels
        self.expand = nn.Sequential(
            nn.Conv2d(inp, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(),
        )
        # dw: depthwise 3x3 conv
        self.dw = nn.Sequential(
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                3,
                stride=stride,
                padding=1,
                groups=hidden_dim,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(),
        )
        # pw-linear: 1x1 projection, no activation
        self.project = nn.Sequential(
            nn.Conv2d(hidden_dim, oup, 1, bias=False),
            nn.BatchNorm2d(oup),
        )

        self.out_channels = oup
        self._is_cn: bool = stride > 1

    def forward[B, H, W](self, x: Tensor[B, Inp, H, W]) -> Tensor[B, Oup, H, W]:
        out: Tensor[B, Inp * ER, H, W]
        if self.expand_ratio != 1:
            out = self.expand(x)
        else:
            out = x  # type: ignore[assignment]  # conditional: Inp*ER == Inp when ER==1
        assert_type(out, Tensor[B, Inp * ER, H, W])
        out2 = self.dw(out)
        assert_type(out2, Tensor[B, Inp * ER, H, W])
        out3 = self.project(out2)
        assert_type(out3, Tensor[B, Oup, H, W])
        if self.use_res_connect:
            return x + out3  # type: ignore[return-value]  # conditional: Inp==Oup
        return out3


class MobileNetV2[NC = 1000, LC = 1280](nn.Module):
    """MobileNet V2 main class.

    Bridge dim LC (last_channel) connects the untracked feature extractor
    (built via nn.Sequential(*list)) to the typed classifier, recovering
    Tensor[B, NC] at the output.
    """

    def __init__(
        self,
        num_classes: Dim[NC] = 1000,
        width_mult: float = 1.0,
        inverted_residual_setting: list[list[int]] | None = None,
        round_nearest: int = 8,
        dropout: float = 0.2,
        last_channel: Dim[LC] = 1280,
    ) -> None:
        super().__init__()

        input_channel = 32
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(
            last_channel * max(1.0, width_mult), round_nearest
        )

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        if (
            len(inverted_residual_setting) == 0
            or len(inverted_residual_setting[0]) != 4
        ):
            raise ValueError(
                f"inverted_residual_setting should be non-empty or "
                f"a 4-element list, got {inverted_residual_setting}"
            )

        # features built as list -> nn.Sequential(*features), shapes lost
        features: list[nn.Module] = [
            nn.Sequential(
                nn.Conv2d(3, input_channel, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(input_channel),
                nn.ReLU6(),
            )
        ]
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(
                    InvertedResidual(
                        input_channel, output_channel, stride, expand_ratio=t
                    )
                )
                input_channel = output_channel
        features.append(
            nn.Sequential(
                nn.Conv2d(input_channel, self.last_channel, 1, bias=False),
                nn.BatchNorm2d(self.last_channel),
                nn.ReLU6(),
            )
        )
        self.features = nn.Sequential(*features)

        # classifier: direct Sequential args — tracked
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)  # type: ignore[arg-type]
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)  # type: ignore[arg-type]

    def _forward_impl[B, H, W](self, x: Tensor[B, 3, H, W]) -> Tensor[B, NC]:
        feat = self.features(x)
        assert_type(feat, Tensor)  # Sequential(*list) — bare
        pooled = nn.functional.adaptive_avg_pool2d(feat, (1, 1))
        assert_type(pooled, Tensor)  # input bare — upstream contagion
        flat = torch.flatten(pooled, 1)
        assert_type(flat, Tensor)  # input bare — upstream contagion
        # annotation fallback: classifier is Sequential(Dropout, Linear[LC, NC])
        out: Tensor[B, NC] = self.classifier(flat)
        assert_type(out, Tensor[B, NC])
        return out

    def forward[B, H, W](self, x: Tensor[B, 3, H, W]) -> Tensor[B, NC]:
        return self._forward_impl(x)


def test_inverted_residual() -> None:
    block = InvertedResidual(32, 64, stride=1, expand_ratio=6)
    x: Tensor[1, 32, 112, 112] = torch.randn(1, 32, 112, 112)
    out = block(x)
    assert_type(out, Tensor[1, 64, 112, 112])


def test_mobilenetv2() -> None:
    model = MobileNetV2(num_classes=1000)
    x: Tensor[1, 3, 224, 224] = torch.randn(1, 3, 224, 224)
    out = model(x)
    assert_type(out, Tensor[1, 1000])
