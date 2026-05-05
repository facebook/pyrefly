# Portions (c) Meta Platforms, Inc. and affiliates.
#
# This source code is adapted from pytorch/benchmark (TorchBenchmark),
# which is licensed under the BSD 3-Clause License:
# https://github.com/pytorch/benchmark/blob/main/LICENSE
#
# Original model: Hzzone/pytorch-openpose
# Reference: Cao et al., "Realtime Multi-Person 2D Pose Estimation
# using Part Affinity Fields," CVPR 2017 (arXiv 1611.08050)
#
# This adaptation adds tensor shape type annotations for pyrefly.

# ## Inventory
# - [x] make_layers — utility, builds nn.Sequential from OrderedDict (retained for faithfulness, unused by typed models)
# - [x] bodypose_model.__init__ — no Dim params (all channels hardcoded)
# - [x] bodypose_model.forward
# - [x] handpose_model.__init__ — no Dim params (all channels hardcoded)
# - [x] handpose_model.forward

from collections import OrderedDict
from typing import assert_type, TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from torch import Tensor


def make_layers(
    block: dict[str, list[int]], no_relu_layers: list[str]
) -> nn.Sequential:
    layers: list[tuple[str, nn.Module]] = []
    for layer_name, v in block.items():
        if "pool" in layer_name:
            layer = nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2])
            layers.append((layer_name, layer))
        else:
            conv2d = nn.Conv2d(
                in_channels=v[0],
                out_channels=v[1],
                kernel_size=v[2],
                stride=v[3],
                padding=v[4],
            )
            layers.append((layer_name, conv2d))
            if layer_name not in no_relu_layers:
                layers.append(("relu_" + layer_name, nn.ReLU(inplace=True)))

    return nn.Sequential(OrderedDict(layers))


class bodypose_model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # Block 0: VGG backbone — 3 → 128 channels, 3 MaxPools halving spatial dims
        self.model0 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
        )

        # Stage 1 L1: 128 → 38 channels (no ReLU on last conv)
        self.model1_1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 512, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 38, 1, 1, 0),
        )

        # Stage 1 L2: 128 → 19 channels (no ReLU on last conv)
        self.model1_2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 512, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 19, 1, 1, 0),
        )

        # Stages 2-6 L1: 185 → 38 channels (no ReLU on last conv)
        self.model2_1 = nn.Sequential(
            nn.Conv2d(185, 128, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 38, 1, 1, 0),
        )
        self.model3_1 = nn.Sequential(
            nn.Conv2d(185, 128, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 38, 1, 1, 0),
        )
        self.model4_1 = nn.Sequential(
            nn.Conv2d(185, 128, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 38, 1, 1, 0),
        )
        self.model5_1 = nn.Sequential(
            nn.Conv2d(185, 128, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 38, 1, 1, 0),
        )
        self.model6_1 = nn.Sequential(
            nn.Conv2d(185, 128, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 38, 1, 1, 0),
        )

        # Stages 2-6 L2: 185 → 19 channels (no ReLU on last conv)
        self.model2_2 = nn.Sequential(
            nn.Conv2d(185, 128, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 19, 1, 1, 0),
        )
        self.model3_2 = nn.Sequential(
            nn.Conv2d(185, 128, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 19, 1, 1, 0),
        )
        self.model4_2 = nn.Sequential(
            nn.Conv2d(185, 128, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 19, 1, 1, 0),
        )
        self.model5_2 = nn.Sequential(
            nn.Conv2d(185, 128, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 19, 1, 1, 0),
        )
        # Stage 6 L2: original bug — Mconv7_stage6_L2 missing from no_relu_layers,
        # so last conv gets a ReLU (Mconv7_stage6_L1 listed twice instead)
        self.model6_2 = nn.Sequential(
            nn.Conv2d(185, 128, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 19, 1, 1, 0),
            nn.ReLU(inplace=True),
        )

    def forward[B, H, W](
        self, x: Tensor[B, 3, H, W]
    ) -> tuple[Tensor[B, 38, H // 8, W // 8], Tensor[B, 19, H // 8, W // 8]]:
        out1 = self.model0(x)
        assert_type(out1, Tensor[B, 128, H // 8, W // 8])

        out1_1 = self.model1_1(out1)
        assert_type(out1_1, Tensor[B, 38, H // 8, W // 8])
        out1_2 = self.model1_2(out1)
        assert_type(out1_2, Tensor[B, 19, H // 8, W // 8])
        out2 = torch.cat((out1_1, out1_2, out1), 1)
        assert_type(out2, Tensor[B, 185, H // 8, W // 8])

        out2_1 = self.model2_1(out2)
        assert_type(out2_1, Tensor[B, 38, H // 8, W // 8])
        out2_2 = self.model2_2(out2)
        assert_type(out2_2, Tensor[B, 19, H // 8, W // 8])
        out3 = torch.cat((out2_1, out2_2, out1), 1)
        assert_type(out3, Tensor[B, 185, H // 8, W // 8])

        out3_1 = self.model3_1(out3)
        assert_type(out3_1, Tensor[B, 38, H // 8, W // 8])
        out3_2 = self.model3_2(out3)
        assert_type(out3_2, Tensor[B, 19, H // 8, W // 8])
        out4 = torch.cat((out3_1, out3_2, out1), 1)
        assert_type(out4, Tensor[B, 185, H // 8, W // 8])

        out4_1 = self.model4_1(out4)
        assert_type(out4_1, Tensor[B, 38, H // 8, W // 8])
        out4_2 = self.model4_2(out4)
        assert_type(out4_2, Tensor[B, 19, H // 8, W // 8])
        out5 = torch.cat((out4_1, out4_2, out1), 1)
        assert_type(out5, Tensor[B, 185, H // 8, W // 8])

        out5_1 = self.model5_1(out5)
        assert_type(out5_1, Tensor[B, 38, H // 8, W // 8])
        out5_2 = self.model5_2(out5)
        assert_type(out5_2, Tensor[B, 19, H // 8, W // 8])
        out6 = torch.cat((out5_1, out5_2, out1), 1)
        assert_type(out6, Tensor[B, 185, H // 8, W // 8])

        out6_1 = self.model6_1(out6)
        assert_type(out6_1, Tensor[B, 38, H // 8, W // 8])
        out6_2 = self.model6_2(out6)
        assert_type(out6_2, Tensor[B, 19, H // 8, W // 8])

        return out6_1, out6_2


class handpose_model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # Block 1_0: VGG backbone — 3 → 128 channels, 3 MaxPools halving spatial dims
        self.model1_0 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
        )

        # Block 1_1: 128 → 22 channels (no ReLU on last conv)
        self.model1_1 = nn.Sequential(
            nn.Conv2d(128, 512, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 22, 1, 1, 0),
        )

        # Stages 2-6: 150 → 22 channels (no ReLU on last conv)
        self.model2 = nn.Sequential(
            nn.Conv2d(150, 128, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 22, 1, 1, 0),
        )
        self.model3 = nn.Sequential(
            nn.Conv2d(150, 128, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 22, 1, 1, 0),
        )
        self.model4 = nn.Sequential(
            nn.Conv2d(150, 128, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 22, 1, 1, 0),
        )
        self.model5 = nn.Sequential(
            nn.Conv2d(150, 128, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 22, 1, 1, 0),
        )
        self.model6 = nn.Sequential(
            nn.Conv2d(150, 128, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 22, 1, 1, 0),
        )

    def forward[B, H, W](self, x: Tensor[B, 3, H, W]) -> Tensor[B, 22, H // 8, W // 8]:
        out1_0 = self.model1_0(x)
        assert_type(out1_0, Tensor[B, 128, H // 8, W // 8])
        out1_1 = self.model1_1(out1_0)
        assert_type(out1_1, Tensor[B, 22, H // 8, W // 8])
        concat_stage2 = torch.cat((out1_1, out1_0), 1)
        assert_type(concat_stage2, Tensor[B, 150, H // 8, W // 8])
        out_stage2 = self.model2(concat_stage2)
        assert_type(out_stage2, Tensor[B, 22, H // 8, W // 8])
        concat_stage3 = torch.cat((out_stage2, out1_0), 1)
        assert_type(concat_stage3, Tensor[B, 150, H // 8, W // 8])
        out_stage3 = self.model3(concat_stage3)
        assert_type(out_stage3, Tensor[B, 22, H // 8, W // 8])
        concat_stage4 = torch.cat((out_stage3, out1_0), 1)
        assert_type(concat_stage4, Tensor[B, 150, H // 8, W // 8])
        out_stage4 = self.model4(concat_stage4)
        assert_type(out_stage4, Tensor[B, 22, H // 8, W // 8])
        concat_stage5 = torch.cat((out_stage4, out1_0), 1)
        assert_type(concat_stage5, Tensor[B, 150, H // 8, W // 8])
        out_stage5 = self.model5(concat_stage5)
        assert_type(out_stage5, Tensor[B, 22, H // 8, W // 8])
        concat_stage6 = torch.cat((out_stage5, out1_0), 1)
        assert_type(concat_stage6, Tensor[B, 150, H // 8, W // 8])
        out_stage6 = self.model6(concat_stage6)
        assert_type(out_stage6, Tensor[B, 22, H // 8, W // 8])
        return out_stage6


def test_bodypose():
    model = bodypose_model()
    x: Tensor[1, 3, 368, 368] = torch.randn(1, 3, 368, 368)
    out1, out2 = model(x)
    assert_type(out1, Tensor[1, 38, 46, 46])
    assert_type(out2, Tensor[1, 19, 46, 46])


def test_handpose():
    model = handpose_model()
    x: Tensor[1, 3, 368, 368] = torch.randn(1, 3, 368, 368)
    out = model(x)
    assert_type(out, Tensor[1, 22, 46, 46])
