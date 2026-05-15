# Portions (c) Meta Platforms, Inc. and affiliates.
#
# This source code is adapted from pytorch/vision (torchvision),
# which is licensed under the BSD 3-Clause License:
# https://github.com/pytorch/vision/blob/main/LICENSE
#
# This adaptation adds tensor shape type annotations for pyrefly.

"""
SqueezeNet 1.0 from torchvision with shape annotations.

Original: pytorch/vision/torchvision/models/squeezenet.py

MaxPool2d ceil_mode=True is not captured by DSL (uses floor formula),
but spatial dims collapse to 1x1 via AdaptiveAvgPool2d so the
off-by-one in intermediate spatial dims does not affect the output; the
gap is in the typing of `features`.
"""

from typing import assert_type, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.init as init

if TYPE_CHECKING:
    from torch import Tensor
    from torch_shapes import Dim


class Fire[InC, SQ, E1, E3](nn.Module):
    """Fire module: squeeze (1x1 conv) then expand (parallel 1x1 + 3x3 convs).

    Input:  Tensor[B, InC, H, W]
    Output: Tensor[B, E1 + E3, H, W]

    The squeeze conv reduces InC -> SQ channels, then two parallel expand
    convs produce E1 and E3 channels respectively, concatenated along dim 1.
    """

    def __init__(
        self,
        inplanes: Dim[InC],
        squeeze_planes: Dim[SQ],
        expand1x1_planes: Dim[E1],
        expand3x3_planes: Dim[E3],
    ) -> None:
        super().__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(
            squeeze_planes, expand3x3_planes, kernel_size=3, padding=1
        )
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward[B, H, W](self, x: Tensor[B, InC, H, W]) -> Tensor[B, E1 + E3, H, W]:
        x1 = self.squeeze_activation(self.squeeze(x))
        assert_type(x1, Tensor[B, SQ, H, W])
        e1 = self.expand1x1_activation(self.expand1x1(x1))
        assert_type(e1, Tensor[B, E1, H, W])
        e3 = self.expand3x3_activation(self.expand3x3(x1))
        assert_type(e3, Tensor[B, E3, H, W])
        result = torch.cat((e1, e3), 1)
        assert_type(result, Tensor[B, E1 + E3, H, W])
        return result


class SqueezeNet[NC = 1000](nn.Module):
    """SqueezeNet 1.0 architecture.

    Input:  Tensor[B, 3, H, W]
    Output: Tensor[B, NC]

    Uses concrete channel dimensions throughout since the Fire module
    channel progression is fixed by architecture design.
    """

    def __init__(self, num_classes: Dim[NC] = 1000, dropout: float = 0.5) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=7, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(96, 16, 64, 64),
            Fire(128, 16, 64, 64),
            Fire(128, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(256, 32, 128, 128),
            Fire(256, 48, 192, 192),
            Fire(384, 48, 192, 192),
            Fire(384, 64, 256, 256),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(512, 64, 256, 256),
        )

        final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward[B, H, W](self, x: Tensor[B, 3, H, W]) -> Tensor[B, NC]:
        x1 = self.features(x)
        assert_type(
            x1,
            Tensor[
                B,
                512,
                ((-3 + ((-7 + H) // 4)) // 4),
                ((-3 + ((-7 + W) // 4)) // 4),
            ],
        )
        x2 = self.classifier(x1)
        assert_type(x2, Tensor[B, NC, 1, 1])
        result = torch.flatten(x2, 1)
        assert_type(result, Tensor[B, NC])
        return result


# ----------------------------------------------------------------------------
# Smoke tests
# ----------------------------------------------------------------------------


def test_fire():
    """Test Fire module: squeeze + expand with cat."""
    fire = Fire(96, 16, 64, 64)
    x: Tensor[2, 96, 55, 55] = torch.randn(2, 96, 55, 55)
    out = fire(x)
    assert_type(out, Tensor[2, 128, 55, 55])


def test_squeezenet():
    """End-to-end: SqueezeNet 1.0 for ImageNet classification."""
    model = SqueezeNet(num_classes=1000)
    x: Tensor[2, 3, 224, 224] = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert_type(out, Tensor[2, 1000])


def test_squeezenet_custom_classes():
    """SqueezeNet with custom number of classes."""
    model = SqueezeNet(num_classes=10)
    x: Tensor[1, 3, 224, 224] = torch.randn(1, 3, 224, 224)
    out = model(x)
    assert_type(out, Tensor[1, 10])
