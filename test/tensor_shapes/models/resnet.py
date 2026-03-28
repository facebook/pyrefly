# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Phlippe ResNet from TorchBenchmark with shape annotations.

Original: pytorch/benchmark/torchbenchmark/models/phlippe_resnet/__init__.py
"""

from typing import Any, assert_type, overload, TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from torch import Tensor
    from torch_shapes import Dim

# A no-arg factory that produces a shape-preserving activation module.
# Each member's forward signature is Tensor[*S] -> Tensor[*S], so
# Sequential chaining and direct calls both preserve shapes.
ShapePreservingActivation = (
    type[nn.ReLU] | type[nn.GELU] | type[nn.SiLU] | type[nn.Tanh]
)

# ============================================================================
# Blocks
# ============================================================================


class ResNetBlock[C](nn.Module):
    """Shape-preserving residual block: (B, C, H, W) -> (B, C, H, W)."""

    def __init__(self, c: Dim[C], act_fn: ShapePreservingActivation) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            act_fn(),
            nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c),
        )
        self.act_fn = act_fn()

    def forward[B, H, W](self, x: Tensor[B, C, H, W]) -> Tensor[B, C, H, W]:
        z = self.net(x)
        assert_type(z, Tensor[B, C, H, W])
        out = z + x
        out = self.act_fn(out)
        return out


class ResNetDownsampleBlock[C_in, C_out](nn.Module):
    """Downsampling residual block: (B, C_in, H, W) -> (B, C_out, H', W')."""

    def __init__(
        self,
        c_in: Dim[C_in],
        c_out: Dim[C_out],
        act_fn: ShapePreservingActivation,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(c_out),
            act_fn(),
            nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
        )
        self.downsample = nn.Conv2d(c_in, c_out, kernel_size=1, stride=2)
        self.act_fn = act_fn()

    def forward[B, H, W](
        self, x: Tensor[B, C_in, H, W]
    ) -> Tensor[B, C_out, (H - 1) // 2 + 1, (W - 1) // 2 + 1]:
        z = self.net(x)
        assert_type(z, Tensor[B, C_out, (H - 1) // 2 + 1, (W - 1) // 2 + 1])
        skip = self.downsample(x)
        assert_type(skip, Tensor[B, C_out, (H - 1) // 2 + 1, (W - 1) // 2 + 1])
        out = z + skip
        out = self.act_fn(out)
        return out


class ResNetGroup[C](nn.Module):
    """A group of shape-preserving ResNet blocks at channel C."""

    def __init__(
        self,
        c: Dim[C],
        num_blocks: int,
        act_fn: ShapePreservingActivation,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([ResNetBlock(c, act_fn) for _ in range(num_blocks)])

    def forward[B, H, W](self, x: Tensor[B, C, H, W]) -> Tensor[B, C, H, W]:
        for block in self.blocks:
            x = block(x)
        return x


# ============================================================================
# Model
# ============================================================================


class ResNetModel[NumClasses](nn.Module):
    """ResNet with recursive downsample chain (Pattern C — exponential).

    Architecture:
    - Input: Conv2d(3, 16, 3, padding=1) + BN + ReLU
    - Initial group: 3 shape-preserving blocks at 16 channels
    - Downsample chain (2 stages): 16→32→64, spatial 32→16→8
        Each stage: ResNetDownsampleBlock(C, 2C) + ResNetGroup(2C, 2 blocks)
    - Output: AdaptiveAvgPool2d(1,1) + Flatten + Linear(64, num_classes)

    The 2 downsample stages use _chain with return type
    Tensor[B, C * 2**I, (H-1) // 2**I + 1, (W-1) // 2**I + 1].
    """

    def __init__(
        self,
        num_classes: Dim[NumClasses],
        act_fn_name: str = "relu",
    ):
        super().__init__()
        self.act_fn_name = act_fn_name
        act_fn = nn.ReLU

        # Input convolution: 3 → 16
        self.input_net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            act_fn(),
        )

        # Initial group: 3 shape-preserving blocks at 16 channels
        self.initial_group = ResNetGroup(16, 3, act_fn)

        # Downsample stages: each doubles channels + 2 shape-preserving blocks
        downs: list[ResNetDownsampleBlock[Any, Any]] = [
            ResNetDownsampleBlock(16, 32, act_fn),
            ResNetDownsampleBlock(32, 64, act_fn),
        ]
        self.downs = nn.ModuleList(downs)
        groups: list[ResNetGroup[Any]] = [
            ResNetGroup(32, 2, act_fn),
            ResNetGroup(64, 2, act_fn),
        ]
        self.groups = nn.ModuleList(groups)

        # Output: AdaptiveAvgPool → Flatten → Linear
        self.output_net = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, num_classes),
        )

        self._init_params()

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity=self.act_fn_name
                )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _apply_stage[B, C, H, W](
        self, x: Tensor[B, C, H, W], depth: int
    ) -> Tensor[B, 2 * C, (H - 1) // 2 + 1, (W - 1) // 2 + 1]:
        idx = len(self.downs) - depth
        down: ResNetDownsampleBlock[C, 2 * C] = self.downs[idx]
        group: ResNetGroup[2 * C] = self.groups[idx]
        y = down(x)
        return group(y)

    @overload
    def _chain[B, C, H, W](
        self, x: Tensor[B, C, H, W], depth: Dim[1]
    ) -> Tensor[B, 2 * C, (H - 1) // 2 + 1, (W - 1) // 2 + 1]: ...

    @overload
    def _chain[I, B, C, H, W](
        self, x: Tensor[B, C, H, W], depth: Dim[I]
    ) -> Tensor[B, C * 2**I, (H - 1) // 2**I + 1, (W - 1) // 2**I + 1]: ...

    def _chain[I, B, C, H, W](
        self, x: Tensor[B, C, H, W], depth: Dim[I]
    ) -> (
        Tensor[B, 2 * C, (H - 1) // 2 + 1, (W - 1) // 2 + 1]
        | Tensor[B, C * 2**I, (H - 1) // 2**I + 1, (W - 1) // 2**I + 1]
    ):
        y = self._apply_stage(x, depth)
        if depth == 1:
            return y
        return self._chain(y, depth - 1)

    def forward[B](self, x: Tensor[B, 3, 32, 32]) -> Tensor[B, NumClasses]:
        x1 = self.input_net(x)
        assert_type(x1, Tensor[B, 16, 32, 32])
        x2 = self.initial_group(x1)
        assert_type(x2, Tensor[B, 16, 32, 32])
        x3 = self._chain(x2, 2)  # 16→64, 32→8
        assert_type(x3, Tensor[B, 64, 8, 8])
        return self.output_net(x3)


# ============================================================================
# Smoke tests
# ============================================================================


def test_resnet_block():
    """Test shape-preserving block."""
    block = ResNetBlock(16, act_fn=nn.ReLU)
    x: Tensor[4, 16, 32, 32] = torch.randn(4, 16, 32, 32)
    out = block(x)
    assert_type(out, Tensor[4, 16, 32, 32])


def test_resnet_group():
    """Test group of shape-preserving blocks."""
    group = ResNetGroup(32, num_blocks=3, act_fn=nn.ReLU)
    x: Tensor[4, 32, 16, 16] = torch.randn(4, 32, 16, 16)
    out = group(x)
    assert_type(out, Tensor[4, 32, 16, 16])


def test_resnet_block_gelu():
    """Test block with GELU activation — enabled by nn.Module as Callable."""
    block = ResNetBlock(16, act_fn=nn.GELU)
    x: Tensor[4, 16, 32, 32] = torch.randn(4, 16, 32, 32)
    out = block(x)
    assert_type(out, Tensor[4, 16, 32, 32])


def test_resnet_block_tanh():
    """Test block with Tanh activation."""
    block = ResNetBlock(16, act_fn=nn.Tanh)
    x: Tensor[4, 16, 32, 32] = torch.randn(4, 16, 32, 32)
    out = block(x)
    assert_type(out, Tensor[4, 16, 32, 32])


def test_resnet_model():
    model = ResNetModel(10)
    x: Tensor[128, 3, 32, 32] = torch.randn(128, 3, 32, 32)
    out = model(x)
    assert_type(out, Tensor[128, 10])
