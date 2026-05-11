# Portions (c) Meta Platforms, Inc. and affiliates.
#
# This source code is adapted from pytorch/benchmark (TorchBenchmark),
# which is licensed under the BSD 3-Clause License:
# https://github.com/pytorch/benchmark/blob/main/LICENSE
#
# This adaptation adds tensor shape type annotations for pyrefly.

"""
Soft Actor-Critic networks from TorchBenchmark with shape annotations.

Original: pytorch/benchmark/torchbenchmark/models/soft_actor_critic/nets.py

Port notes:
- MLP-based networks (BaselineActor, BaselineCritic, BigCritic,
  BaselineDiscreteCritic) are fully typed with shape annotations
- Pixel encoders (BigPixelEncoder, SmallPixelEncoder) use concrete image
  dimensions (84x84, typical RL usage) since the original computes flattened
  size dynamically via sac_utils.compute_conv_output
- Uses nn.Flatten() for conv-to-linear transition
- Variable reassignment to different shapes uses unique names
- StochasticActor supports both dist_impl branches ("pyd" → SquashedNormal,
  "beta" → BetaDist), returns union type
- GracBaselineActor returns pyd.Normal (two-headed MLP)
- BaselineDiscreteActor returns pyd.categorical.Categorical
- BetaDist ported (TransformedDistribution subclass with nested Transform)
- TanhTransform and SquashedNormal ported from torchbenchmark/util/distribution.py
  (used by StochasticActor's "pyd" branch)
- obs /= 255.0 changed to obs = obs / 255.0 (in-place ops on input)
"""

import math
from typing import assert_type, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd

if TYPE_CHECKING:
    from torch import Tensor
    from torch_shapes import Dim


# ============================================================================
# Distribution Subclasses
# ============================================================================
# TanhTransform and SquashedNormal from torchbenchmark/util/distribution.py
# BetaDist from nets.py


class TanhTransform(pyd.transforms.Transform):
    """Tanh transform with optional clamping for numerical stability."""

    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(
        self, cache_size: int = 1, clamp: tuple[float, float] | None = None
    ) -> None:
        super().__init__(cache_size=cache_size)
        self.clamp = clamp

    @staticmethod
    def atanh[*S](x: Tensor[*S]) -> Tensor[*S]:
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other: object) -> bool:
        return isinstance(other, TanhTransform)

    def _call[*S](self, x: Tensor[*S]) -> Tensor[*S]:
        return x.tanh()

    def _inverse[*S](self, y: Tensor[*S]) -> Tensor[*S]:
        return self.atanh(y if self.clamp is None else y.clamp(*self.clamp))

    def log_abs_det_jacobian[*S](self, x: Tensor[*S], y: Tensor[*S]) -> Tensor[*S]:
        return 2.0 * (math.log(2.0) - x - F.softplus(-2.0 * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    """Normal distribution followed by tanh squashing."""

    def __init__(
        self,
        loc: Tensor,
        scale: Tensor,
        tanh_transform_clamp: tuple[float, float] | None = None,
    ) -> None:
        self.loc = loc
        self.scale = scale
        self.tanh_transform_clamp = tanh_transform_clamp
        self.base_dist = pyd.Normal(loc, scale)
        transforms: list[pyd.transforms.Transform] = [
            TanhTransform(clamp=tanh_transform_clamp)
        ]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self) -> Tensor:
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu


class BetaDist(pyd.transformed_distribution.TransformedDistribution):
    """Beta distribution mapped to [-1, 1] via affine transform."""

    class _BetaDistTransform(pyd.transforms.Transform):
        domain = pyd.constraints.real
        codomain = pyd.constraints.interval(-1.0, 1.0)

        def __init__(self, cache_size: int = 1) -> None:
            super().__init__(cache_size=cache_size)

        def __eq__(self, other: object) -> bool:
            return isinstance(other, BetaDist._BetaDistTransform)

        def _inverse[*S](self, y: Tensor[*S]) -> Tensor[*S]:
            return (y.clamp(-0.99, 0.99) + 1.0) / 2.0

        def _call[*S](self, x: Tensor[*S]) -> Tensor[*S]:
            return (2.0 * x) - 1.0

        def log_abs_det_jacobian[*S](self, x: Tensor[*S], y: Tensor[*S]) -> Tensor[*S]:
            # Constant Jacobian — scalar broadcasts to match input shape
            return torch.tensor(math.log(2.0)).unsqueeze(0)  # type: ignore[bad-return]

    def __init__(self, alpha: Tensor, beta: Tensor) -> None:
        self.base_dist = pyd.beta.Beta(alpha, beta)
        transforms: list[pyd.transforms.Transform] = [self._BetaDistTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self) -> Tensor:
        mu = self.base_dist.mean
        for tr in self.transforms:
            mu = tr(mu)
        return mu


# ============================================================================
# MLP-based Networks
# ============================================================================


class BaselineActor[S, A](nn.Module):
    """Simple MLP actor: state → action.

    Architecture: Linear(S, 400) → ReLU → Linear(400, 400) → ReLU →
                  Linear(400, A) → Tanh
    """

    def __init__(self, state_size: Dim[S], action_size: Dim[A]) -> None:
        super().__init__()
        self.fc1 = nn.Linear(state_size, 400)
        self.fc2 = nn.Linear(400, 400)
        self.out = nn.Linear(400, action_size)

    def forward[B](self, state: Tensor[B, S]) -> Tensor[B, A]:
        h1 = F.relu(self.fc1(state))
        assert_type(h1, Tensor[B, 400])
        h2 = F.relu(self.fc2(h1))
        assert_type(h2, Tensor[B, 400])
        act = torch.tanh(self.out(h2))
        assert_type(act, Tensor[B, A])
        return act


class BaselineCritic[S, A](nn.Module):
    """Simple MLP critic: (state, action) → Q-value.

    Concatenates state and action, then MLP to scalar output.
    Architecture: Cat(S+A) → Linear(S+A, 400) → ReLU →
                  Linear(400, 300) → ReLU → Linear(300, 1)
    """

    def __init__(self, state_size: Dim[S], action_size: Dim[A]) -> None:
        super().__init__()
        self.fc1 = nn.Linear(state_size + action_size, 400)
        self.fc2 = nn.Linear(400, 300)
        self.out = nn.Linear(300, 1)

    def forward[B](self, state: Tensor[B, S], action: Tensor[B, A]) -> Tensor[B, 1]:
        sa = torch.cat((state, action), dim=1)
        h1 = F.relu(self.fc1(sa))
        assert_type(h1, Tensor[B, 400])
        h2 = F.relu(self.fc2(h1))
        assert_type(h2, Tensor[B, 300])
        val = self.out(h2)
        assert_type(val, Tensor[B, 1])
        return val


class BigCritic[S, A, H](nn.Module):
    """Large MLP critic with configurable hidden size.

    Architecture: Cat(S+A) → Linear(S+A, H) → ReLU →
                  Linear(H, H) → ReLU → Linear(H, 1)
    """

    def __init__(
        self,
        state_space_size: Dim[S],
        act_space_size: Dim[A],
        hidden_size: Dim[H],
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(state_space_size + act_space_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward[B](self, state: Tensor[B, S], action: Tensor[B, A]) -> Tensor[B, 1]:
        sa = torch.cat((state, action), dim=1)
        h1 = F.relu(self.fc1(sa))
        assert_type(h1, Tensor[B, H])
        h2 = F.relu(self.fc2(h1))
        assert_type(h2, Tensor[B, H])
        out = self.fc3(h2)
        assert_type(out, Tensor[B, 1])
        return out


class BaselineDiscreteCritic[S, A, H](nn.Module):
    """MLP critic for discrete actions: state → Q-values for each action.

    Architecture: Linear(S, H) → ReLU → Linear(H, H) → ReLU →
                  Linear(H, A)
    """

    def __init__(
        self, obs_shape: Dim[S], action_shape: Dim[A], hidden_size: Dim[H]
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(obs_shape, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, action_shape)

    def forward[B](self, state: Tensor[B, S]) -> Tensor[B, A]:
        h1 = F.relu(self.fc1(state))
        assert_type(h1, Tensor[B, H])
        h2 = F.relu(self.fc2(h1))
        assert_type(h2, Tensor[B, H])
        vals = self.out(h2)
        assert_type(vals, Tensor[B, A])
        return vals


class StochasticActor[S, A, H](nn.Module):
    """Stochastic MLP actor: state → distribution over actions.

    Architecture: Linear(S, H) → ReLU → Linear(H, H) → ReLU →
                  Linear(H, 2*A) → chunk → distribution

    Two dist_impl modes:
    - "pyd": chunk → tanh-clamped log_std → SquashedNormal(mu, std)
    - "beta": softplus → chunk → BetaDist(alpha, beta)
    """

    def __init__(
        self,
        state_space_size: Dim[S],
        act_space_size: Dim[A],
        hidden_size: Dim[H],
        log_std_low: float = -10.0,
        log_std_high: float = 2.0,
        dist_impl: str = "pyd",
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(state_space_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 2 * act_space_size)
        self.log_std_low = log_std_low
        self.log_std_high = log_std_high
        self.dist_impl = dist_impl

    def forward[B](self, state: Tensor[B, S]) -> SquashedNormal | BetaDist:
        h1 = F.relu(self.fc1(state))
        assert_type(h1, Tensor[B, H])
        h2 = F.relu(self.fc2(h1))
        assert_type(h2, Tensor[B, H])
        out = self.fc3(h2)
        assert_type(out, Tensor[B, 2 * A])
        if self.dist_impl == "pyd":
            mu, log_std = out.chunk(2, dim=1)
            assert_type(mu, Tensor[B, A])
            assert_type(log_std, Tensor[B, A])
            log_std_clamped = torch.tanh(log_std)
            assert_type(log_std_clamped, Tensor[B, A])
            log_std_scaled = self.log_std_low + 0.5 * (
                self.log_std_high - self.log_std_low
            ) * (log_std_clamped + 1)
            std = log_std_scaled.exp()
            assert_type(std, Tensor[B, A])
            dist: SquashedNormal | BetaDist = SquashedNormal(
                mu, std, tanh_transform_clamp=(-0.99, 0.99)
            )
        elif self.dist_impl == "beta":
            out_pos = 1.0 + F.softplus(out)
            assert_type(out_pos, Tensor[B, 2 * A])
            alpha, beta = out_pos.chunk(2, dim=1)
            assert_type(alpha, Tensor[B, A])
            assert_type(beta, Tensor[B, A])
            dist = BetaDist(alpha, beta)
        else:
            raise ValueError(f"Unknown dist_impl: {self.dist_impl}")
        return dist


class GracBaselineActor[S, A](nn.Module):
    """Two-headed MLP actor: state → Normal(mean, std).

    Architecture: Linear(S, 400) → ReLU → Linear(400, 300) → ReLU →
                  fc_mean: Linear(300, A) → tanh
                  fc_std:  Linear(300, A) → softplus
    Returns Normal(mean, std).
    """

    def __init__(self, obs_size: Dim[S], action_size: Dim[A]) -> None:
        super().__init__()
        self.fc1 = nn.Linear(obs_size, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc_mean = nn.Linear(300, action_size)
        self.fc_std = nn.Linear(300, action_size)

    def forward[B](self, state: Tensor[B, S]) -> pyd.Normal:
        h1 = F.relu(self.fc1(state))
        assert_type(h1, Tensor[B, 400])
        h2 = F.relu(self.fc2(h1))
        assert_type(h2, Tensor[B, 300])
        mean = torch.tanh(self.fc_mean(h2))
        assert_type(mean, Tensor[B, A])
        std = F.softplus(self.fc_std(h2)) + 1e-3
        assert_type(std, Tensor[B, A])
        return pyd.Normal(mean, std)


class BaselineDiscreteActor[S, A, H](nn.Module):
    """Discrete MLP actor: state → Categorical distribution.

    Architecture: Linear(S, H) → ReLU → Linear(H, H) → ReLU →
                  Linear(H, A) → softmax → Categorical
    """

    def __init__(
        self, obs_shape: Dim[S], action_size: Dim[A], hidden_size: Dim[H]
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(obs_shape, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.act_p = nn.Linear(hidden_size, action_size)

    def forward[B](self, state: Tensor[B, S]) -> pyd.Categorical:
        h1 = F.relu(self.fc1(state))
        assert_type(h1, Tensor[B, H])
        h2 = F.relu(self.fc2(h1))
        assert_type(h2, Tensor[B, H])
        act_p = F.softmax(self.act_p(h2), dim=1)
        assert_type(act_p, Tensor[B, A])
        return pyd.categorical.Categorical(act_p)


# ============================================================================
# Pixel Encoders (concrete 84x84 input)
# ============================================================================


class SmallPixelEncoder[C, OutDim](nn.Module):
    """Small CNN encoder for 84×84 pixel observations.

    Architecture:
    - Conv2d(C, 32, 8, stride=4): 84 → 20
    - Conv2d(32, 64, 4, stride=2): 20 → 9
    - Conv2d(64, 64, 3, stride=1): 9 → 7
    - Flatten: 64 * 7 * 7 = 3136
    - Linear(3136, OutDim)
    """

    def __init__(self, channels: Dim[C], out_dim: Dim[OutDim]) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(3136, out_dim)

    def forward[B](self, obs: Tensor[B, C, 84, 84]) -> Tensor[B, OutDim]:
        # obs = obs / 255.0 omitted (scalar div, doesn't change shape)
        h1 = F.relu(self.conv1(obs))
        assert_type(h1, Tensor[B, 32, 20, 20])
        h2 = F.relu(self.conv2(h1))
        assert_type(h2, Tensor[B, 64, 9, 9])
        h3 = F.relu(self.conv3(h2))
        assert_type(h3, Tensor[B, 64, 7, 7])
        h3_flat = self.flatten(h3)
        assert_type(h3_flat, Tensor[B, 3136])
        state = self.fc(h3_flat)
        assert_type(state, Tensor[B, OutDim])
        return state


class BigPixelEncoder[C, OutDim](nn.Module):
    """Large CNN encoder for 84×84 pixel observations.

    Architecture:
    - Conv2d(C, 32, 3, stride=2): 84 → 41
    - Conv2d(32, 32, 3, stride=1): 41 → 39
    - Conv2d(32, 32, 3, stride=1): 39 → 37
    - Conv2d(32, 32, 3, stride=1): 37 → 35
    - Flatten: 32 * 35 * 35 = 39200
    - Linear(39200, OutDim) → LayerNorm(OutDim) → Tanh
    """

    def __init__(self, channels: Dim[C], out_dim: Dim[OutDim]) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(39200, out_dim)
        self.ln = nn.LayerNorm(out_dim)

    def forward[B](self, obs: Tensor[B, C, 84, 84]) -> Tensor[B, OutDim]:
        h1 = F.relu(self.conv1(obs))
        assert_type(h1, Tensor[B, 32, 41, 41])
        h2 = F.relu(self.conv2(h1))
        assert_type(h2, Tensor[B, 32, 39, 39])
        h3 = F.relu(self.conv3(h2))
        assert_type(h3, Tensor[B, 32, 37, 37])
        h4 = F.relu(self.conv4(h3))
        assert_type(h4, Tensor[B, 32, 35, 35])
        h4_flat = self.flatten(h4)
        assert_type(h4_flat, Tensor[B, 39200])
        h5 = self.fc(h4_flat)
        assert_type(h5, Tensor[B, OutDim])
        h6 = self.ln(h5)
        assert_type(h6, Tensor[B, OutDim])
        state = torch.tanh(h6)
        assert_type(state, Tensor[B, OutDim])
        return state


# ============================================================================
# Smoke tests
# ============================================================================


def test_baseline_actor():
    """Test simple MLP actor: state(24) → action(4)."""
    actor = BaselineActor(24, 4)
    state: Tensor[8, 24] = torch.randn(8, 24)
    act = actor(state)
    assert_type(act, Tensor[8, 4])


def test_baseline_critic():
    """Test simple MLP critic: (state(24), action(4)) → Q-value(1)."""
    critic = BaselineCritic(24, 4)
    state: Tensor[8, 24] = torch.randn(8, 24)
    action: Tensor[8, 4] = torch.randn(8, 4)
    val = critic(state, action)
    assert_type(val, Tensor[8, 1])


def test_big_critic():
    """Test large MLP critic with explicit hidden size."""
    critic = BigCritic(50, 6, 1024)
    state: Tensor[8, 50] = torch.randn(8, 50)
    action: Tensor[8, 6] = torch.randn(8, 6)
    val = critic(state, action)
    assert_type(val, Tensor[8, 1])


def test_baseline_discrete_critic():
    """Test discrete MLP critic: state(128) → Q-values(18)."""
    critic = BaselineDiscreteCritic(128, 18, 300)
    state: Tensor[8, 128] = torch.randn(8, 128)
    vals = critic(state)
    assert_type(vals, Tensor[8, 18])


def test_stochastic_actor():
    """Test stochastic actor MLP: state(50) → distribution (pyd mode)."""
    actor = StochasticActor(50, 6, 1024)
    state: Tensor[8, 50] = torch.randn(8, 50)
    dist = actor(state)
    assert_type(dist, SquashedNormal | BetaDist)


def test_stochastic_actor_beta():
    """Test stochastic actor MLP: state(50) → distribution (beta mode)."""
    actor = StochasticActor(50, 6, 1024, dist_impl="beta")
    state: Tensor[8, 50] = torch.randn(8, 50)
    dist = actor(state)
    assert_type(dist, SquashedNormal | BetaDist)


def test_grac_baseline_actor():
    """Test two-headed actor: state(24) → Normal distribution."""
    actor = GracBaselineActor(24, 4)
    state: Tensor[8, 24] = torch.randn(8, 24)
    dist = actor(state)
    assert_type(dist, pyd.Normal)


def test_baseline_discrete_actor():
    """Test discrete actor: state(128) → Categorical distribution."""
    actor = BaselineDiscreteActor(128, 18, 300)
    state: Tensor[8, 128] = torch.randn(8, 128)
    dist = actor(state)
    assert_type(dist, pyd.categorical.Categorical)


def test_small_pixel_encoder():
    """Test small CNN encoder: (B, 3, 84, 84) → (B, 50)."""
    encoder = SmallPixelEncoder(3, 50)
    obs: Tensor[4, 3, 84, 84] = torch.randn(4, 3, 84, 84)
    state = encoder(obs)
    assert_type(state, Tensor[4, 50])


def test_big_pixel_encoder():
    """Test large CNN encoder: (B, 3, 84, 84) → (B, 50)."""
    encoder = BigPixelEncoder(3, 50)
    obs: Tensor[4, 3, 84, 84] = torch.randn(4, 3, 84, 84)
    state = encoder(obs)
    assert_type(state, Tensor[4, 50])
