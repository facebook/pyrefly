# Portions (c) Meta Platforms, Inc. and affiliates.
#
# This source code is adapted from pytorch/benchmark (TorchBenchmark),
# which is licensed under the BSD 3-Clause License:
# https://github.com/pytorch/benchmark/blob/main/LICENSE
#
# This adaptation adds tensor shape type annotations for pyrefly.

"""
DRQ (Data-Regularized Q) actor-critic from TorchBenchmark with shape annotations.

Original: pytorch/benchmark/torchbenchmark/models/drq/drq.py

Port notes:
- CNN encoder with 4 Conv2d layers on 84x84 input (standard Atari RL resolution),
  followed by flatten → Linear → LayerNorm → tanh. Architecture is identical to
  BigPixelEncoder in soft_actor_critic.py but presented as a standalone module.
- Actor: Encoder → MLP (nn.Sequential) → chunk into (mu, log_std) → constrain
  log_std → construct SquashedNormal(mu, std) distribution.
- Critic: Encoder → cat(features, action) → two parallel MLP networks
  (nn.Sequential) → (q1, q2). Double Q-learning pattern.
- Original mlp() helper returns nn.Sequential(Linear, ReLU, Linear, ReLU, Linear).
  Port preserves this as direct nn.Sequential construction.
- TanhTransform + SquashedNormal: distribution classes from the original
  (torchbenchmark/util/distribution.py). SquashedNormal wraps Normal(mu, std)
  with a TanhTransform. Distribution shapes are not tracked by the type
  checker — these are included for faithfulness.
- DRQAgent: wraps Actor + Critic with shared encoder. Includes
  copy_conv_weights_from for weight sharing, and full training methods
  (update_critic, update_actor_and_alpha, update) from the original.
- obs /= 255.0 changed to obs = obs / 255.0 (in-place ops on input).

Key patterns exercised:
- CNN feature extractor → MLP head (common RL pattern)
- nn.Sequential for MLP pipelines (faithful to original mlp() helper)
- Dual Q-networks: two parallel MLPs producing two scalar outputs
- Concatenation of features + action: Tensor[B, FeatDim + ActDim]
- Chunk operation: Tensor[B, 2*A] → (Tensor[B, A], Tensor[B, A])
- Shared encoder weights via copy_conv_weights_from
- Distribution wrapping (SquashedNormal = Normal + TanhTransform)
- Agent training loop with actor/critic/alpha updates
"""

import math
from typing import Any, assert_type, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, TransformedDistribution
from torch.distributions.transforms import Transform

if TYPE_CHECKING:
    from torch import Tensor
    from torch_shapes import Dim


# ============================================================================
# Distribution classes (from torchbenchmark/util/distribution.py)
# ============================================================================


class TanhTransform(Transform):
    """Bijective tanh transform with numerically stable log-det-jacobian.

    Original: torchbenchmark/util/distribution.py TanhTransform class.
    """

    domain: Any = torch.distributions.constraints.real
    codomain: Any = torch.distributions.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __eq__(self, other: object) -> bool:
        return isinstance(other, TanhTransform)

    def _call[*S](self, x: Tensor[*S]) -> Tensor[*S]:
        return x.tanh()

    def _inverse[*S](self, y: Tensor[*S]) -> Tensor[*S]:
        return y.clamp(-0.99999997, 0.99999997).atanh()

    def log_abs_det_jacobian[*S](self, x: Tensor[*S], y: Tensor[*S]) -> Tensor[*S]:
        # Numerically stable: log(1 - tanh(x)^2) = 2*(log(2) - x - softplus(-2x))
        return 2.0 * (math.log(2.0) - x - F.softplus(-2.0 * x))


class SquashedNormal[*EventShape](TransformedDistribution):
    """Normal distribution followed by tanh squashing.

    Original: torchbenchmark/util/distribution.py SquashedNormal class.

    Wraps Normal(loc, scale) with a TanhTransform to produce samples in (-1, 1).
    Shape-preserving: rsample/log_prob return same shape as loc/scale.
    """

    def __init__(self, loc: Tensor[*EventShape], scale: Tensor[*EventShape]) -> None:
        self.loc = loc
        self.scale = scale
        base_dist = Normal(loc, scale)
        tfms: list[Transform] = [TanhTransform()]
        super().__init__(base_dist, tfms)

    @property
    def mean(self) -> Tensor[*EventShape]:
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu


# ============================================================================
# Encoder
# ============================================================================


class Encoder[C, FeatDim](nn.Module):
    """CNN encoder for 84x84 pixel observations.

    Architecture:
    - Conv2d(C, 32, 3, stride=2): 84 → 41
    - Conv2d(32, 32, 3, stride=1): 41 → 39
    - Conv2d(32, 32, 3, stride=1): 39 → 37
    - Conv2d(32, 32, 3, stride=1): 37 → 35
    - Flatten: 32 * 35 * 35 = 39200
    - Linear(39200, FeatDim) → LayerNorm → tanh

    (B, C, 84, 84) → (B, FeatDim)
    """

    def __init__(self, channels: Dim[C], feature_dim: Dim[FeatDim]) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        self.fc = nn.Linear(39200, feature_dim)
        self.ln = nn.LayerNorm(feature_dim)
        self.flatten = nn.Flatten()

    def copy_conv_weights_from(self, source: "Encoder[C, FeatDim]") -> None:
        """Copy conv layer weights from another encoder (for weight sharing).

        Original: drq.py Encoder.copy_conv_weights_from.
        Used to share conv weights between actor and critic encoders.
        """
        self.conv1.weight = source.conv1.weight
        self.conv2.weight = source.conv2.weight
        self.conv3.weight = source.conv3.weight
        self.conv4.weight = source.conv4.weight

    def log(self, prefix: str) -> dict[str, Tensor]:
        """Log encoder conv layer weights for monitoring.

        Original: drq.py Encoder.log.
        Returns dict of weight norms for logging.
        """
        return {
            f"{prefix}_conv1_w": self.conv1.weight,
            f"{prefix}_conv2_w": self.conv2.weight,
            f"{prefix}_conv3_w": self.conv3.weight,
            f"{prefix}_conv4_w": self.conv4.weight,
        }

    def forward[B](self, obs: Tensor[B, C, 84, 84]) -> Tensor[B, FeatDim]:
        obs = obs / 255.0
        h1 = torch.relu(self.conv1(obs))
        assert_type(h1, Tensor[B, 32, 41, 41])
        h2 = torch.relu(self.conv2(h1))
        assert_type(h2, Tensor[B, 32, 39, 39])
        h3 = torch.relu(self.conv3(h2))
        assert_type(h3, Tensor[B, 32, 37, 37])
        h4 = torch.relu(self.conv4(h3))
        assert_type(h4, Tensor[B, 32, 35, 35])
        h4_flat = self.flatten(h4)
        assert_type(h4_flat, Tensor[B, 39200])
        h5 = self.fc(h4_flat)
        assert_type(h5, Tensor[B, FeatDim])
        h6 = self.ln(h5)
        assert_type(h6, Tensor[B, FeatDim])
        return torch.tanh(h6)


# ============================================================================
# Actor
# ============================================================================


class Actor[C, FeatDim, ActDim, H](nn.Module):
    """MLP policy head on top of CNN encoder.

    Architecture:
        Encoder(C, FeatDim) → mlp(FeatDim, H, 2*ActDim) →
        chunk → (mu, log_std) → constrain log_std → (mu, std)

    mlp() = nn.Sequential(Linear, ReLU, Linear, ReLU, Linear)

    (B, C, 84, 84) → (Tensor[B, ActDim], Tensor[B, ActDim])
    """

    def __init__(
        self,
        channels: Dim[C],
        feature_dim: Dim[FeatDim],
        action_dim: Dim[ActDim],
        hidden_dim: Dim[H],
        log_std_bounds: tuple[float, float] = (-10.0, 2.0),
    ) -> None:
        super().__init__()
        self.encoder = Encoder(channels, feature_dim)
        self.log_std_bounds = log_std_bounds
        # mlp(feature_dim, hidden_dim, 2*action_dim)
        self.trunk = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * action_dim),
        )

    def forward[B](
        self, obs: Tensor[B, C, 84, 84], detach_encoder: bool = False
    ) -> SquashedNormal[B, ActDim]:
        feat = self.encoder(obs)
        assert_type(feat, Tensor[B, FeatDim])
        if detach_encoder:
            feat = feat.detach()
        out = self.trunk(feat)
        assert_type(out, Tensor[B, 2 * ActDim])
        mu, log_std = out.chunk(2, dim=-1)
        assert_type(mu, Tensor[B, ActDim])
        assert_type(log_std, Tensor[B, ActDim])
        # Constrain log_std to [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)
        std = log_std.exp()
        assert_type(std, Tensor[B, ActDim])
        return SquashedNormal(mu, std)


# ============================================================================
# Critic
# ============================================================================


class Critic[C, FeatDim, ActDim, H](nn.Module):
    """Double Q-network critic on top of CNN encoder.

    Architecture:
        Encoder(C, FeatDim) → cat(features, action) →
        Two parallel mlp(FeatDim+ActDim, H, 1) networks

    mlp() = nn.Sequential(Linear, ReLU, Linear, ReLU, Linear)

    (B, C, 84, 84), (B, ActDim) → (Tensor[B, 1], Tensor[B, 1])
    """

    def __init__(
        self,
        channels: Dim[C],
        feature_dim: Dim[FeatDim],
        action_dim: Dim[ActDim],
        hidden_dim: Dim[H],
    ) -> None:
        super().__init__()
        self.encoder = Encoder(channels, feature_dim)
        # Q1: mlp(feature_dim + action_dim, hidden_dim, 1)
        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Q2: mlp(feature_dim + action_dim, hidden_dim, 1)
        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward[B](
        self,
        obs: Tensor[B, C, 84, 84],
        action: Tensor[B, ActDim],
        detach_encoder: bool = False,
    ) -> tuple[Tensor[B, 1], Tensor[B, 1]]:
        feat = self.encoder(obs)
        assert_type(feat, Tensor[B, FeatDim])
        if detach_encoder:
            feat = feat.detach()
        obs_action = torch.cat((feat, action), dim=1)
        assert_type(obs_action, Tensor[B, FeatDim + ActDim])
        q1 = self.Q1(obs_action)
        assert_type(q1, Tensor[B, 1])
        q2 = self.Q2(obs_action)
        assert_type(q2, Tensor[B, 1])
        return q1, q2


# ============================================================================
# DRQ Agent
# ============================================================================


def soft_update_params(net: nn.Module, target_net: nn.Module, tau: float) -> None:
    """Polyak averaging: target = tau * net + (1 - tau) * target.

    Original: drq/drqutils.py soft_update_params.
    """
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


class DRQAgent[C, FeatDim, ActDim, H](nn.Module):
    """DRQ agent wrapping actor and critic with shared encoder weights.

    Original: drq.py DrQAgent class.
    The actor and critic each have their own encoder, but the critic's
    conv weights are copied from the actor's encoder after each update.

    Includes full training methods: update_critic, update_actor_and_alpha,
    and the orchestrating update method. Uses SAC-style entropy-regularized
    RL with learned temperature (alpha).
    """

    def __init__(
        self,
        channels: Dim[C],
        feature_dim: Dim[FeatDim],
        action_dim: Dim[ActDim],
        hidden_dim: Dim[H],
        discount: float = 0.99,
        init_temperature: float = 0.1,
        lr: float = 1e-3,
        actor_update_frequency: int = 2,
        critic_tau: float = 0.01,
        critic_target_update_frequency: int = 2,
    ) -> None:
        super().__init__()
        self.discount = discount
        self.critic_tau = critic_tau
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency

        self.actor = Actor(channels, feature_dim, action_dim, hidden_dim)
        self.critic = Critic(channels, feature_dim, action_dim, hidden_dim)
        self.critic_target = Critic(channels, feature_dim, action_dim, hidden_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        # Copy conv weights from actor encoder to critic encoder
        self.critic.encoder.copy_conv_weights_from(self.actor.encoder)

        # Learned temperature (alpha) for entropy regularization
        self.log_alpha = torch.tensor(math.log(init_temperature))
        self.log_alpha.requires_grad = True
        # Target entropy = -|A| (standard SAC heuristic)
        self.target_entropy: float = -float(action_dim)

        # Optimizers (not nn.Module attributes — plain optimizer objects)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=1e-4)

    @property
    def alpha(self) -> Tensor:
        return self.log_alpha.exp()

    def act[B](self, obs: Tensor[B, C, 84, 84]) -> SquashedNormal[B, ActDim]:
        """Select action: returns SquashedNormal distribution from actor."""
        return self.actor(obs)

    def criticize[B](
        self, obs: Tensor[B, C, 84, 84], action: Tensor[B, ActDim]
    ) -> tuple[Tensor[B, 1], Tensor[B, 1]]:
        """Evaluate action: returns (q1, q2) from critic."""
        return self.critic(obs, action)

    def update_critic(
        self,
        obs: Tensor,
        action: Tensor,
        reward: Tensor,
        next_obs: Tensor,
        not_done: Tensor,
        obs_aug: Tensor,
        next_obs_aug: Tensor,
    ) -> Tensor:
        """Update critic networks using clipped double-Q learning.

        Original: drq.py DrQAgent.update_critic.
        Uses both original and augmented observations for target Q computation,
        averaging the two target values (DRQ's data augmentation trick).
        """
        with torch.no_grad():
            dist = self.actor(next_obs)
            next_action: Tensor = dist.rsample()
            log_prob: Tensor = dist.log_prob(next_action).sum(-1, keepdim=True)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V: Tensor = (
                torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob
            )
            target_Q: Tensor = reward + not_done * self.discount * target_V

            dist_aug = self.actor(next_obs_aug)
            next_action_aug: Tensor = dist_aug.rsample()
            log_prob_aug: Tensor = dist_aug.log_prob(next_action_aug).sum(
                -1, keepdim=True
            )
            target_Q1_aug, target_Q2_aug = self.critic_target(
                next_obs_aug, next_action_aug
            )
            target_V_aug: Tensor = (
                torch.min(target_Q1_aug, target_Q2_aug)
                - self.alpha.detach() * log_prob_aug
            )
            target_Q_aug: Tensor = reward + not_done * self.discount * target_V_aug

            target_Q = (target_Q + target_Q_aug) / 2

        # Current Q estimates on original obs
        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss: Tensor = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q
        )

        # Current Q estimates on augmented obs
        current_Q1_aug, current_Q2_aug = self.critic(
            obs_aug, action, detach_encoder=True
        )
        critic_loss = (
            critic_loss
            + F.mse_loss(current_Q1_aug, target_Q)
            + F.mse_loss(current_Q2_aug, target_Q)
        )

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        return critic_loss

    def update_actor_and_alpha(self, obs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Update actor and entropy temperature alpha.

        Original: drq.py DrQAgent.update_actor_and_alpha.
        Actor loss: (alpha * log_prob - min(Q1, Q2)).mean()
        Alpha loss: -(log_alpha * (log_prob + target_entropy)).mean()
        """
        dist = self.actor(obs, detach_encoder=True)
        action: Tensor = dist.rsample()
        log_prob: Tensor = dist.log_prob(action).sum(-1, keepdim=True)

        actor_Q1, actor_Q2 = self.critic(obs, action, detach_encoder=True)
        actor_Q: Tensor = torch.min(actor_Q1, actor_Q2)
        actor_loss: Tensor = (self.alpha.detach() * log_prob - actor_Q).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update alpha
        alpha_loss: Tensor = (
            self.log_alpha * (-log_prob - self.target_entropy).detach()
        ).mean()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        return actor_loss, alpha_loss, self.alpha

    def update(
        self,
        obs: Tensor,
        action: Tensor,
        reward: Tensor,
        next_obs: Tensor,
        not_done: Tensor,
        obs_aug: Tensor,
        next_obs_aug: Tensor,
        step: int,
    ) -> dict[str, Tensor]:
        """Full update step: critic, then optionally actor + alpha + target.

        Original: drq.py DrQAgent.update.
        - Critic updated every step
        - Actor updated every actor_update_frequency steps
        - Critic target soft-updated every critic_target_update_frequency steps
        """
        metrics: dict[str, Tensor] = {}
        critic_loss = self.update_critic(
            obs, action, reward, next_obs, not_done, obs_aug, next_obs_aug
        )
        metrics["critic_loss"] = critic_loss

        if step % self.actor_update_frequency == 0:
            actor_loss, alpha_loss, alpha = self.update_actor_and_alpha(obs)
            metrics["actor_loss"] = actor_loss
            metrics["alpha_loss"] = alpha_loss
            metrics["alpha"] = alpha

        if step % self.critic_target_update_frequency == 0:
            soft_update_params(self.critic.Q1, self.critic_target.Q1, self.critic_tau)
            soft_update_params(self.critic.Q2, self.critic_target.Q2, self.critic_tau)
            soft_update_params(
                self.critic.encoder, self.critic_target.encoder, self.critic_tau
            )

        return metrics


# ============================================================================
# Smoke tests
# ============================================================================


def test_encoder():
    """Test CNN encoder: (B, 9, 84, 84) → (B, 50)."""
    enc = Encoder(9, 50)
    obs: Tensor[4, 9, 84, 84] = torch.randn(4, 9, 84, 84)
    feat = enc(obs)
    assert_type(feat, Tensor[4, 50])


def test_encoder_rgb():
    """Test CNN encoder with 3-channel (RGB) input."""
    enc = Encoder(3, 128)
    obs: Tensor[8, 3, 84, 84] = torch.randn(8, 3, 84, 84)
    feat = enc(obs)
    assert_type(feat, Tensor[8, 128])


def test_actor():
    """Test actor: (B, 9, 84, 84) → SquashedNormal distribution."""
    actor = Actor(9, 50, 6, 1024)
    obs: Tensor[4, 9, 84, 84] = torch.randn(4, 9, 84, 84)
    dist = actor(obs)
    assert_type(dist, SquashedNormal[4, 6])


def test_critic():
    """Test critic: (B, 9, 84, 84), (B, 6) → (q1, q2) each (B, 1)."""
    critic = Critic(9, 50, 6, 1024)
    obs: Tensor[4, 9, 84, 84] = torch.randn(4, 9, 84, 84)
    action: Tensor[4, 6] = torch.randn(4, 6)
    q1, q2 = critic(obs, action)
    assert_type(q1, Tensor[4, 1])
    assert_type(q2, Tensor[4, 1])


def test_critic_different_dims():
    """Test critic with different channel/action dimensions."""
    critic = Critic(3, 128, 4, 512)
    obs: Tensor[8, 3, 84, 84] = torch.randn(8, 3, 84, 84)
    action: Tensor[8, 4] = torch.randn(8, 4)
    q1, q2 = critic(obs, action)
    assert_type(q1, Tensor[8, 1])
    assert_type(q2, Tensor[8, 1])


def test_drq_agent():
    """Test DRQ agent: act and criticize."""
    agent = DRQAgent(9, 50, 6, 1024)
    obs: Tensor[4, 9, 84, 84] = torch.randn(4, 9, 84, 84)
    dist = agent.act(obs)
    assert_type(dist, SquashedNormal[4, 6])
    action: Tensor[4, 6] = torch.randn(4, 6)
    q1, q2 = agent.criticize(obs, action)
    assert_type(q1, Tensor[4, 1])
    assert_type(q2, Tensor[4, 1])
