# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Type stubs for torch.distributions.

Covers distribution classes used as return types in model ports (SAC).
These are not tensor shape operations — they exist so that model ports
returning distributions can type-check without errors.

Submodules re-exported to support original import patterns:
  pyd.transforms.Transform, pyd.constraints.real, etc.
"""

from typing import Any

from torch import Tensor

# Re-export submodules for pyd.transforms.X, pyd.constraints.X,
# pyd.beta.Beta, pyd.categorical.Categorical, etc. access patterns
from . import (
    beta as beta,
    categorical as categorical,
    constraints as constraints,
    transformed_distribution as transformed_distribution,
    transforms as transforms,
)

class Distribution:
    """Base class for probability distributions."""
    def sample(self, sample_shape: Any = ...) -> Tensor: ...
    def log_prob(self, value: Tensor) -> Tensor: ...
    @property
    def mean(self) -> Tensor: ...
    @property
    def variance(self) -> Tensor: ...

class Normal(Distribution):
    """Normal (Gaussian) distribution."""

    loc: Tensor
    scale: Tensor
    def __init__(self, loc: Tensor, scale: Tensor) -> None: ...

class Categorical(Distribution):
    """Categorical distribution."""
    def __init__(
        self, probs: Tensor | None = None, logits: Tensor | None = None
    ) -> None: ...

class Beta(Distribution):
    """Beta distribution."""

    mean: Tensor
    def __init__(self, concentration1: Tensor, concentration0: Tensor) -> None: ...

class TransformedDistribution(Distribution):
    """Distribution transformed by a sequence of transforms."""

    transforms: list[transforms.Transform]
    def __init__(
        self, base_distribution: Distribution, transforms: list[transforms.Transform]
    ) -> None: ...
