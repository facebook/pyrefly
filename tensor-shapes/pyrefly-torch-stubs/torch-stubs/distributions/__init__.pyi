# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Type stubs for torch.distributions.

Distribution classes track event shape via an `IntTuple`-bound `EventShape`.
rsample() and log_prob() preserve the event shape of the distribution.

Submodules re-exported to support original import patterns:
  pyd.transforms.Transform, pyd.constraints.real, etc.
"""

from typing import Any

from shape_extensions import IntTuple
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
from .bernoulli import Bernoulli as Bernoulli
from .binomial import Binomial as Binomial
from .cauchy import Cauchy as Cauchy
from .chi2 import Chi2 as Chi2
from .constraint_registry import biject_to as biject_to, transform_to as transform_to
from .continuous_bernoulli import ContinuousBernoulli as ContinuousBernoulli
from .dirichlet import Dirichlet as Dirichlet
from .exp_family import ExponentialFamily as ExponentialFamily
from .exponential import Exponential as Exponential
from .fishersnedecor import FisherSnedecor as FisherSnedecor
from .gamma import Gamma as Gamma
from .generalized_pareto import GeneralizedPareto as GeneralizedPareto
from .geometric import Geometric as Geometric
from .gumbel import Gumbel as Gumbel
from .half_cauchy import HalfCauchy as HalfCauchy
from .half_normal import HalfNormal as HalfNormal
from .independent import Independent as Independent
from .inverse_gamma import InverseGamma as InverseGamma
from .kl import kl_divergence as kl_divergence, register_kl as register_kl
from .kumaraswamy import Kumaraswamy as Kumaraswamy
from .laplace import Laplace as Laplace
from .lkj_cholesky import LKJCholesky as LKJCholesky
from .log_normal import LogNormal as LogNormal
from .logistic_normal import LogisticNormal as LogisticNormal
from .lowrank_multivariate_normal import (
    LowRankMultivariateNormal as LowRankMultivariateNormal,
)
from .mixture_same_family import MixtureSameFamily as MixtureSameFamily
from .multinomial import Multinomial as Multinomial
from .multivariate_normal import MultivariateNormal as MultivariateNormal
from .negative_binomial import NegativeBinomial as NegativeBinomial
from .one_hot_categorical import (
    OneHotCategorical as OneHotCategorical,
    OneHotCategoricalStraightThrough as OneHotCategoricalStraightThrough,
)
from .pareto import Pareto as Pareto
from .poisson import Poisson as Poisson
from .relaxed_bernoulli import RelaxedBernoulli as RelaxedBernoulli
from .relaxed_categorical import RelaxedOneHotCategorical as RelaxedOneHotCategorical
from .studentT import StudentT as StudentT
from .transforms import (
    AbsTransform as AbsTransform,
    AffineTransform as AffineTransform,
    CatTransform as CatTransform,
    ComposeTransform as ComposeTransform,
    CorrCholeskyTransform as CorrCholeskyTransform,
    CumulativeDistributionTransform as CumulativeDistributionTransform,
    ExpTransform as ExpTransform,
    identity_transform as identity_transform,
    IndependentTransform as IndependentTransform,
    LowerCholeskyTransform as LowerCholeskyTransform,
    PositiveDefiniteTransform as PositiveDefiniteTransform,
    PowerTransform as PowerTransform,
    ReshapeTransform as ReshapeTransform,
    SigmoidTransform as SigmoidTransform,
    SoftmaxTransform as SoftmaxTransform,
    SoftplusTransform as SoftplusTransform,
    StackTransform as StackTransform,
    StickBreakingTransform as StickBreakingTransform,
    TanhTransform as TanhTransform,
    Transform as Transform,
)
from .uniform import Uniform as Uniform
from .von_mises import VonMises as VonMises
from .weibull import Weibull as Weibull
from .wishart import Wishart as Wishart

class Distribution[EventShape: IntTuple]:
    """Base class for probability distributions."""
    def sample(self, sample_shape: Any = ...) -> Tensor[EventShape]: ...
    def rsample(self, sample_shape: Any = ...) -> Tensor[EventShape]: ...
    def log_prob(self, value: Tensor) -> Tensor[EventShape]: ...
    @property
    def mean(self) -> Tensor[EventShape]: ...
    @property
    def variance(self) -> Tensor[EventShape]: ...

class Normal[EventShape: IntTuple](Distribution[EventShape]):
    """Normal (Gaussian) distribution."""

    loc: Tensor[EventShape]
    scale: Tensor[EventShape]
    def __init__(self, loc: Tensor[EventShape], scale: Tensor[EventShape]) -> None: ...

class Categorical(Distribution):
    """Categorical distribution."""
    def __init__(
        self, probs: Tensor | None = None, logits: Tensor | None = None
    ) -> None: ...

class Beta(Distribution):
    """Beta distribution."""

    mean: Tensor
    def __init__(self, concentration1: Tensor, concentration0: Tensor) -> None: ...

class TransformedDistribution[EventShape: IntTuple](Distribution[EventShape]):
    """Distribution transformed by a sequence of transforms."""

    transforms: list[transforms.Transform]
    def __init__(
        self,
        base_distribution: Distribution[EventShape],
        transforms: list[transforms.Transform],
    ) -> None: ...
