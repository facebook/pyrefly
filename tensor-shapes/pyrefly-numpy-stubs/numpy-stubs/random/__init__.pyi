# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import overload

# Preserve NumPy's canonical re-exports before local shape-aware declarations.
from numpy.random._generator import (
    default_rng as default_rng,
    Generator as Generator,
)
from numpy.random._mt19937 import (
    MT19937 as MT19937,
)
from numpy.random._pcg64 import (
    PCG64 as PCG64,
    PCG64DXSM as PCG64DXSM,
)
from numpy.random._philox import (
    Philox as Philox,
)
from numpy.random._sfc64 import (
    SFC64 as SFC64,
)
from numpy.random.bit_generator import (
    BitGenerator as BitGenerator,
    SeedSequence as SeedSequence,
)
from numpy.random.mtrand import (
    beta as beta,
    binomial as binomial,
    bytes as bytes,
    chisquare as chisquare,
    choice as choice,
    dirichlet as dirichlet,
    exponential as exponential,
    f as f,
    gamma as gamma,
    geometric as geometric,
    get_state as get_state,
    gumbel as gumbel,
    hypergeometric as hypergeometric,
    laplace as laplace,
    logistic as logistic,
    lognormal as lognormal,
    logseries as logseries,
    multinomial as multinomial,
    multivariate_normal as multivariate_normal,
    negative_binomial as negative_binomial,
    noncentral_chisquare as noncentral_chisquare,
    noncentral_f as noncentral_f,
    normal as normal,
    pareto as pareto,
    permutation as permutation,
    poisson as poisson,
    power as power,
    rand as rand,
    randint as randint,
    random as random,
    random_integers as random_integers,
    random_sample as random_sample,
    RandomState as RandomState,
    ranf as ranf,
    rayleigh as rayleigh,
    sample as sample,
    seed as seed,
    set_state as set_state,
    shuffle as shuffle,
    standard_cauchy as standard_cauchy,
    standard_exponential as standard_exponential,
    standard_gamma as standard_gamma,
    standard_normal as standard_normal,
    standard_t as standard_t,
    triangular as triangular,
    uniform as uniform,
    vonmises as vonmises,
    wald as wald,
    weibull as weibull,
    zipf as zipf,
)
from shape_extensions import Int, IntVar

from .. import dtype, float64, ndarray

@overload
def randn[N: IntVar](d0: Int[N], /) -> ndarray[[N], dtype[float64]]: ...
@overload
def randn[N: IntVar, M: IntVar](
    d0: Int[N], d1: Int[M], /
) -> ndarray[[N, M], dtype[float64]]: ...
