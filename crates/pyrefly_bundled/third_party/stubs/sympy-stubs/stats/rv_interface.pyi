from typing import Any

from sympy import Equality, Integral, Piecewise
from sympy.concrete.summations import Sum
from sympy.core.basic import Basic
from sympy.core.function import UndefinedFunction
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.relational import Ne, Relational
from sympy.series.order import Order
from sympy.sets.sets import FiniteSet, Set
from sympy.stats.rv import (
    PSpace,
    cdf,
    characteristic_function,
    density,
    dependent,
    expectation,
    given,
    independent,
    is_random,
    moment_generating_function,
    probability,
    pspace,
    quantile,
    random_symbols,
    sample,
    sample_iter,
    sample_stochastic_process,
    sampling_density,
    where,
)
from sympy.stats.symbolic_multivariate_probability import CrossCovarianceMatrix, ExpectationMatrix, VarianceMatrix
from sympy.stats.symbolic_probability import CentralMoment, Covariance, Expectation, Moment, Variance

__all__ = [
    "P",
    "E",
    "H",
    "density",
    "where",
    "given",
    "sample",
    "cdf",
    "characteristic_function",
    "pspace",
    "sample_iter",
    "variance",
    "std",
    "skewness",
    "kurtosis",
    "covariance",
    "dependent",
    "entropy",
    "median",
    "independent",
    "random_symbols",
    "correlation",
    "factorial_moment",
    "moment",
    "cmoment",
    "sampling_density",
    "moment_generating_function",
    "smoment",
    "quantile",
    "sample_stochastic_process",
]

def moment(X, n, c=..., condition=..., *, evaluate=..., **kwargs) -> Any | Moment: ...
def variance(X, condition=..., **kwargs) -> VarianceMatrix | Variance | Any | CentralMoment: ...
def standard_deviation(X, condition=..., **kwargs) -> Pow: ...

std = ...

def entropy(
    expr, condition=..., **kwargs
) -> (
    int
    | Mul
    | Basic
    | Expectation
    | tuple[Any, ...]
    | Sum
    | Order
    | Any
    | Piecewise
    | Equality
    | Relational
    | Ne
    | Integral
    | ExpectationMatrix
    | None
): ...
def covariance(
    X, Y, condition=..., **kwargs
) -> (
    CrossCovarianceMatrix
    | Covariance
    | Basic
    | Expectation
    | tuple[Any, ...]
    | Sum
    | Order
    | Any
    | Piecewise
    | Equality
    | Relational
    | Ne
    | Integral
    | ExpectationMatrix
    | None
): ...
def correlation(X, Y, condition=..., **kwargs): ...
def cmoment(X, n, condition=..., *, evaluate=..., **kwargs) -> Any | CentralMoment: ...
def smoment(X, n, condition=..., **kwargs): ...
def skewness(X, condition=..., **kwargs): ...
def kurtosis(X, condition=..., **kwargs): ...
def factorial_moment(
    X, n, condition=..., **kwargs
) -> (
    type[UndefinedFunction]
    | Basic
    | Expectation
    | tuple[Any, ...]
    | Sum
    | Order
    | Any
    | Piecewise
    | Equality
    | Relational
    | Ne
    | Integral
    | ExpectationMatrix
    | None
): ...
def median(X, evaluate=..., **kwargs) -> FiniteSet | Set: ...
def coskewness(X, Y, Z, condition=..., **kwargs): ...

P = ...
E = ...
H = ...
