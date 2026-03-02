from typing import Any, Literal
from typing_extensions import Self

from sympy import Basic, Equality, Integral, Ne, Piecewise, Sum
from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.function import Lambda
from sympy.core.relational import Relational
from sympy.series.order import Order
from sympy.stats.frv_types import BernoulliDistribution
from sympy.stats.rv import RandomSymbol, is_random
from sympy.stats.symbolic_multivariate_probability import CrossCovarianceMatrix, ExpectationMatrix, VarianceMatrix

__all__ = ["Probability", "Expectation", "Variance", "Covariance"]

@is_random.register(Expr)  # type: ignore[has-type]
def _(x) -> bool: ...
@is_random.register(RandomSymbol)  # type: ignore[has-type]
def _(x) -> Literal[True]: ...

class Probability(Expr):
    def __new__(cls, prob, condition=..., **kwargs) -> Self: ...
    def doit(self, **hints) -> Any | BernoulliDistribution | Probability | Equality | Lambda | Order | Relational | Ne | int: ...

    _eval_rewrite_as_Sum = ...
    def evaluate_integral(
        self,
    ) -> Any | BernoulliDistribution | Probability | Equality | Lambda | Order | Relational | Ne | int: ...

class Expectation(Expr):
    def __new__(cls, expr, condition=..., **kwargs) -> ExpectationMatrix | Self: ...
    def expand(self, **hints) -> Basic | Add | Self: ...
    def doit(
        self, **hints
    ) -> (
        Basic
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
        | Self
        | None
    ): ...

    _eval_rewrite_as_Sum = ...
    def evaluate_integral(
        self,
    ) -> (
        Basic
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
        | Self
        | None
    ): ...

    evaluate_sum = ...

class Variance(Expr):
    def __new__(cls, arg, condition=..., **kwargs) -> VarianceMatrix | Self: ...
    def expand(self, **hints) -> Self: ...

    _eval_rewrite_as_Sum = ...
    def evaluate_integral(self) -> Self | Any: ...

class Covariance(Expr):
    def __new__(cls, arg1, arg2, condition=..., **kwargs) -> CrossCovarianceMatrix | Self: ...
    def expand(self, **hints) -> CrossCovarianceMatrix | Covariance | Add: ...

    _eval_rewrite_as_Sum = ...
    def evaluate_integral(self) -> Self | Any: ...

class Moment(Expr):
    def __new__(cls, X, n, c=..., condition=..., **kwargs) -> Self: ...
    def doit(self, **hints) -> Any: ...

class CentralMoment(Expr):
    def __new__(cls, X, n, condition=..., **kwargs) -> Self: ...
    def doit(self, **hints) -> Any: ...
