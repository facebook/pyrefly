from typing import Literal

from sympy.assumptions.predicates.calculus import (
    FinitePredicate as FinitePredicate,
    InfinitePredicate as InfinitePredicate,
    NegativeInfinitePredicate as NegativeInfinitePredicate,
    PositiveInfinitePredicate as PositiveInfinitePredicate,
)
from sympy.core import Add, Mul, Pow, Symbol
from sympy.core.numbers import (
    ComplexInfinity,
    Exp1,
    GoldenRatio,
    ImaginaryUnit,
    Infinity,
    NaN,
    NegativeInfinity,
    Number,
    Pi,
    TribonacciConstant,
)
from sympy.functions import cos, exp, log, sign, sin

@FinitePredicate.register(Symbol)
def _(expr, assumptions) -> Literal[True] | None: ...
@FinitePredicate.register(Add)
def _(expr, assumptions) -> bool | None: ...
@FinitePredicate.register(Mul)
def _(expr, assumptions) -> bool | None: ...
@FinitePredicate.register(Pow)
def _(expr, assumptions) -> bool | None: ...
@FinitePredicate.register(exp)
def _(expr, assumptions) -> bool | None: ...
@FinitePredicate.register(log)
def _(expr, assumptions) -> bool | None: ...
@FinitePredicate.register_many(cos, sin, Number, Pi, Exp1, GoldenRatio, TribonacciConstant, ImaginaryUnit, sign)
def _(expr, assumptions) -> Literal[True]: ...
@FinitePredicate.register_many(ComplexInfinity, Infinity, NegativeInfinity)
def _(expr, assumptions) -> Literal[False]: ...
@FinitePredicate.register(NaN)
def _(expr, assumptions) -> None: ...
@InfinitePredicate.register_many(ComplexInfinity, Infinity, NegativeInfinity)
def _(expr, assumptions) -> Literal[True]: ...
@PositiveInfinitePredicate.register(Infinity)
def _(expr, assumptions) -> Literal[True]: ...
@PositiveInfinitePredicate.register_many(NegativeInfinity, ComplexInfinity)
def _(expr, assumptions) -> Literal[False]: ...
@NegativeInfinitePredicate.register(NegativeInfinity)
def _(expr, assumptions) -> Literal[True]: ...
@NegativeInfinitePredicate.register_many(Infinity, ComplexInfinity)
def _(expr, assumptions) -> Literal[False]: ...
