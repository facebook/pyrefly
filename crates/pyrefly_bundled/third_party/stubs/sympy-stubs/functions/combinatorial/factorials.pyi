from typing import Any

from sympy.core.function import Function, UndefinedFunction
from sympy.core.numbers import Integer
from sympy.series.order import Order

class CombinatorialFunction(Function): ...

class factorial(CombinatorialFunction):
    def fdiff(self, argindex=...): ...

    _small_swing = ...
    _small_factorials: list[int] = ...
    @classmethod
    def eval(cls, n) -> Integer | None: ...

class MultiFactorial(CombinatorialFunction): ...

class subfactorial(CombinatorialFunction):
    @classmethod
    def eval(cls, arg) -> int | None: ...

class factorial2(CombinatorialFunction):
    @classmethod
    def eval(cls, arg) -> None: ...

class RisingFactorial(CombinatorialFunction):
    @classmethod
    def eval(cls, x, k) -> type[UndefinedFunction] | int | float | None: ...

class FallingFactorial(CombinatorialFunction):
    @classmethod
    def eval(cls, x, k) -> type[UndefinedFunction] | int | float | None: ...

rf = RisingFactorial
ff = FallingFactorial

class binomial(CombinatorialFunction):
    def fdiff(self, argindex=...): ...
    @classmethod
    def eval(cls, n, k) -> Order | Any | Integer | float | None: ...
