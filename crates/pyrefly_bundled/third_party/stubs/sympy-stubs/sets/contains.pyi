from typing import Any

from sympy.core.basic import Basic
from sympy.logic.boolalg import BooleanFunction
from sympy.sets.sets import Set

class Contains(BooleanFunction):
    @classmethod
    def eval(cls, x, s) -> Set | None: ...
    @property
    def binary_symbols(self) -> set[Any | Basic]: ...
    def as_set(self) -> Basic: ...
