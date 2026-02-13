from typing import Any
from typing_extensions import Self

from sympy.core import Expr
from sympy.matrices.expressions.slice import MatrixSlice

class DotProduct(Expr):
    def __new__(cls, arg1, arg2) -> Self: ...
    def doit(self, expand=..., **hints) -> Any | MatrixSlice: ...
