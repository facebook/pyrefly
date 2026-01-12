from typing import Any
from typing_extensions import Self

from sympy.matrices.expressions.matexpr import MatrixExpr

class CompanionMatrix(MatrixExpr):
    def __new__(cls, poly) -> Self: ...
    @property
    def shape(self) -> tuple[Any, Any]: ...
    def as_explicit(self): ...
