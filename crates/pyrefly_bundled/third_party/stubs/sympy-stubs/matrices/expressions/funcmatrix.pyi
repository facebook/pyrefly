from typing_extensions import Self

from sympy.core.basic import Basic
from sympy.matrices.expressions.matexpr import MatrixExpr

class FunctionMatrix(MatrixExpr):
    def __new__(cls, rows, cols, lamda) -> Self: ...
    @property
    def shape(self) -> tuple[Basic, ...]: ...
    @property
    def lamda(self) -> Basic: ...
