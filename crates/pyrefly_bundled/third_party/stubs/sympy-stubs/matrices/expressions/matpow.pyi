from typing import Any
from typing_extensions import Self

from sympy.core.basic import Basic
from sympy.matrices.expressions.inverse import Inverse
from sympy.matrices.expressions.matexpr import MatrixExpr
from sympy.matrices.expressions.special import Identity

class MatPow(MatrixExpr):
    def __new__(cls, base, exp, evaluate=..., **options) -> Basic | Identity | Inverse | Any | MatPow | Self: ...
    @property
    def base(self) -> Basic: ...
    @property
    def exp(self) -> Basic: ...
    @property
    def shape(self): ...
    def doit(self, **hints) -> Basic | Identity | Inverse | Any | MatPow: ...
