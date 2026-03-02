from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.mul import Mul
from sympy.series.order import Order
from sympy.utilities import public

@public
def together(expr, deep=..., fraction=...) -> Basic | Add | Order | Mul | dict: ...
