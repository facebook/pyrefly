from typing import Any

from sympy.core.expr import Expr
from sympy.core.function import Function
from sympy.core.power import Pow
from sympy.functions.elementary.miscellaneous import Max, Min

def decompogen(f, symbol) -> list[Expr] | list[Function | Pow] | list[Min | Max] | Any: ...
def compogen(g_s, symbol): ...
