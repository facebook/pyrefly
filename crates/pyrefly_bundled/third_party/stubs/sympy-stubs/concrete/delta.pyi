from typing import Any

from sympy.concrete.summations import Sum
from sympy.core.basic import Basic
from sympy.core.cache import cacheit
from sympy.core.relational import Equality, Ne, Relational
from sympy.functions.elementary.piecewise import Piecewise

@cacheit
def deltaproduct(f, limit) -> Equality | Relational | Ne | Any: ...
@cacheit
def deltasummation(f, limit, no_piecewise=...) -> Equality | Relational | Ne | Basic | Piecewise | Sum: ...
