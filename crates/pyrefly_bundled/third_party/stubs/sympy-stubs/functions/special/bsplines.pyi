from functools import lru_cache
from typing import Any

from sympy.functions.elementary.piecewise import Piecewise
from sympy.series.order import Order

@lru_cache(maxsize=128)
def bspline_basis(d, knots, n, x) -> Piecewise | Order | Any: ...
def bspline_basis_set(d, knots, x) -> list[Any | Piecewise | Order]: ...
def interpolating_spline(d, x, X, Y) -> Piecewise: ...
