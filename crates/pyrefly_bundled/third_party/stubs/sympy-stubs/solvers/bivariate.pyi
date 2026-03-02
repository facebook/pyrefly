from typing import Any

from sympy.core.symbol import Dummy
from sympy.series.order import Order

def bivariate_type(
    f, x, y, *, first=...
) -> tuple[Any, Any, Any] | tuple[Any, Any | Order, Dummy] | tuple[Any, Any, Dummy] | None: ...
