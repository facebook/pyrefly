from typing import Any

from sympy.core.function import UndefinedFunction
from sympy.core.relational import Equality, Ne, Relational

def singularityintegrate(f, x) -> type[UndefinedFunction] | Equality | Any | Relational | Ne | None: ...
