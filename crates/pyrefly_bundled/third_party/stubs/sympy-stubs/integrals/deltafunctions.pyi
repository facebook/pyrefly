from typing import Any

from sympy.core.relational import Equality, Ne, Relational
from sympy.functions.special.delta_functions import DiracDelta, Heaviside
from sympy.series.order import Order

def change_mul(node, x) -> tuple[None, Any | Order | None] | tuple[DiracDelta, Any | Order]: ...
def deltaintegrate(f, x) -> Heaviside | Equality | Relational | Ne | None: ...
