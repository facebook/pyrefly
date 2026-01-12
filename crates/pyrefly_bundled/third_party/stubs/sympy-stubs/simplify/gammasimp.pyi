from typing import Any

from sympy.core import Function
from sympy.series.order import Order

def gammasimp(expr) -> Any: ...

class _rf(Function):
    @classmethod
    def eval(cls, a, b) -> Order | None: ...
