from typing_extensions import Self

from sympy import Basic
from sympy.stats.rv import PSpace

class RandomMatrixPSpace(PSpace):
    def __new__(cls, sym, model=...) -> Self: ...
    @property
    def model(self) -> Basic | None: ...
    def compute_density(self, expr, *args): ...
