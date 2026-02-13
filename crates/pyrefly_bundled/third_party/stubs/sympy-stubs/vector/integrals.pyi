from typing import Literal
from typing_extensions import Self

from sympy import Equality, Ne
from sympy.core import Basic
from sympy.core.relational import Relational

class ParametricIntegral(Basic):
    def __new__(cls, field, parametricregion) -> Equality | Relational | Ne | Self: ...
    @property
    def field(self) -> Basic: ...
    @property
    def parametricregion(self) -> Basic: ...

def vector_integrate(field, *region) -> Equality | Relational | Ne | ParametricIntegral | Literal[0]: ...
