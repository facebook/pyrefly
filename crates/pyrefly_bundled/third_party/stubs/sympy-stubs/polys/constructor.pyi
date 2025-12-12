from typing import Any

from sympy.polys.domains.gaussiandomains import GaussianIntegerRing, GaussianRationalField
from sympy.utilities import public

@public
def construct_domain(
    obj, **args
) -> (
    tuple[Any | GaussianRationalField | GaussianIntegerRing, dict[Any, Any | int]]
    | tuple[Any | GaussianRationalField | GaussianIntegerRing, list[Any | int] | list[Any] | Any]
    | tuple[Any | GaussianRationalField | GaussianIntegerRing, Any | int]
): ...
