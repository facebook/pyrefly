from typing import Any

from sympy import Idx, Indexed, Piecewise
from sympy.core.function import Function
from sympy.functions.elementary.exponential import exp

class IndexConformanceException(Exception): ...

def get_indices(
    expr,
) -> (
    tuple[set[Any], dict[Any, Any]]
    | tuple[set[Idx], dict[Any, Any]]
    | tuple[set[Any], dict[Any, Any], tuple[Any, ...]]
    | tuple[Any, Any | dict[Any, Any]]
    | tuple[Any, dict[Any, Any]]
    | tuple[set[Any] | Any, Any]
): ...
def get_contraction_structure(
    expr,
) -> (
    dict[tuple[Any, ...] | None, set[Indexed]]
    | dict[None, set[Any]]
    | dict[tuple[Any, ...] | None, set[Any]]
    | dict[None, set[Any | exp]]
    | dict[Any, Any]
    | dict[None, Piecewise]
    | dict[None, set[Function]]
): ...
