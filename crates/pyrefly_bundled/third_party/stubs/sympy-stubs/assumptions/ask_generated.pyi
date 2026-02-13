from typing import Any

from sympy.assumptions.cnf import AND, OR, Literal
from sympy.core.cache import cacheit

@cacheit
def get_all_known_facts() -> set[frozenset[OR | AND | Literal]]: ...
@cacheit
def get_known_facts_dict() -> dict[Any, Any]: ...
