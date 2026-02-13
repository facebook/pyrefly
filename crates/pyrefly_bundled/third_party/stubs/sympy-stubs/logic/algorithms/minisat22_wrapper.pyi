from collections.abc import Generator
from typing import Any, Literal, NoReturn

def minisat22_satisfiable(
    expr, all_models=..., minimal=...
) -> (
    Generator[bool, None, None] | dict[Any, Any] | Generator[dict[Any, Any] | Literal[False], Any, NoReturn] | Literal[False]
): ...
