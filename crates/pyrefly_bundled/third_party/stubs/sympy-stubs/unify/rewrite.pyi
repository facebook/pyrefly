from collections.abc import Generator
from typing import Any, Callable

from sympy import Basic

def rewriterule(source, target, variables=..., condition=..., assume=...) -> Callable[..., Generator[Any | Basic, Any, None]]: ...
