from collections.abc import Generator
from typing import Any, Literal

__all__ = ["default_sort_key", "ordered"]

def default_sort_key(
    item, order=...
) -> (
    tuple[tuple[Literal[5], Literal[0], str], tuple[int, tuple[Any, ...]], Any, Any]
    | tuple[tuple[Literal[10, 0], Literal[0], str | Any], tuple[int, tuple[Any, ...]] | tuple[Literal[1], tuple[str]], Any, Any]
): ...
def ordered(seq, keys=..., default=..., warn=...) -> Generator[Any, Any, None]: ...
