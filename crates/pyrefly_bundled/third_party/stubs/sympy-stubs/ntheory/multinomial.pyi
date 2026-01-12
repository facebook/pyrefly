from collections.abc import Generator
from typing import Any, Literal

def binomial_coefficients(n) -> dict[tuple[Literal[0], int] | tuple[int, Literal[0]], int]: ...
def binomial_coefficients_list(n) -> list[int]: ...
def multinomial_coefficients(
    m, n
) -> (
    dict[Any, Any]
    | dict[tuple[()], int]
    | dict[tuple[Literal[0], int] | tuple[int, Literal[0]], int]
    | dict[tuple[Any, ...], Any]
    | dict[tuple[int, ...], int]
): ...
def multinomial_coefficients_iterator(m, n, _tuple=...) -> Generator[Any | tuple[tuple[Any, ...], Any], Any, None]: ...
