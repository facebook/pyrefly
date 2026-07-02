# @generated
from typing import Callable, ClassVar, Self

from enum import Enum


class C:
    FILEMAP: ClassVar[dict[str, str]]
    DEFAULTS: ClassVar[list[int]]
    name: str

    def some_method(self) -> str: ...

    __repr__: ClassVar[Callable[[Self], str]]


class Color(Enum):
    RED: Literal[1] = 1
    GREEN: Literal[2] = 2
    BLUE: Literal[3] = 3
