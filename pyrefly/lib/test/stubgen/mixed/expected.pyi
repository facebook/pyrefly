# @generated
from typing import TypeAlias

Vector: TypeAlias = list[float]
MAX: int = 42


class Point:
    x: float
    y: float

    def __init__(self, x: float, y: float) -> None: ...

    def distance(self, other: "Point") -> float: ...


def origin() -> Point: ...
