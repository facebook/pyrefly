# @generated
from typing import TypeAlias

__all__ = ["public_func", "PublicClass", "PUBLIC_VAR", "Vector"]
Vector: TypeAlias


def public_func(x: int) -> str: ...


class PublicClass:
    value: int

    def method(self) -> None: ...


PUBLIC_VAR: int = 20260317
