# @generated
from typing import Callable

def transform(x: int, s: str) -> bool: ...


def variadic(*args: int) -> str: ...


handler: Callable[[int, str], bool]
collect: Callable[..., str]
