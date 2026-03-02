from collections.abc import Generator
from typing import Any, Callable

def canon(*rules) -> Callable[..., Generator[Any, Any, None]]: ...
