from collections.abc import Generator
from contextlib import contextmanager
from threading import local
from typing import Any

class DotProdSimpState(local):
    def __init__(self) -> None: ...

_dotprodsimp_state = ...

@contextmanager
def dotprodsimp(x) -> Generator[None, Any, None]: ...
