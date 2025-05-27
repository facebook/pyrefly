"""
Bug from https://github.com/facebook/pyrefly/issues/359 where overlaoded stub declared with docstring instead of ellipses shows an error
"""
from typing import *

@overload
def foo(a: str) -> str:      # <-- Function declared to return `str` but is missing an explicit `return`
    """Docstring"""

@overload
def foo(a: int) -> int: ...  # This is OK

def foo(*args, **kwargs) -> Any:
    pass