import functools
import warnings
from collections.abc import Iterable, Iterator
from functools import update_wrapper as update_wrapper
from typing import Any, Callable

import joblib
from numpy import ndarray

from .._config import config_context as config_context, get_config as get_config

class Parallel(joblib.Parallel):
    def __call__(self, iterable: Iterable | Iterator[Any]) -> ndarray: ...

# remove when https://github.com/joblib/joblib/issues/1071 is fixed
def delayed(function: Callable) -> tuple | Callable: ...

class _FuncWrapper:
    def __init__(self, function: Callable) -> None: ...
    def with_config(self, config: dict[str, bool | int | str]) -> _FuncWrapper: ...
    def __call__(self, *args, **kwargs): ...
