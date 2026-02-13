from collections.abc import Sequence
from typing import Literal

from numpy import ndarray

_MODULE_TO_IGNORE: set = ...

def all_estimators(
    type_filter: (
        None
        | Sequence[Literal["classifier", "regressor", "cluster", "transformer"]]
        | Literal["classifier", "regressor", "cluster", "transformer"]
    ) = None,
) -> ndarray: ...
def all_displays() -> ndarray: ...
def all_functions() -> ndarray: ...
