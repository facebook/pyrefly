from collections.abc import Sequence
from typing import Literal

from ..base import BaseEstimator

class _IDCounter:
    def __init__(self, prefix: str) -> None: ...
    def get_id(self): ...

_CONTAINER_ID_COUNTER = ...
_ESTIMATOR_ID_COUNTER = ...

class _VisualBlock:
    def __init__(
        self,
        kind: Literal["serial", "parallel", "single"],
        estimators: Sequence[BaseEstimator] | BaseEstimator | Sequence[_VisualBlock],
        *,
        names: None | Sequence[str] = None,
        name_details: Sequence[str] | None | str = None,
        dash_wrapped: bool = True,
    ) -> None: ...

_STYLE = ...

def estimator_html_repr(estimator: BaseEstimator) -> str: ...
