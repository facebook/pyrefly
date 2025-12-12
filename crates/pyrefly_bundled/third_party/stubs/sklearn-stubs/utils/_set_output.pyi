from typing import Literal

from ..base import BaseEstimator

class _SetOutputMixin:
    def __init_subclass__(cls, auto_wrap_output_keys: None | tuple[str] = ..., **kwargs) -> None: ...
    def set_output(self, *, transform: None | Literal["default", "pandas"] = None) -> BaseEstimator: ...
