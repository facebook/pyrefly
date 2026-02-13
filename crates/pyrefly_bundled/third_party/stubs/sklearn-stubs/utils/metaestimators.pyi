from abc import ABCMeta, abstractmethod
from contextlib import suppress as suppress
from operator import attrgetter as attrgetter
from typing import Any, ClassVar

from ..base import BaseEstimator
from ._available_if import available_if

__all__ = ["available_if"]

class _BaseComposition(BaseEstimator, metaclass=ABCMeta):
    steps: ClassVar[list[Any]] = ...

    @abstractmethod
    def __init__(self) -> None: ...
