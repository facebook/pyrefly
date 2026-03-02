from abc import ABCMeta, abstractmethod
from collections.abc import Sequence
from typing import Any, ClassVar
from typing_extensions import Self

from .._typing import Int
from ..base import BaseEstimator, MetaEstimatorMixin
from ..utils import Bunch
from ..utils.metaestimators import _BaseComposition

class BaseEnsemble(MetaEstimatorMixin, BaseEstimator, metaclass=ABCMeta):
    estimators_: list[BaseEstimator] = ...

    # overwrite _required_parameters from MetaEstimatorMixin
    _required_parameters: ClassVar[list[str]] = ...

    @abstractmethod
    def __init__(
        self,
        estimator: Any = None,
        *,
        n_estimators: Int = 10,
        estimator_params: Sequence[str] = ...,
        base_estimator: Any = "deprecated",
    ) -> None: ...
    def __len__(self) -> int: ...
    def __getitem__(self, index): ...
    def __iter__(self): ...

class _BaseHeterogeneousEnsemble(MetaEstimatorMixin, _BaseComposition, metaclass=ABCMeta):
    estimators_: list[BaseEstimator] = ...

    _required_parameters: ClassVar[list] = ...

    @property
    def named_estimators(self) -> Bunch: ...
    @abstractmethod
    def __init__(self, estimators: list[tuple[str, BaseEstimator]]) -> None: ...
    def set_params(self, **params) -> Self: ...
    def get_params(self, deep: bool = True) -> dict: ...
