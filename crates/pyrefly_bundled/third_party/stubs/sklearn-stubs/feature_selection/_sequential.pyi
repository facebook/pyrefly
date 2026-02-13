from collections.abc import Iterable
from typing import Callable, ClassVar, Literal
from typing_extensions import Self

from numpy import ndarray

from .._typing import ArrayLike, Float, Int, MatrixLike
from ..base import BaseEstimator, MetaEstimatorMixin
from ..model_selection import BaseCrossValidator
from ._base import SelectorMixin

class SequentialFeatureSelector(SelectorMixin, MetaEstimatorMixin, BaseEstimator):
    support_: ndarray = ...
    n_features_to_select_: int = ...
    feature_names_in_: ndarray = ...
    n_features_in_: int = ...

    _parameter_constraints: ClassVar[dict] = ...

    def __init__(
        self,
        estimator: BaseEstimator,
        *,
        n_features_to_select: float | Literal["auto", "warn"] = "warn",
        tol: None | Float = None,
        direction: Literal["forward", "backward"] = "forward",
        scoring: None | str | Callable = None,
        cv: Iterable | int | BaseCrossValidator = 5,
        n_jobs: None | Int = None,
    ) -> None: ...
    def fit(self, X: MatrixLike, y: None | ArrayLike = None) -> Self: ...
