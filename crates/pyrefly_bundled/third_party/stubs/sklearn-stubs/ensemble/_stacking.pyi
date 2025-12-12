from abc import ABCMeta, abstractmethod
from collections.abc import Iterable, Sequence
from typing import ClassVar, Literal
from typing_extensions import Self

from numpy import ndarray

from .._typing import ArrayLike, Int, MatrixLike
from ..base import (
    BaseEstimator,
    ClassifierMixin,
    RegressorMixin,
    TransformerMixin,
)
from ..linear_model._logistic import LogisticRegression
from ..linear_model._ridge import RidgeCV
from ..model_selection import BaseCrossValidator
from ..pipeline import Pipeline
from ..utils import Bunch
from ._base import _BaseHeterogeneousEnsemble

class _BaseStacking(TransformerMixin, _BaseHeterogeneousEnsemble, metaclass=ABCMeta):
    _parameter_constraints: ClassVar[dict] = ...

    @abstractmethod
    def __init__(
        self,
        estimators,
        final_estimator=None,
        *,
        cv=None,
        stack_method: str = "auto",
        n_jobs=None,
        verbose: int = 0,
        passthrough: bool = False,
    ) -> None: ...
    def fit(
        self,
        X: MatrixLike | ArrayLike,
        y: ArrayLike,
        sample_weight: None | ArrayLike = None,
    ) -> Self | StackingClassifier: ...
    def n_features_in_(self): ...
    def get_feature_names_out(self, input_features: None | ArrayLike = None) -> ndarray: ...
    def predict(self, X: MatrixLike | ArrayLike, **predict_params) -> ndarray: ...

class StackingClassifier(ClassifierMixin, _BaseStacking):
    stack_method_: list[str] = ...
    final_estimator_: BaseEstimator = ...
    feature_names_in_: ndarray = ...
    n_features_in_: int = ...
    named_estimators_: Bunch = ...
    estimators_: list[BaseEstimator] = ...
    classes_: ndarray = ...

    _parameter_constraints: ClassVar[dict] = ...

    def __init__(
        self,
        estimators: Sequence[tuple[str, BaseEstimator]],
        final_estimator: None | BaseEstimator | LogisticRegression = None,
        *,
        cv: int | BaseCrossValidator | Iterable | None | str = None,
        stack_method: Literal["auto", "predict_proba", "decision_function", "predict"] = "auto",
        n_jobs: None | Int = None,
        passthrough: bool = False,
        verbose: Int = 0,
    ) -> None: ...
    def fit(
        self,
        X: MatrixLike | ArrayLike,
        y: ArrayLike,
        sample_weight: None | ArrayLike = None,
    ) -> Self: ...
    def predict(self, X: MatrixLike | ArrayLike, **predict_params) -> ndarray: ...
    def predict_proba(self, X: MatrixLike | ArrayLike) -> ndarray | list[ndarray]: ...
    def decision_function(self, X: MatrixLike | ArrayLike) -> ndarray: ...
    def transform(self, X: MatrixLike | ArrayLike) -> ndarray: ...

class StackingRegressor(RegressorMixin, _BaseStacking):
    stack_method_: list[str] = ...
    final_estimator_: BaseEstimator = ...
    feature_names_in_: ndarray = ...
    n_features_in_: int = ...
    named_estimators_: Bunch = ...
    estimators_: list[BaseEstimator] = ...

    def __init__(
        self,
        estimators: Sequence[tuple[str, BaseEstimator]] | list[tuple[str, Pipeline]],
        final_estimator: None | BaseEstimator | RidgeCV = None,
        *,
        cv: int | BaseCrossValidator | Iterable | None | str = None,
        n_jobs: None | Int = None,
        passthrough: bool = False,
        verbose: Int = 0,
    ) -> None: ...
    def fit(
        self,
        X: MatrixLike | ArrayLike,
        y: ArrayLike,
        sample_weight: None | ArrayLike = None,
    ) -> Self: ...
    def transform(self, X: MatrixLike | ArrayLike) -> ndarray: ...
