from abc import ABCMeta, abstractmethod
from typing import Any, ClassVar
from typing_extensions import Self

from numpy import ndarray
from numpy.random import RandomState

from .._typing import ArrayLike, Int, MatrixLike
from ..base import BaseEstimator, ClassifierMixin, RegressorMixin
from ._base import BaseEnsemble

# Author: Gilles Louppe <g.louppe@gmail.com>
# License: BSD 3 clause

__all__ = ["BaggingClassifier", "BaggingRegressor"]

MAX_INT = ...

class BaseBagging(BaseEnsemble, metaclass=ABCMeta):
    _parameter_constraints: ClassVar[dict] = ...

    @abstractmethod
    def __init__(
        self,
        estimator=None,
        n_estimators: int = 10,
        *,
        max_samples: float = 1.0,
        max_features: float = 1.0,
        bootstrap: bool = True,
        bootstrap_features: bool = False,
        oob_score: bool = False,
        warm_start: bool = False,
        n_jobs=None,
        random_state=None,
        verbose: int = 0,
        base_estimator: str = "deprecated",
    ) -> None: ...
    def fit(
        self,
        X: MatrixLike | ArrayLike,
        y: ArrayLike,
        sample_weight: None | ArrayLike = None,
    ) -> BaggingRegressor | Self: ...
    def estimators_samples_(self): ...

class BaggingClassifier(ClassifierMixin, BaseBagging):
    oob_decision_function_: ndarray = ...
    oob_score_: float = ...
    n_classes_: list | int = ...
    classes_: ndarray = ...
    estimators_features_: list[ArrayLike] = ...
    estimators_samples_: list[ArrayLike] = ...
    estimators_: list[BaseEstimator] = ...
    feature_names_in_: ndarray = ...
    n_features_in_: int = ...
    base_estimator_: BaseEstimator = ...
    estimator_: BaseEstimator = ...

    def __init__(
        self,
        estimator: Any = None,
        n_estimators: Int = 10,
        *,
        max_samples: float = 1.0,
        max_features: float = 1.0,
        bootstrap: bool = True,
        bootstrap_features: bool = False,
        oob_score: bool = False,
        warm_start: bool = False,
        n_jobs: None | Int = None,
        random_state: RandomState | None | Int = None,
        verbose: Int = 0,
        base_estimator: Any = "deprecated",
    ) -> None: ...
    def predict(self, X: MatrixLike | ArrayLike) -> ndarray: ...
    def predict_proba(self, X: MatrixLike | ArrayLike) -> ndarray: ...
    def predict_log_proba(self, X: MatrixLike | ArrayLike) -> ndarray: ...
    def decision_function(self, X: MatrixLike | ArrayLike) -> ndarray: ...

class BaggingRegressor(RegressorMixin, BaseBagging):
    oob_prediction_: ndarray = ...
    oob_score_: float = ...
    estimators_features_: list[ArrayLike] = ...
    estimators_samples_: list[ArrayLike] = ...
    estimators_: list[BaseEstimator] = ...
    feature_names_in_: ndarray = ...
    n_features_in_: int = ...
    base_estimator_: BaseEstimator = ...
    estimator_: BaseEstimator = ...

    def __init__(
        self,
        estimator: Any = None,
        n_estimators: Int = 10,
        *,
        max_samples: float = 1.0,
        max_features: float = 1.0,
        bootstrap: bool = True,
        bootstrap_features: bool = False,
        oob_score: bool = False,
        warm_start: bool = False,
        n_jobs: None | Int = None,
        random_state: RandomState | None | Int = None,
        verbose: Int = 0,
        base_estimator: Any = "deprecated",
    ) -> None: ...
    def predict(self, X: MatrixLike | ArrayLike) -> ndarray: ...
