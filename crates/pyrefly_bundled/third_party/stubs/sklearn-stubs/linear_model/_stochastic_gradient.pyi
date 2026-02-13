from abc import ABCMeta, abstractmethod
from collections.abc import Mapping
from typing import Any, ClassVar, Literal
from typing_extensions import Self

from numpy import ndarray
from numpy.random import RandomState
from scipy.sparse._csr import csr_matrix

from .._typing import ArrayLike, Float, Int, MatrixLike
from ..base import BaseEstimator, OutlierMixin, RegressorMixin
from ._base import LinearClassifierMixin, SparseCoefMixin
from ._sgd_fast import (
    LossFunction,
)

# Authors: Peter Prettenhofer <peter.prettenhofer@gmail.com> (main author)
#          Mathieu Blondel (partial_fit support)
#
# License: BSD 3 clause

LEARNING_RATE_TYPES: dict = ...

PENALTY_TYPES: dict = ...

DEFAULT_EPSILON: float = ...
# Default value of ``epsilon`` parameter.

MAX_INT = ...

class _ValidationScoreCallback:
    def __init__(
        self,
        estimator: SGDClassifier,
        X_val: csr_matrix | ndarray,
        y_val: ndarray,
        sample_weight_val: ndarray,
        classes: None | ndarray = None,
    ) -> None: ...
    def __call__(self, coef: ndarray, intercept: float) -> Float: ...

class BaseSGD(SparseCoefMixin, BaseEstimator, metaclass=ABCMeta):
    _parameter_constraints: ClassVar[dict] = ...

    def __init__(
        self,
        loss: str,
        *,
        penalty: str = "l2",
        alpha: float = 0.0001,
        C: float = 1.0,
        l1_ratio: float = 0.15,
        fit_intercept: bool = True,
        max_iter: int = 1000,
        tol: float = 1e-3,
        shuffle: bool = True,
        verbose: int = 0,
        epsilon: float = 0.1,
        random_state=None,
        learning_rate: str = "optimal",
        eta0: float = 0.0,
        power_t: float = 0.5,
        early_stopping: bool = False,
        validation_fraction: float = 0.1,
        n_iter_no_change: int = 5,
        warm_start: bool = False,
        average: bool = False,
    ) -> None: ...
    @abstractmethod
    def fit(self, X, y): ...

def fit_binary(
    est: BaseEstimator,
    i: Int,
    X: MatrixLike,
    y: ArrayLike,
    alpha: Float,
    C: Float,
    learning_rate: str,
    max_iter: Int,
    pos_weight: Float,
    neg_weight: Float,
    sample_weight: ArrayLike,
    validation_mask: None | ArrayLike = None,
    random_state: RandomState | None | Int = None,
) -> tuple[ndarray, float, int]: ...

class BaseSGDClassifier(LinearClassifierMixin, BaseSGD, metaclass=ABCMeta):
    loss_functions: ClassVar[dict] = ...

    _parameter_constraints: ClassVar[dict] = ...

    @abstractmethod
    def __init__(
        self,
        loss: str = "hinge",
        *,
        penalty: str = "l2",
        alpha: float = 0.0001,
        l1_ratio: float = 0.15,
        fit_intercept: bool = True,
        max_iter: int = 1000,
        tol: float = 1e-3,
        shuffle: bool = True,
        verbose: int = 0,
        epsilon=...,
        n_jobs=None,
        random_state=None,
        learning_rate: str = "optimal",
        eta0: float = 0.0,
        power_t: float = 0.5,
        early_stopping: bool = False,
        validation_fraction: float = 0.1,
        n_iter_no_change: int = 5,
        class_weight=None,
        warm_start: bool = False,
        average: bool = False,
    ) -> None: ...
    def partial_fit(
        self,
        X: MatrixLike,
        y: ArrayLike,
        classes: None | ArrayLike = None,
        sample_weight: None | ArrayLike = None,
    ) -> Self: ...
    def fit(
        self,
        X: MatrixLike,
        y: ArrayLike,
        coef_init: None | MatrixLike = None,
        intercept_init: None | ArrayLike = None,
        sample_weight: None | ArrayLike = None,
    ) -> Self: ...

class SGDClassifier(BaseSGDClassifier):
    feature_names_in_: ndarray = ...
    n_features_in_: int = ...
    t_: int = ...
    classes_: ndarray = ...
    loss_function_: LossFunction = ...
    n_iter_: int = ...
    intercept_: ndarray = ...
    coef_: ndarray = ...

    _parameter_constraints: ClassVar[dict] = ...

    def __init__(
        self,
        loss: Literal[
            "hinge",
            "log_loss",
            "log",
            "modified_huber",
            "squared_hinge",
            "perceptron",
            "squared_error",
            "huber",
            "epsilon_insensitive",
            "squared_epsilon_insensitive",
        ] = "hinge",
        *,
        penalty: None | Literal["l2", "l1", "elasticnet"] = "l2",
        alpha: Float = 0.0001,
        l1_ratio: Float = 0.15,
        fit_intercept: bool = True,
        max_iter: Int = 1000,
        tol: None | Float = 1e-3,
        shuffle: bool = True,
        verbose: Int = 0,
        epsilon: Float = ...,
        n_jobs: None | Int = None,
        random_state: RandomState | None | Int = None,
        learning_rate: str = "optimal",
        eta0: Float = 0.0,
        power_t: Float = 0.5,
        early_stopping: bool = False,
        validation_fraction: Float = 0.1,
        n_iter_no_change: Int = 5,
        class_weight: Mapping[str, float] | None | str = None,
        warm_start: bool = False,
        average: int | bool = False,
    ) -> None: ...
    def predict_proba(self, X: MatrixLike) -> ndarray: ...
    def predict_log_proba(self, X: MatrixLike | ArrayLike) -> ndarray: ...

class BaseSGDRegressor(RegressorMixin, BaseSGD):
    loss_functions: ClassVar[dict] = ...

    _parameter_constraints: ClassVar[dict] = ...

    @abstractmethod
    def __init__(
        self,
        loss: str = "squared_error",
        *,
        penalty: str = "l2",
        alpha: float = 0.0001,
        l1_ratio: float = 0.15,
        fit_intercept: bool = True,
        max_iter: int = 1000,
        tol: float = 1e-3,
        shuffle: bool = True,
        verbose: int = 0,
        epsilon=...,
        random_state=None,
        learning_rate: str = "invscaling",
        eta0: float = 0.01,
        power_t: float = 0.25,
        early_stopping: bool = False,
        validation_fraction: float = 0.1,
        n_iter_no_change: int = 5,
        warm_start: bool = False,
        average: bool = False,
    ) -> None: ...
    def partial_fit(
        self,
        X: MatrixLike,
        y: ArrayLike,
        sample_weight: None | ArrayLike = None,
    ) -> Self: ...
    def fit(
        self,
        X: MatrixLike,
        y: ArrayLike,
        coef_init: None | ArrayLike = None,
        intercept_init: None | ArrayLike = None,
        sample_weight: None | ArrayLike = None,
    ) -> SGDRegressor | Self: ...
    def predict(self, X: MatrixLike) -> ndarray: ...

class SGDRegressor(BaseSGDRegressor):
    feature_names_in_: ndarray = ...
    n_features_in_: int = ...
    t_: int = ...
    n_iter_: int = ...
    intercept_: ndarray = ...
    coef_: ndarray = ...

    _parameter_constraints: ClassVar[dict] = ...

    def __init__(
        self,
        loss: str = "squared_error",
        *,
        penalty: None | Literal["l2", "l1", "elasticnet"] = "l2",
        alpha: Float = 0.0001,
        l1_ratio: Float = 0.15,
        fit_intercept: bool = True,
        max_iter: Int = 1000,
        tol: None | Float = 1e-3,
        shuffle: bool = True,
        verbose: Int = 0,
        epsilon: Float = ...,
        random_state: RandomState | None | Int = None,
        learning_rate: str = "invscaling",
        eta0: Float = 0.01,
        power_t: Float = 0.25,
        early_stopping: bool = False,
        validation_fraction: Float = 0.1,
        n_iter_no_change: Int = 5,
        warm_start: bool = False,
        average: int | bool = False,
    ) -> None: ...

class SGDOneClassSVM(BaseSGD, OutlierMixin):
    feature_names_in_: ndarray = ...
    n_features_in_: int = ...
    loss_function_: LossFunction = ...
    t_: int = ...
    n_iter_: int = ...
    offset_: ndarray = ...
    coef_: ndarray = ...

    loss_functions: ClassVar[dict] = ...

    _parameter_constraints: ClassVar[dict] = ...

    def __init__(
        self,
        nu: Float = 0.5,
        fit_intercept: bool = True,
        max_iter: Int = 1000,
        tol: None | Float = 1e-3,
        shuffle: bool = True,
        verbose: Int = 0,
        random_state: RandomState | None | Int = None,
        learning_rate: Literal["optimal", "constant", "invscaling", "adaptive"] = "optimal",
        eta0: Float = 0.0,
        power_t: Float = 0.5,
        warm_start: bool = False,
        average: int | bool = False,
    ) -> None: ...
    def partial_fit(
        self,
        X: MatrixLike,
        y: Any = None,
        sample_weight: None | ArrayLike = None,
    ) -> Self: ...
    def fit(
        self,
        X: MatrixLike,
        y: Any = None,
        coef_init: None | MatrixLike = None,
        offset_init: None | ArrayLike = None,
        sample_weight: None | ArrayLike = None,
    ) -> Self: ...
    def decision_function(self, X: MatrixLike) -> ndarray: ...
    def score_samples(self, X: MatrixLike) -> ndarray: ...
    def predict(self, X: MatrixLike) -> ndarray: ...
