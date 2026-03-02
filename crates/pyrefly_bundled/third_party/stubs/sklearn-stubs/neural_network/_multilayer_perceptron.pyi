from abc import ABCMeta, abstractmethod
from typing import ClassVar, Literal
from typing_extensions import Self

from numpy import ndarray
from numpy.random import RandomState

from .._typing import ArrayLike, Float, Int, MatrixLike
from ..base import BaseEstimator, ClassifierMixin, RegressorMixin

_STOCHASTIC_SOLVERS: list = ...

class BaseMultilayerPerceptron(BaseEstimator, metaclass=ABCMeta):
    _parameter_constraints: ClassVar[dict] = ...

    @abstractmethod
    def __init__(
        self,
        hidden_layer_sizes: tuple[int, int] | tuple[int] | list[int],
        activation: str,
        solver: str,
        alpha: Float,
        batch_size: str,
        learning_rate: str,
        learning_rate_init: float,
        power_t: float,
        max_iter: int,
        loss: str,
        shuffle: bool,
        random_state: None | int,
        tol: float,
        verbose: int | bool,
        warm_start: bool,
        momentum: float,
        nesterovs_momentum: bool,
        early_stopping: bool,
        validation_fraction: float,
        beta_1: float,
        beta_2: float,
        epsilon: float,
        n_iter_no_change: int,
        max_fun: int,
    ) -> None: ...
    def fit(self, X: MatrixLike, y: MatrixLike | ArrayLike) -> Self: ...

class MLPClassifier(ClassifierMixin, BaseMultilayerPerceptron):
    out_activation_: str = ...
    n_outputs_: int = ...
    n_layers_: int = ...
    n_iter_: int = ...
    feature_names_in_: ndarray = ...
    n_features_in_: int = ...
    intercepts_: list = ...
    coefs_: list = ...
    t_: int = ...
    best_validation_score_: float | None = ...
    validation_scores_: None | list = ...
    loss_curve_: list = ...
    best_loss_: float | None = ...
    loss_: float = ...
    classes_: ndarray | list[ndarray] = ...

    def __init__(
        self,
        hidden_layer_sizes: ArrayLike | tuple[int] = ...,
        activation: Literal["relu", "identity", "logistic", "tanh"] = "relu",
        *,
        solver: Literal["lbfgs", "sgd", "adam"] = "adam",
        alpha: Float = 0.0001,
        batch_size: str | Int = "auto",
        learning_rate: Literal["constant", "invscaling", "adaptive"] = "constant",
        learning_rate_init: Float = 0.001,
        power_t: Float = 0.5,
        max_iter: Int = 200,
        shuffle: bool = True,
        random_state: RandomState | None | Int = None,
        tol: Float = 1e-4,
        verbose: bool = False,
        warm_start: bool = False,
        momentum: Float = 0.9,
        nesterovs_momentum: bool = True,
        early_stopping: bool = False,
        validation_fraction: Float = 0.1,
        beta_1: Float = 0.9,
        beta_2: Float = 0.999,
        epsilon: Float = 1e-8,
        n_iter_no_change: Int = 10,
        max_fun: Int = 15000,
    ) -> None: ...
    def predict(self, X: MatrixLike | ArrayLike) -> ndarray: ...
    def partial_fit(
        self,
        X: MatrixLike | ArrayLike,
        y: ArrayLike,
        classes: None | ArrayLike = None,
    ) -> Self: ...
    def predict_log_proba(self, X: ArrayLike) -> ndarray: ...
    def predict_proba(self, X: MatrixLike | ArrayLike) -> ndarray: ...

class MLPRegressor(RegressorMixin, BaseMultilayerPerceptron):
    out_activation_: str = ...
    n_outputs_: int = ...
    n_layers_: int = ...
    n_iter_: int = ...
    feature_names_in_: ndarray = ...
    n_features_in_: int = ...
    intercepts_: list = ...
    coefs_: list = ...
    t_: int = ...
    best_validation_score_: float | None = ...
    validation_scores_: None | list = ...
    loss_curve_: list = ...
    best_loss_: float = ...
    loss_: float = ...

    def __init__(
        self,
        hidden_layer_sizes: ArrayLike | tuple[int, int] = ...,
        activation: Literal["relu", "identity", "logistic", "tanh"] = "relu",
        *,
        solver: Literal["lbfgs", "sgd", "adam"] = "adam",
        alpha: Float = 0.0001,
        batch_size: str | Int = "auto",
        learning_rate: Literal["constant", "invscaling", "adaptive"] = "constant",
        learning_rate_init: Float = 0.001,
        power_t: Float = 0.5,
        max_iter: Int = 200,
        shuffle: bool = True,
        random_state: RandomState | None | Int = None,
        tol: Float = 1e-4,
        verbose: bool = False,
        warm_start: bool = False,
        momentum: Float = 0.9,
        nesterovs_momentum: bool = True,
        early_stopping: bool = False,
        validation_fraction: Float = 0.1,
        beta_1: Float = 0.9,
        beta_2: Float = 0.999,
        epsilon: Float = 1e-8,
        n_iter_no_change: Int = 10,
        max_fun: Int = 15000,
    ) -> None: ...
    def predict(self, X: MatrixLike | ArrayLike) -> ndarray: ...
    def partial_fit(self, X: MatrixLike | ArrayLike, y: ArrayLike) -> Self: ...
