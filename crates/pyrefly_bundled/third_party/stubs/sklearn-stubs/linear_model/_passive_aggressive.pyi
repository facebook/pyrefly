from collections.abc import Mapping
from typing import Callable, ClassVar
from typing_extensions import Self

from numpy import ndarray
from numpy.random import RandomState

from .._typing import ArrayLike, Float, Int, MatrixLike
from ._stochastic_gradient import BaseSGDClassifier, BaseSGDRegressor

# Authors: Rob Zinkov, Mathieu Blondel
# License: BSD 3 clause

class PassiveAggressiveClassifier(BaseSGDClassifier):
    loss_function_: Callable = ...
    t_: int = ...
    classes_: ndarray = ...
    n_iter_: int = ...
    feature_names_in_: ndarray = ...
    n_features_in_: int = ...
    intercept_: ndarray = ...
    coef_: ndarray = ...

    _parameter_constraints: ClassVar[dict] = ...

    def __init__(
        self,
        *,
        C: Float = 1.0,
        fit_intercept: bool = True,
        max_iter: Int = 1000,
        tol: None | Float = 1e-3,
        early_stopping: bool = False,
        validation_fraction: Float = 0.1,
        n_iter_no_change: Int = 5,
        shuffle: bool = True,
        verbose: Int = 0,
        loss: str = "hinge",
        n_jobs: None | int = None,
        random_state: RandomState | None | Int = None,
        warm_start: bool = False,
        class_weight: None | str | Mapping[str, float] = None,
        average: int | bool = False,
    ) -> None: ...
    def partial_fit(
        self,
        X: MatrixLike | ArrayLike,
        y: ArrayLike,
        classes: None | ArrayLike = None,
    ) -> Self: ...
    def fit(
        self,
        X: MatrixLike | ArrayLike,
        y: ArrayLike,
        coef_init: None | MatrixLike = None,
        intercept_init: None | ArrayLike = None,
    ) -> Self: ...

class PassiveAggressiveRegressor(BaseSGDRegressor):
    t_: int = ...
    n_iter_: int = ...
    feature_names_in_: ndarray = ...
    n_features_in_: int = ...
    intercept_: ndarray = ...
    coef_: ndarray = ...

    _parameter_constraints: ClassVar[dict] = ...

    def __init__(
        self,
        *,
        C: Float = 1.0,
        fit_intercept: bool = True,
        max_iter: Int = 1000,
        tol: None | Float = 1e-3,
        early_stopping: bool = False,
        validation_fraction: Float = 0.1,
        n_iter_no_change: Int = 5,
        shuffle: bool = True,
        verbose: Int = 0,
        loss: str = "epsilon_insensitive",
        epsilon: Float = ...,
        random_state: RandomState | None | Int = None,
        warm_start: bool = False,
        average: int | bool = False,
    ) -> None: ...
    def partial_fit(self, X: MatrixLike | ArrayLike, y: ArrayLike) -> Self: ...
    def fit(
        self,
        X: MatrixLike | ArrayLike,
        y: ArrayLike,
        coef_init: None | ArrayLike = None,
        intercept_init: None | ArrayLike = None,
    ) -> Self: ...
