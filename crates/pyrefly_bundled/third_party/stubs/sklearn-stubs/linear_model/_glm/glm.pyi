from typing import ClassVar, Literal
from typing_extensions import Self

from numpy import ndarray

from ..._loss.loss import (
    BaseLoss,
)
from ..._typing import ArrayLike, Float, Int, MatrixLike
from ...base import BaseEstimator, RegressorMixin

class _GeneralizedLinearRegressor(RegressorMixin, BaseEstimator):
    _base_loss: BaseLoss = ...
    n_iter_: int = ...
    intercept_: float = ...
    coef_: ndarray = ...

    # We allow for NewtonSolver classes for the "solver" parameter but do not
    # make them public in the docstrings. This facilitates testing and
    # benchmarking.
    _parameter_constraints: ClassVar[dict] = ...

    def __init__(
        self,
        *,
        alpha: Float = 1.0,
        fit_intercept: bool = True,
        solver: Literal["lbfgs", "newton-cholesky"] = "lbfgs",
        max_iter: Int = 100,
        tol: Float = 1e-4,
        warm_start: bool = False,
        verbose: Int = 0,
    ) -> None: ...
    def fit(
        self,
        X: MatrixLike | ArrayLike,
        y: ArrayLike,
        sample_weight: None | ArrayLike = None,
    ) -> Self: ...
    def predict(self, X: MatrixLike | ArrayLike) -> ndarray: ...
    def score(
        self,
        X: MatrixLike | ArrayLike,
        y: ArrayLike,
        sample_weight: None | ArrayLike = None,
    ) -> Float: ...

class PoissonRegressor(_GeneralizedLinearRegressor):
    n_iter_: int = ...
    feature_names_in_: ndarray = ...
    n_features_in_: int = ...
    intercept_: float = ...
    coef_: ndarray = ...

    _parameter_constraints: ClassVar[dict] = ...

    def __init__(
        self,
        *,
        alpha: Float = 1.0,
        fit_intercept: bool = True,
        solver: Literal["lbfgs", "newton-cholesky"] = "lbfgs",
        max_iter: Int = 100,
        tol: Float = 1e-4,
        warm_start: bool = False,
        verbose: Int = 0,
    ) -> None: ...

class GammaRegressor(_GeneralizedLinearRegressor):
    feature_names_in_: ndarray = ...
    n_iter_: int = ...
    n_features_in_: int = ...
    intercept_: float = ...
    coef_: ndarray = ...

    _parameter_constraints: ClassVar[dict] = ...

    def __init__(
        self,
        *,
        alpha: Float = 1.0,
        fit_intercept: bool = True,
        solver: Literal["lbfgs", "newton-cholesky"] = "lbfgs",
        max_iter: Int = 100,
        tol: Float = 1e-4,
        warm_start: bool = False,
        verbose: Int = 0,
    ) -> None: ...

class TweedieRegressor(_GeneralizedLinearRegressor):
    feature_names_in_: ndarray = ...
    n_features_in_: int = ...
    n_iter_: int = ...
    intercept_: float = ...
    coef_: ndarray = ...

    _parameter_constraints: ClassVar[dict] = ...

    def __init__(
        self,
        *,
        power: Float = 0.0,
        alpha: Float = 1.0,
        fit_intercept: bool = True,
        link: Literal["auto", "identity", "log"] = "auto",
        solver: Literal["lbfgs", "newton-cholesky"] = "lbfgs",
        max_iter: Int = 100,
        tol: Float = 1e-4,
        warm_start: bool = False,
        verbose: Int = 0,
    ) -> None: ...
