from typing import ClassVar, Literal
from typing_extensions import Self

from numpy import ndarray

from .._typing import ArrayLike, Float, MatrixLike
from ..base import BaseEstimator, RegressorMixin
from ._base import LinearModel

# Authors: David Dale <dale.david@mail.ru>
#          Christian Lorentzen <lorentzen.ch@gmail.com>
# License: BSD 3 clause

class QuantileRegressor(LinearModel, RegressorMixin, BaseEstimator):
    n_iter_: int = ...
    feature_names_in_: ndarray = ...
    n_features_in_: int = ...
    intercept_: float = ...
    coef_: ndarray = ...

    _parameter_constraints: ClassVar[dict] = ...

    def __init__(
        self,
        *,
        quantile: Float = 0.5,
        alpha: Float = 1.0,
        fit_intercept: bool = True,
        solver: Literal[
            "highs-ds",
            "highs-ipm",
            "highs",
            "interior-point",
            "revised simplex",
            "warn",
        ] = "warn",
        solver_options: None | dict = None,
    ) -> None: ...
    def fit(
        self,
        X: MatrixLike | ArrayLike,
        y: ArrayLike,
        sample_weight: None | ArrayLike = None,
    ) -> Self: ...
