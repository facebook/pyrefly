from typing import ClassVar
from typing_extensions import Self

from numpy import ndarray

from .._typing import ArrayLike, Float, Int, MatrixLike
from ..base import BaseEstimator, RegressorMixin
from ._base import LinearModel

# Authors: Manoj Kumar mks542@nyu.edu
# License: BSD 3 clause

class HuberRegressor(LinearModel, RegressorMixin, BaseEstimator):
    outliers_: ndarray = ...
    n_iter_: int = ...
    feature_names_in_: ndarray = ...
    n_features_in_: int = ...
    scale_: float = ...
    intercept_: float = ...
    coef_: ndarray = ...

    _parameter_constraints: ClassVar[dict] = ...

    def __init__(
        self,
        *,
        epsilon: Float = 1.35,
        max_iter: Int = 100,
        alpha: Float = 0.0001,
        warm_start: bool = False,
        fit_intercept: bool = True,
        tol: Float = 1e-05,
    ) -> None: ...
    def fit(
        self,
        X: MatrixLike,
        y: ArrayLike,
        sample_weight: None | ArrayLike = None,
    ) -> Self: ...
