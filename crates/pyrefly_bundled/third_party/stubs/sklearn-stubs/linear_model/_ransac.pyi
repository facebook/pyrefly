from typing import Any, Callable, ClassVar
from typing_extensions import Self

from numpy import ndarray
from numpy.random import RandomState

from .._typing import ArrayLike, Float, Int, MatrixLike
from ..base import BaseEstimator, MetaEstimatorMixin, MultiOutputMixin, RegressorMixin

# Author: Johannes Schönberger
#
# License: BSD 3 clause

_EPSILON = ...

class RANSACRegressor(MetaEstimatorMixin, RegressorMixin, MultiOutputMixin, BaseEstimator):
    feature_names_in_: ndarray = ...
    n_features_in_: int = ...
    n_skips_invalid_model_: int = ...
    n_skips_invalid_data_: int = ...
    n_skips_no_inliers_: int = ...
    inlier_mask_: ndarray = ...
    n_trials_: int = ...
    estimator_: Any = ...

    _parameter_constraints: ClassVar[dict] = ...

    def __init__(
        self,
        estimator: Any = None,
        *,
        min_samples: float | None = None,
        residual_threshold: None | Float = None,
        is_data_valid: None | Callable = None,
        is_model_valid: None | Callable = None,
        max_trials: Int = 100,
        max_skips: Int = ...,
        stop_n_inliers: Int = ...,
        stop_score: Float = ...,
        stop_probability: float = 0.99,
        loss: str | Callable = "absolute_error",
        random_state: RandomState | None | Int = None,
        base_estimator: Any = "deprecated",
    ) -> None: ...
    def fit(
        self,
        X: MatrixLike | ArrayLike,
        y: MatrixLike | ArrayLike,
        sample_weight: None | ArrayLike = None,
    ) -> Self: ...
    def predict(self, X: MatrixLike) -> ndarray: ...
    def score(self, X: MatrixLike, y: MatrixLike | ArrayLike) -> float: ...
