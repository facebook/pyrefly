from typing import Any, ClassVar, Literal
from typing_extensions import Self

from numpy import ndarray
from numpy.random import RandomState

from .._typing import ArrayLike, Int, MatrixLike
from ..base import OutlierMixin
from ..tree import ExtraTreeRegressor
from ._bagging import BaseBagging

# Authors: Nicolas Goix <nicolas.goix@telecom-paristech.fr>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
# License: BSD 3 clause

__all__ = ["IsolationForest"]

class IsolationForest(OutlierMixin, BaseBagging):
    feature_names_in_: ndarray = ...
    n_features_in_: int = ...
    offset_: float = ...
    max_samples_: int = ...
    estimators_samples_: list[ndarray] = ...
    estimators_features_: list[ndarray] = ...
    estimators_: list[ExtraTreeRegressor] = ...
    base_estimator_: ExtraTreeRegressor = ...
    estimator_: ExtraTreeRegressor = ...

    _parameter_constraints: ClassVar[dict] = ...

    def __init__(
        self,
        *,
        n_estimators: Int = 100,
        max_samples: float | Literal["auto"] = "auto",
        contamination: float | str = "auto",
        max_features: float = 1.0,
        bootstrap: bool = False,
        n_jobs: None | Int = None,
        random_state: RandomState | None | Int = None,
        verbose: Int = 0,
        warm_start: bool = False,
    ) -> None: ...
    def fit(
        self,
        X: MatrixLike | ArrayLike,
        y: Any = None,
        sample_weight: None | ArrayLike = None,
    ) -> Self: ...
    def predict(self, X: MatrixLike | ArrayLike) -> ndarray: ...
    def decision_function(self, X: MatrixLike | ArrayLike) -> ndarray: ...
    def score_samples(self, X: MatrixLike | ArrayLike) -> ndarray: ...
