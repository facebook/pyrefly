from typing import Any, Callable, ClassVar, Literal
from typing_extensions import Self

from numpy import ndarray

from .._typing import ArrayLike, Float, Int, MatrixLike
from ..base import BaseEstimator, ClusterMixin

# Author: Robert Layton <robertlayton@gmail.com>
#         Joel Nothman <joel.nothman@gmail.com>
#         Lars Buitinck
#
# License: BSD 3 clause

def dbscan(
    X: MatrixLike,
    eps: Float = 0.5,
    *,
    min_samples: Int = 5,
    metric: str | Callable = "minkowski",
    metric_params: None | dict = None,
    algorithm: Literal["auto", "ball_tree", "kd_tree", "brute"] = "auto",
    leaf_size: Int = 30,
    p: Float = 2,
    sample_weight: None | ArrayLike = None,
    n_jobs: None | Int = None,
) -> tuple[ndarray, ndarray]: ...

class DBSCAN(ClusterMixin, BaseEstimator):
    feature_names_in_: ndarray = ...
    n_features_in_: int = ...
    labels_: ndarray = ...
    components_: ndarray = ...
    core_sample_indices_: ndarray = ...

    _parameter_constraints: ClassVar[dict] = ...

    def __init__(
        self,
        eps: Float = 0.5,
        *,
        min_samples: Int = 5,
        metric: str | Callable = "euclidean",
        metric_params: None | dict = None,
        algorithm: Literal["auto", "ball_tree", "kd_tree", "brute"] = "auto",
        leaf_size: Int = 30,
        p: None | Float = None,
        n_jobs: None | Int = None,
    ) -> None: ...
    def fit(
        self,
        X: MatrixLike,
        y: Any = None,
        sample_weight: None | ArrayLike = None,
    ) -> Self: ...
    def fit_predict(self, X: MatrixLike, y: Any = None, sample_weight: None | ArrayLike = None) -> ndarray: ...
