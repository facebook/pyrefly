from abc import ABC
from typing import Any, Callable, ClassVar, Literal
from typing_extensions import Self

from numpy import ndarray
from numpy.random import RandomState

from .._typing import ArrayLike, Float, Int, MatrixLike
from ..base import BaseEstimator, ClassNamePrefixFeaturesOutMixin, ClusterMixin, TransformerMixin

###############################################################################
# Initialization heuristic

def kmeans_plusplus(
    X: MatrixLike | ArrayLike,
    n_clusters: Int,
    *,
    x_squared_norms: None | ArrayLike = None,
    random_state: None | RandomState | int = None,
    n_local_trials: None | Int = None,
) -> tuple[ndarray, ndarray]: ...
def k_means(
    X: MatrixLike | ArrayLike,
    n_clusters: Int,
    *,
    sample_weight: None | ArrayLike = None,
    init: MatrixLike | Callable | Literal["k-means++", "random"] = "k-means++",
    n_init: Literal["auto", "warn"] | int = "warn",
    max_iter: Int = 300,
    verbose: bool = False,
    tol: Float = 1e-4,
    random_state: RandomState | None | Int = None,
    copy_x: bool = True,
    algorithm: Literal["lloyd", "elkan", "auto", "full"] = "lloyd",
    return_n_iter: bool = False,
) -> tuple[ndarray, ndarray, float] | tuple[ndarray, ndarray, float, int]: ...

class _BaseKMeans(ClassNamePrefixFeaturesOutMixin, TransformerMixin, ClusterMixin, BaseEstimator, ABC):
    _parameter_constraints: ClassVar[dict] = ...

    def __init__(
        self,
        n_clusters: Int,
        *,
        init,
        n_init,
        max_iter,
        tol,
        verbose,
        random_state,
    ) -> None: ...
    def fit_predict(
        self,
        X: MatrixLike | ArrayLike,
        y: Any = None,
        sample_weight: None | ArrayLike = None,
    ) -> ndarray: ...
    def predict(self, X: MatrixLike | ArrayLike, sample_weight: None | ArrayLike = None) -> ndarray: ...
    def fit_transform(
        self,
        X: MatrixLike | ArrayLike,
        y: Any = None,
        sample_weight: None | ArrayLike = None,
    ) -> ndarray: ...
    def transform(self, X: MatrixLike | ArrayLike) -> ndarray: ...
    def score(
        self,
        X: MatrixLike | ArrayLike,
        y: Any = None,
        sample_weight: None | ArrayLike = None,
    ) -> float: ...

class KMeans(_BaseKMeans):
    feature_names_in_: ndarray = ...
    n_features_in_: int = ...
    n_iter_: int = ...
    inertia_: float = ...
    labels_: ndarray = ...
    cluster_centers_: ndarray = ...

    _parameter_constraints: ClassVar[dict] = ...

    def __init__(
        self,
        n_clusters: Int = 8,
        *,
        init: MatrixLike | Callable | Literal["k-means++", "random"] = "k-means++",
        n_init: Literal["auto", "warn"] | int = "warn",
        max_iter: Int = 300,
        tol: Float = 1e-4,
        verbose: Int = 0,
        random_state: RandomState | None | Int = None,
        copy_x: bool = True,
        algorithm: Literal["lloyd", "elkan", "auto", "full"] = "lloyd",
    ) -> None: ...
    def fit(
        self,
        X: MatrixLike | ArrayLike,
        y: Any = None,
        sample_weight: None | ArrayLike = None,
    ) -> Self: ...

class MiniBatchKMeans(_BaseKMeans):
    feature_names_in_: ndarray = ...
    n_features_in_: int = ...
    n_steps_: int = ...
    n_iter_: int = ...
    inertia_: float = ...
    labels_: ndarray = ...
    cluster_centers_: ndarray = ...

    _parameter_constraints: ClassVar[dict] = ...

    def __init__(
        self,
        n_clusters: Int = 8,
        *,
        init: MatrixLike | Callable | Literal["k-means++", "random"] = "k-means++",
        max_iter: Int = 100,
        batch_size: Int = 1024,
        verbose: Int = 0,
        compute_labels: bool = True,
        random_state: RandomState | None | Int = None,
        tol: Float = 0.0,
        max_no_improvement: Int = 10,
        init_size: None | Int = None,
        n_init: Literal["auto", "warn"] | int = "warn",
        reassignment_ratio: Float = 0.01,
    ) -> None: ...
    def fit(
        self,
        X: MatrixLike | ArrayLike,
        y: Any = None,
        sample_weight: None | ArrayLike = None,
    ) -> Self: ...
    def partial_fit(
        self,
        X: MatrixLike | ArrayLike,
        y: Any = None,
        sample_weight: None | ArrayLike = None,
    ) -> Self: ...
