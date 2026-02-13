from collections.abc import Mapping
from typing import Any, Callable, ClassVar, Literal
from typing_extensions import Self

from numpy import ndarray
from numpy.random import RandomState
from scipy.sparse._coo import coo_matrix

from .._typing import ArrayLike, Float, Int, MatrixLike
from ..base import BaseEstimator, ClusterMixin

def cluster_qr(vectors: MatrixLike) -> ndarray: ...
def discretize(
    vectors: MatrixLike,
    *,
    copy: bool = True,
    max_svd_restarts: Int = 30,
    n_iter_max: Int = 20,
    random_state: RandomState | None | Int = None,
) -> ndarray: ...
def spectral_clustering(
    affinity: coo_matrix | MatrixLike,
    *,
    n_clusters: Int = 8,
    n_components: None | Int = None,
    eigen_solver: Literal["arpack", "lobpcg", "amg"] | None = None,
    random_state: RandomState | None | Int = None,
    n_init: Int = 10,
    eigen_tol: str | Float = "auto",
    assign_labels: Literal["kmeans", "discretize", "cluster_qr"] = "kmeans",
    verbose: bool = False,
) -> ndarray: ...

class SpectralClustering(ClusterMixin, BaseEstimator):
    feature_names_in_: ndarray = ...
    n_features_in_: int = ...
    labels_: ndarray = ...
    affinity_matrix_: ArrayLike = ...

    _parameter_constraints: ClassVar[dict] = ...

    def __init__(
        self,
        n_clusters: Int = 8,
        *,
        eigen_solver: None | Literal["arpack", "lobpcg", "amg"] = None,
        n_components: None | Int = None,
        random_state: RandomState | None | Int = None,
        n_init: Int = 10,
        gamma: Float = 1.0,
        affinity: str | Callable = "rbf",
        n_neighbors: Int = 10,
        eigen_tol: str | Float = "auto",
        assign_labels: Literal["kmeans", "discretize", "cluster_qr"] = "kmeans",
        degree: Float = 3,
        coef0: Float = 1,
        kernel_params: None | Mapping[str, Any] = None,
        n_jobs: None | Int = None,
        verbose: bool = False,
    ) -> None: ...
    def fit(self, X: MatrixLike, y: Any = None) -> Self: ...
    def fit_predict(self, X: MatrixLike, y: Any = None) -> ndarray: ...
