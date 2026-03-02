from typing import Any, Callable, ClassVar, Literal
from typing_extensions import Self

from joblib import Memory
from numpy import ndarray

from .._typing import ArrayLike, Float, Int, MatrixLike
from ..base import BaseEstimator, ClusterMixin

class OPTICS(ClusterMixin, BaseEstimator):
    feature_names_in_: ndarray = ...
    n_features_in_: int = ...
    cluster_hierarchy_: ndarray = ...
    predecessor_: ndarray = ...
    core_distances_: ndarray = ...
    ordering_: ndarray = ...
    reachability_: ndarray = ...
    labels_: ndarray = ...

    _parameter_constraints: ClassVar[dict] = ...

    def __init__(
        self,
        *,
        min_samples: float = 5,
        max_eps: Float = ...,
        metric: str | Callable = "minkowski",
        p: Float = 2,
        metric_params: None | dict = None,
        cluster_method: str = "xi",
        eps: None | Float = None,
        xi: float = 0.05,
        predecessor_correction: bool = True,
        min_cluster_size: float | None = None,
        algorithm: Literal["auto", "ball_tree", "kd_tree", "brute"] = "auto",
        leaf_size: Int = 30,
        memory: None | Memory | str = None,
        n_jobs: None | Int = None,
    ) -> None: ...
    def fit(self, X: MatrixLike, y: Any = None) -> Self: ...

def compute_optics_graph(
    X: MatrixLike,
    *,
    min_samples: float,
    max_eps: Float,
    metric: str | Callable,
    p: Int,
    metric_params: dict,
    algorithm: Literal["auto", "ball_tree", "kd_tree", "brute"],
    leaf_size: Int,
    n_jobs: Int,
) -> tuple[ndarray, ndarray, ndarray, ndarray]: ...
def cluster_optics_dbscan(
    *,
    reachability: ArrayLike,
    core_distances: ArrayLike,
    ordering: ArrayLike,
    eps: Float,
) -> ndarray: ...
def cluster_optics_xi(
    *,
    reachability: ArrayLike,
    predecessor: ArrayLike,
    ordering: ArrayLike,
    min_samples: float,
    min_cluster_size: float | None = None,
    xi: float = 0.05,
    predecessor_correction: bool = True,
) -> tuple[ndarray, ndarray]: ...
