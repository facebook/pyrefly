from typing import Any, Callable, Literal
from typing_extensions import Self

from numpy import ndarray

from .._typing import Float, Int, MatrixLike
from ._base import KNeighborsMixin, NeighborsBase, RadiusNeighborsMixin

class NearestNeighbors(KNeighborsMixin, RadiusNeighborsMixin, NeighborsBase):
    n_samples_fit_: int = ...
    feature_names_in_: ndarray = ...
    n_features_in_: int = ...
    effective_metric_params_: dict = ...
    effective_metric_: str = ...

    def __init__(
        self,
        *,
        n_neighbors: Int = 5,
        radius: Float = 1.0,
        algorithm: Literal["auto", "ball_tree", "kd_tree", "brute"] = "auto",
        leaf_size: Int = 30,
        metric: str | Callable = "minkowski",
        p: Float = 2,
        metric_params: None | dict = None,
        n_jobs: None | Int = None,
    ) -> None: ...
    def fit(self, X: MatrixLike, y: Any = None) -> Self: ...
