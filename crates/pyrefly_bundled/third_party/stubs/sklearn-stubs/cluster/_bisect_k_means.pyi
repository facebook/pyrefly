from collections.abc import Iterator
from typing import Any, Callable, ClassVar, Literal
from typing_extensions import Self

from numpy import ndarray
from numpy.random import RandomState

from .._typing import ArrayLike, Float, Int, MatrixLike
from ._kmeans import _BaseKMeans

# Author: Michal Krawczyk <mkrwczyk.1@gmail.com>

class _BisectingTree:
    def __init__(self, center: ndarray, indices: ndarray, score: Float) -> None: ...
    def split(self, labels: ndarray, centers: ndarray, scores: ndarray) -> None: ...
    def get_cluster_to_bisect(self) -> _BisectingTree: ...
    def iter_leaves(self) -> Iterator[_BisectingTree]: ...

class BisectingKMeans(_BaseKMeans):
    feature_names_in_: ndarray = ...
    n_features_in_: int = ...
    inertia_: float = ...
    labels_: ndarray = ...
    cluster_centers_: ndarray = ...

    _parameter_constraints: ClassVar[dict] = ...

    def __init__(
        self,
        n_clusters: Int = 8,
        *,
        init: Literal["k-means++", "random"] | Callable = "random",
        n_init: Int = 1,
        random_state: RandomState | None | Int = None,
        max_iter: Int = 300,
        verbose: Int = 0,
        tol: Float = 1e-4,
        copy_x: bool = True,
        algorithm: Literal["lloyd", "elkan"] = "lloyd",
        bisecting_strategy: Literal["biggest_inertia", "largest_cluster"] = "biggest_inertia",
    ) -> None: ...
    def fit(
        self,
        X: MatrixLike | ArrayLike,
        y: Any = None,
        sample_weight: None | ArrayLike = None,
    ) -> Self: ...
    def predict(self, X: MatrixLike | ArrayLike) -> ndarray: ...
