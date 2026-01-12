from typing import Any, Callable, ClassVar, Literal
from typing_extensions import Self

from numpy import ndarray
from scipy.sparse import spmatrix

from .._typing import Float, Int, MatrixLike
from ..base import ClassNamePrefixFeaturesOutMixin, TransformerMixin
from ._ball_tree import BallTree
from ._base import KNeighborsMixin, NeighborsBase, RadiusNeighborsMixin
from ._unsupervised import NearestNeighbors

def kneighbors_graph(
    X: MatrixLike | BallTree | NearestNeighbors,
    n_neighbors: Int,
    *,
    mode: Literal["connectivity", "distance"] = "connectivity",
    metric: str = "minkowski",
    p: Int = 2,
    metric_params: None | dict = None,
    include_self: str | bool = False,
    n_jobs: None | Int = None,
) -> spmatrix: ...
def radius_neighbors_graph(
    X: MatrixLike | BallTree,
    radius: Float,
    *,
    mode: Literal["connectivity", "distance"] = "connectivity",
    metric: str = "minkowski",
    p: Int = 2,
    metric_params: None | dict = None,
    include_self: str | bool = False,
    n_jobs: None | Int = None,
): ...

class KNeighborsTransformer(ClassNamePrefixFeaturesOutMixin, KNeighborsMixin, TransformerMixin, NeighborsBase):
    n_samples_fit_: int = ...
    feature_names_in_: ndarray = ...
    n_features_in_: int = ...
    effective_metric_params_: dict = ...
    effective_metric_: str | Callable = ...

    _parameter_constraints: ClassVar[dict] = ...

    def __init__(
        self,
        *,
        mode: Literal["distance", "connectivity"] = "distance",
        n_neighbors: Int = 5,
        algorithm: Literal["auto", "ball_tree", "kd_tree", "brute"] = "auto",
        leaf_size: Int = 30,
        metric: str | Callable = "minkowski",
        p: Int = 2,
        metric_params: None | dict = None,
        n_jobs: None | Int = None,
    ) -> None: ...
    def fit(self, X: MatrixLike, y: Any = None) -> Self: ...
    def transform(self, X: MatrixLike) -> spmatrix: ...
    def fit_transform(self, X: MatrixLike, y: Any = None) -> spmatrix: ...

class RadiusNeighborsTransformer(
    ClassNamePrefixFeaturesOutMixin,
    RadiusNeighborsMixin,
    TransformerMixin,
    NeighborsBase,
):
    n_samples_fit_: int = ...
    feature_names_in_: ndarray = ...
    n_features_in_: int = ...
    effective_metric_params_: dict = ...
    effective_metric_: str | Callable = ...

    _parameter_constraints: ClassVar[dict] = ...

    def __init__(
        self,
        *,
        mode: Literal["distance", "connectivity"] = "distance",
        radius: Float = 1.0,
        algorithm: Literal["auto", "ball_tree", "kd_tree", "brute"] = "auto",
        leaf_size: Int = 30,
        metric: str | Callable = "minkowski",
        p: Int = 2,
        metric_params: None | dict = None,
        n_jobs: None | Int = None,
    ) -> None: ...
    def fit(self, X: MatrixLike, y: Any = None) -> Self: ...
    def transform(self, X: MatrixLike): ...
    def fit_transform(self, X: MatrixLike, y: Any = None): ...
