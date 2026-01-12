from abc import ABCMeta, abstractmethod
from typing import Any, ClassVar, Literal
from typing_extensions import Self

from numpy import ndarray
from numpy.random import RandomState

from .._typing import ArrayLike, Int, MatrixLike
from ..base import BaseEstimator, BiclusterMixin

__all__ = ["SpectralCoclustering", "SpectralBiclustering"]

class BaseSpectral(BiclusterMixin, BaseEstimator, metaclass=ABCMeta):
    _parameter_constraints: ClassVar[dict] = ...

    @abstractmethod
    def __init__(
        self,
        n_clusters: int | tuple[int, int] = 3,
        svd_method: str = "randomized",
        n_svd_vecs=None,
        mini_batch: bool = False,
        init: str = "k-means++",
        n_init: int = 10,
        random_state: None | int = None,
    ) -> None: ...
    def fit(self, X: MatrixLike, y: Any = None) -> Self: ...

class SpectralCoclustering(BaseSpectral):
    feature_names_in_: ndarray = ...
    n_features_in_: int = ...
    biclusters_: tuple[ndarray, ndarray] = ...
    column_labels_: ArrayLike = ...
    row_labels_: ArrayLike = ...
    columns_: ArrayLike = ...
    rows_: ArrayLike = ...

    _parameter_constraints: ClassVar[dict] = ...

    def __init__(
        self,
        n_clusters: Int = 3,
        *,
        svd_method: Literal["randomized", "arpack"] = "randomized",
        n_svd_vecs: None | Int = None,
        mini_batch: bool = False,
        init: MatrixLike | Literal["k-means++", "random"] = "k-means++",
        n_init: Int = 10,
        random_state: RandomState | None | Int = None,
    ) -> None: ...

class SpectralBiclustering(BaseSpectral):
    feature_names_in_: ndarray = ...
    n_features_in_: int = ...
    biclusters_: tuple[ndarray, ndarray] = ...
    column_labels_: ArrayLike = ...
    row_labels_: ArrayLike = ...
    columns_: ArrayLike = ...
    rows_: ArrayLike = ...

    _parameter_constraints: ClassVar[dict] = ...

    def __init__(
        self,
        n_clusters: tuple[int, int] | int = 3,
        *,
        method: Literal["bistochastic", "scale", "log"] = "bistochastic",
        n_components: Int = 6,
        n_best: Int = 3,
        svd_method: Literal["randomized", "arpack"] = "randomized",
        n_svd_vecs: None | Int = None,
        mini_batch: bool = False,
        init: MatrixLike | Literal["k-means++", "random"] = "k-means++",
        n_init: Int = 10,
        random_state: RandomState | None | Int = None,
    ) -> None: ...
