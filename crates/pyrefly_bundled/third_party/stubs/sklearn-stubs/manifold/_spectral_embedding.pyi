from typing import Any, Callable, ClassVar, Literal
from typing_extensions import Self

from numpy import ndarray
from numpy.random import RandomState
from scipy.sparse._coo import coo_matrix

from .._typing import ArrayLike, Float, Int, MatrixLike
from ..base import BaseEstimator

def spectral_embedding(
    adjacency: coo_matrix | MatrixLike,
    *,
    n_components: Int = 8,
    eigen_solver: None | Literal["arpack", "lobpcg", "amg"] = None,
    random_state: RandomState | None | Int = None,
    eigen_tol: str | Float = "auto",
    norm_laplacian: bool = True,
    drop_first: bool = True,
) -> ndarray: ...

class SpectralEmbedding(BaseEstimator):
    n_neighbors_: int = ...
    feature_names_in_: ndarray = ...
    n_features_in_: int = ...
    affinity_matrix_: ndarray = ...
    embedding_: ndarray = ...

    _parameter_constraints: ClassVar[dict] = ...

    def __init__(
        self,
        n_components: Int = 2,
        *,
        affinity: (
            Literal["nearest_neighbors", "rbf", "precomputed", "precomputed_nearest_neighbors"] | Callable
        ) = "nearest_neighbors",
        gamma: None | Float = None,
        random_state: RandomState | None | Int = None,
        eigen_solver: None | Literal["arpack", "lobpcg", "amg"] = None,
        eigen_tol: str | Float = "auto",
        n_neighbors: None | Int = None,
        n_jobs: None | Int = None,
    ) -> None: ...
    def fit(self, X: MatrixLike | ArrayLike, y: Any = None) -> Self: ...
    def fit_transform(self, X: MatrixLike | ArrayLike, y: Any = None) -> ndarray: ...
