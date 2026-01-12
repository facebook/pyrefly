from typing import Any, ClassVar, Literal
from typing_extensions import Self

from numpy import ndarray
from numpy.random import RandomState

from .._typing import ArrayLike, Float, Int, MatrixLike
from ..base import BaseEstimator, ClusterMixin

###############################################################################
# Public API

def affinity_propagation(
    S: MatrixLike,
    *,
    preference: float | None | ArrayLike = None,
    convergence_iter: Int = 15,
    max_iter: Int = 200,
    damping: Float = 0.5,
    copy: bool = True,
    verbose: bool = False,
    return_n_iter: bool = False,
    random_state: RandomState | None | Int = None,
) -> tuple[ndarray, ndarray, int] | tuple[ndarray, ndarray]: ...

class AffinityPropagation(ClusterMixin, BaseEstimator):
    feature_names_in_: ndarray = ...
    n_features_in_: int = ...
    n_iter_: int = ...
    affinity_matrix_: ndarray = ...
    labels_: ndarray = ...
    cluster_centers_: ndarray = ...
    cluster_centers_indices_: ndarray = ...

    _parameter_constraints: ClassVar[dict] = ...

    def __init__(
        self,
        *,
        damping: Float = 0.5,
        max_iter: Int = 200,
        convergence_iter: Int = 15,
        copy: bool = True,
        preference: float | None | ArrayLike = None,
        affinity: Literal["euclidean", "precomputed"] = "euclidean",
        verbose: bool = False,
        random_state: RandomState | None | Int = None,
    ) -> None: ...
    def fit(self, X: MatrixLike, y: Any = None) -> Self: ...
    def predict(self, X: MatrixLike | ArrayLike) -> ndarray: ...
    def fit_predict(self, X: MatrixLike, y: Any = None) -> ndarray: ...
