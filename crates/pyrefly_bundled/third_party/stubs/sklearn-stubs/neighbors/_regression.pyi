from typing import Callable, ClassVar, Literal
from typing_extensions import Self

from numpy import ndarray

from .._typing import ArrayLike, Float, Int, MatrixLike
from ..base import RegressorMixin
from ._base import KNeighborsMixin, NeighborsBase, RadiusNeighborsMixin

# Authors: Jake Vanderplas <vanderplas@astro.washington.edu>
#          Fabian Pedregosa <fabian.pedregosa@inria.fr>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Sparseness support by Lars Buitinck
#          Multi-output support by Arnaud Joly <a.joly@ulg.ac.be>
#          Empty radius support by Andreas Bjerre-Nielsen
#
# License: BSD 3 clause (C) INRIA, University of Amsterdam,
#                           University of Copenhagen

class KNeighborsRegressor(KNeighborsMixin, RegressorMixin, NeighborsBase):
    n_samples_fit_: int = ...
    feature_names_in_: ndarray = ...
    n_features_in_: int = ...
    effective_metric_params_: dict = ...
    effective_metric_: str | Callable = ...

    _parameter_constraints: ClassVar[dict] = ...

    def __init__(
        self,
        n_neighbors: Int = 5,
        *,
        weights: None | Literal["uniform", "distance"] | Callable = "uniform",
        algorithm: Literal["auto", "ball_tree", "kd_tree", "brute"] = "auto",
        leaf_size: Int = 30,
        p: Int = 2,
        metric: str | Callable = "minkowski",
        metric_params: None | dict = None,
        n_jobs: None | Int = None,
    ) -> None: ...
    def fit(self, X: MatrixLike, y: MatrixLike | ArrayLike) -> Self: ...
    def predict(self, X: MatrixLike) -> ndarray: ...

class RadiusNeighborsRegressor(RadiusNeighborsMixin, RegressorMixin, NeighborsBase):
    n_samples_fit_: int = ...
    feature_names_in_: ndarray = ...
    n_features_in_: int = ...
    effective_metric_params_: dict = ...
    effective_metric_: str | Callable = ...

    _parameter_constraints: ClassVar[dict] = ...

    def __init__(
        self,
        radius: Float = 1.0,
        *,
        weights: None | Literal["uniform", "distance"] | Callable = "uniform",
        algorithm: Literal["auto", "ball_tree", "kd_tree", "brute"] = "auto",
        leaf_size: Int = 30,
        p: Int = 2,
        metric: str | Callable = "minkowski",
        metric_params: None | dict = None,
        n_jobs: None | Int = None,
    ) -> None: ...
    def fit(self, X: MatrixLike, y: MatrixLike | ArrayLike) -> Self: ...
    def predict(self, X: MatrixLike) -> ndarray: ...
