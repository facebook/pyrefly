# Authors: Jake Vanderplas <vanderplas@astro.washington.edu>
#          Fabian Pedregosa <fabian.pedregosa@inria.fr>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Sparseness support by Lars Buitinck
#          Multi-output support by Arnaud Joly <a.joly@ulg.ac.be>
#
# License: BSD 3 clause (C) INRIA, University of Amsterdam
from abc import ABCMeta, abstractmethod
from typing import ClassVar, Literal

from numpy import ndarray
from scipy.sparse import spmatrix

from .._typing import ArrayLike, Float, Int, MatrixLike
from ..base import BaseEstimator, MultiOutputMixin

VALID_METRICS = ...

VALID_METRICS_SPARSE = ...

def sort_graph_by_row_values(graph: MatrixLike, copy: bool = False, warn_when_not_sorted: bool = True) -> spmatrix: ...

class NeighborsBase(MultiOutputMixin, BaseEstimator, metaclass=ABCMeta):
    _parameter_constraints: ClassVar[dict] = ...

    @abstractmethod
    def __init__(
        self,
        n_neighbors: None | int = None,
        radius: None | Float = None,
        algorithm: str = "auto",
        leaf_size: int = 30,
        metric: str = "minkowski",
        p: None | int = 2,
        metric_params=None,
        n_jobs: None | int = None,
    ) -> None: ...

class KNeighborsMixin:
    def kneighbors(
        self,
        X: None | MatrixLike = None,
        n_neighbors: None | Int = None,
        return_distance: bool = True,
    ) -> ndarray | tuple[ndarray, ndarray]: ...
    def kneighbors_graph(
        self,
        X: None | MatrixLike = None,
        n_neighbors: None | Int = None,
        mode: Literal["connectivity", "distance"] = "connectivity",
    ) -> spmatrix: ...

class RadiusNeighborsMixin:
    def radius_neighbors(
        self,
        X: None | MatrixLike | list[ndarray] = None,
        radius: None | Float = None,
        return_distance: bool = True,
        sort_results: bool = False,
    ) -> ndarray | tuple[ndarray, ndarray]: ...
    def radius_neighbors_graph(
        self,
        X: None | MatrixLike | ArrayLike = None,
        radius: None | Float = None,
        mode: Literal["connectivity", "distance"] = "connectivity",
        sort_results: bool = False,
    ): ...
