from collections.abc import Iterable
from typing import Any, ClassVar, Literal
from typing_extensions import Self

from numpy import ndarray

from .._typing import ArrayLike, Float, Int, MatrixLike
from ..model_selection import BaseCrossValidator
from . import EmpiricalCovariance

# Author: Gael Varoquaux <gael.varoquaux@normalesup.org>
# License: BSD 3 clause
# Copyright: INRIA

def alpha_max(emp_cov: MatrixLike) -> Float: ...

# The g-lasso algorithm
def graphical_lasso(
    emp_cov: MatrixLike,
    alpha: Float,
    *,
    cov_init: None | MatrixLike = None,
    mode: Literal["cd", "lars"] = "cd",
    tol: Float = 1e-4,
    enet_tol: Float = 1e-4,
    max_iter: Int = 100,
    verbose: bool = False,
    return_costs: bool = False,
    eps: Float = ...,
    return_n_iter: bool = False,
) -> tuple[ndarray, ndarray, list[tuple], int]: ...

class BaseGraphicalLasso(EmpiricalCovariance):
    feature_names_in_: ndarray = ...
    n_features_in_: int = ...
    precision_: ndarray = ...
    covariance_: ndarray = ...
    location_: ndarray = ...
    _parameter_constraints: ClassVar[dict] = ...

    def __init__(
        self,
        tol: float = 1e-4,
        enet_tol: float = 1e-4,
        max_iter: int = 100,
        mode: str = "cd",
        verbose: bool = False,
        assume_centered: bool = False,
    ) -> None: ...

class GraphicalLasso(BaseGraphicalLasso):
    feature_names_in_: ndarray = ...
    n_features_in_: int = ...
    n_iter_: int = ...
    precision_: ndarray = ...
    covariance_: ndarray = ...
    location_: ndarray = ...

    _parameter_constraints: ClassVar[dict] = ...

    def __init__(
        self,
        alpha: Float = 0.01,
        *,
        mode: Literal["cd", "lars"] = "cd",
        tol: Float = 1e-4,
        enet_tol: Float = 1e-4,
        max_iter: Int = 100,
        verbose: bool = False,
        assume_centered: bool = False,
    ) -> None: ...
    def fit(self, X: MatrixLike, y: Any = None) -> Self: ...

# Cross-validation with GraphicalLasso
def graphical_lasso_path(
    X: ArrayLike,
    alphas: ArrayLike,
    cov_init: None | MatrixLike = None,
    X_test: None | MatrixLike = None,
    mode: Literal["cd", "lars"] = "cd",
    tol: Float = 1e-4,
    enet_tol: Float = 1e-4,
    max_iter: Int = 100,
    verbose: int | bool = False,
) -> tuple[list[ndarray], list[ndarray], list[Float]] | tuple[list[ndarray], list[ndarray], list[float]]: ...

class GraphicalLassoCV(BaseGraphicalLasso):
    feature_names_in_: ndarray = ...
    n_features_in_: int = ...
    n_iter_: int = ...
    cv_results_: dict[str, ndarray] = ...
    alpha_: float = ...
    precision_: ndarray = ...
    covariance_: ndarray = ...
    location_: ndarray = ...

    _parameter_constraints: ClassVar[dict] = ...

    def __init__(
        self,
        *,
        alphas: ArrayLike | int = 4,
        n_refinements: Int = 4,
        cv: int | BaseCrossValidator | Iterable | None = None,
        tol: Float = 1e-4,
        enet_tol: Float = 1e-4,
        max_iter: Int = 100,
        mode: Literal["cd", "lars"] = "cd",
        n_jobs: None | Int = None,
        verbose: bool = False,
        assume_centered: bool = False,
    ) -> None: ...
    def fit(self, X: MatrixLike, y: Any = None) -> Self: ...
